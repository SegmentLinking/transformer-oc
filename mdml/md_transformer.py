"""
MDTransformerNet: a tracking algorithm from outer-tracker MDs to track candidates.

A transformer (MDTransformerModule) sees the MDs of one (eta, phi) patch at a time and predicts, for each MD,
whether it is fake and the kinematics of its sim, SIM_KINEMATICS = (q/pT, eta, cos phi, sin phi), and its sim's
production vertex (sim_vz, sim_vtxperp; trained only if sim_vz_weight, sim_vtxperp_weight > 0). Optionally
(embed_dim > 0) it also predicts a learned embedding, trained with a contrastive loss to put MDs of one sim close.

- Patches: a patch is the MDs within half-widths (patch_eta, patch_phi) of a centre. Around it, a margin of
  half-widths (margin_eta, margin_phi) adds context: margin MDs are seen by attention but are not in the loss or
  the predictions. phi is relative to the patch centre in the inputs and targets, so the network is phi-symmetric
  unless absolute_phi also gives it each MD's absolute phi.
- Attention: an MD attends to another only if their |deta| and |dphi| are within the window of their layer pair:
  a quantile of the |deta| and |dphi| of same-sim MD pairs in training events (LayerPairWindows.from_events).
- Inference: a patch grid (patches with margin 0) covers the event, so each MD is in exactly one grid patch; each
  grid patch gets its margin and goes through the network.
- Decoding: MDs with p(fake) >= fake_cut are dropped, DBSCAN (min_samples=MIN_MDS) clusters the rest with radius
  eps in cluster_space (the predicted SIM_KINEMATICS, or the embedding), and each cluster is a track candidate.
  The embedding is only trained within a patch, so clustering it over the whole event is questionable.

MDTransformerNet owns the network and the working point (fake_cut, eps). MDTransformerLightningModule trains the
network. All settings, including the run's paths, are in the config yaml (see md_transformer.yaml).

From the repo root, train, choose the working point and log the per-MD quality into one new run directory
(on phi3, data/ is bound to /data as on NRP):
apptainer exec --nv -B data:/data ml-tracking.sif python3 -m mdml.md_transformer --config mdml/md_transformer.yaml

The run directory is output_dir/<UTC time as YYYY_MM_DD_hhmmss>, and must be new. It gets train.log (every
line printed here, see print_and_log), hparams.yaml (the config, with run_dir) and metrics.csv (Lightning's
CSVLogger), Lightning checkpoints, and md_transformer.pt (the whole algorithm, see MDTransformerNet.save).
"""

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from sklearn.neighbors import NearestNeighbors
from torch import nn

from mdml.clustering import MIN_MDS, find_mdclusters
from mdml.data import (
    MD_FEATURES,
    SIM_FEATURES,
    MDsFromOneEvent,
    event_paths,
    read_event,
)
from mdml.trackcands import TrackCandidatesFromOneEvent, trackcands_from_mdclusters

# LST outer-tracker layers 1-6 (barrel) and 7-11 (endcap), one-hot encoded
FIRST_LAYER = 1
N_LAYERS = 11
# Fixed input scales, about the largest |value| of each feature. No absolute phi here: see absolute_phi
FEATURE_SCALES = {
    "rt": 110.0,  # cm
    "z": 270.0,  # cm
    "drt": 5.0,  # cm
    "dphi": 0.01,
    "dz": 5.0,  # cm
    "eta": 2.5,
    "qoverpt": 1.0,  # GeV^-1
}
# + layer one-hot, + (deta, dphi) to the patch centre scaled by the half-widths of patch + margin
N_INPUTS = N_LAYERS + len(FEATURE_SCALES) + 2
# absolute_phi: + each MD's absolute (cos phi, sin phi)
N_ABSOLUTE_PHI_INPUTS = 2
SIM_KINEMATICS = ["sim_qoverpt", "sim_eta", "sim_cosphi", "sim_sinphi"]
# Production vertex of each MD's sim, regressed as asinh(sim_vz / scale) and log1p(sim_vtxperp / scale): linear
# within about one scale (the luminous region, ~5 cm in z), logarithmic in the tails of secondaries (to ~2 m).
# asinh keeps the sign of sim_vz. Not in cluster_space: the resolution is unknown
SIM_VZ_SCALE = 5.0  # cm
SIM_VTXPERP_SCALE = 1.0  # cm
# One per vertex output: (asinh(sim_vz / SIM_VZ_SCALE), log1p(sim_vtxperp / SIM_VTXPERP_SCALE))
SIM_VERTEX_SCALES = [SIM_VZ_SCALE, SIM_VTXPERP_SCALE]
# The patch grid covers |eta| <= MD_ETA_MAX; the largest |md eta| seen is 2.78
MD_ETA_MAX = 2.8
# Layer pairs with fewer same-sim MD pairs get a zero-width window (no attention between them)
MIN_WINDOW_PAIRS = 50
# Gradients are rescaled so their total norm is at most this
GRADIENT_CLIP_NORM = 1.0


def main() -> None:
    config = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    run_name = time.strftime("%Y_%m_%d_%H%M%S", time.gmtime())
    config["run_dir"] = os.path.join(config["output_dir"], run_name)
    # exist_ok=False: a second run started in the same second fails instead of mixing into this one
    os.makedirs(config["run_dir"], exist_ok=False)
    print_and_log(
        line=f"[run] run_dir={config['run_dir']}",
        config=config,
    )
    torch.set_num_threads(config["cpu_threads"])
    pl.seed_everything(config["seed"])
    train_paths = event_paths(pt_dir=config["train_dir"])[: config["n_train_events"]]
    # The last validation events: the harness evaluates the first validation file
    val_paths = event_paths(pt_dir=config["val_dir"])[-config["n_val_events"] :]
    windows = LayerPairWindows.from_events(
        paths=train_paths[: config["window_n_events"]],
        config=config,
    )
    algo = MDTransformerNet(config=config, windows=windows)
    algo.train(train_paths=train_paths, val_paths=val_paths)
    # The working point is chosen on validation events, never on training events
    algo.choose_fake_cut(paths=val_paths)
    algo.choose_eps(paths=val_paths)
    algo.save(path=os.path.join(config["run_dir"], "md_transformer.pt"))
    log_md_quality(algo=algo, paths=val_paths)


def parse_args() -> dict:
    """The config yaml as a dict; the only command-line argument is its path."""
    parser = argparse.ArgumentParser(
        description="Train MDTransformerNet and choose its working point."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="config yaml, e.g. mdml/md_transformer.yaml",
    )
    args = parser.parse_args()
    with open(args.config) as cfg:
        return yaml.safe_load(cfg)


def print_and_log(line: str, config: dict) -> None:
    """Print line, and append it to train.log in the run directory, so train.log has every line this script prints."""
    print(line, flush=True)
    with open(os.path.join(config["run_dir"], "train.log"), "a") as log_file:
        log_file.write(line + "\n")


# ---------------------------------------------------------------- attention windows


@dataclass
class LayerPairWindows:
    """
    Attention half-widths per layer pair: symmetric (N_LAYERS, N_LAYERS) tensors, indexed by layer - FIRST_LAYER.
    """

    eta: torch.Tensor
    phi: torch.Tensor

    @classmethod
    def from_events(cls, paths: list[str], config: dict) -> "LayerPairWindows":
        """
        For each layer pair, quantile window_quantile of |deta| and |dphi| over the same-sim MD pairs (sim pT >=
        pt_min) in paths, times window_scale. Pairs with fewer than MIN_WINDOW_PAIRS get width 0. Logs one
        [window] line per layer pair.
        """
        # One row per same-sim MD pair: (lower layer, higher layer) and (|deta|, |dphi|)
        pair_layers, pair_absDeltas = [], []
        start = time.time()
        for idx, path in enumerate(paths):
            print_and_log(
                line=f"[windows] event={idx + 1} n_events={len(paths)} seconds={time.time() - start:.0f}",
                config=config,
            )
            mdi = MDInputs.from_mds(mds=read_event(path=path), pt_min=config["pt_min"])
            md_isReal = ~mdi.md_simIsFake
            for sim in mdi.md_simIdx[md_isReal].unique().tolist():
                md_idx = torch.nonzero(md_isReal & (mdi.md_simIdx == sim)).squeeze(1)
                first, second = torch.triu_indices(len(md_idx), len(md_idx), offset=1)
                md_a, md_b = md_idx[first], md_idx[second]
                layer_a, layer_b = mdi.md_layer[md_a], mdi.md_layer[md_b]
                pair_layers.append(
                    torch.stack(
                        [
                            torch.minimum(layer_a, layer_b),
                            torch.maximum(layer_a, layer_b),
                        ],
                        dim=1,
                    )
                )
                pair_absDeltas.append(
                    torch.stack(
                        [
                            (mdi.md_eta[md_a] - mdi.md_eta[md_b]).abs(),
                            wrap_phi(mdi.md_phi[md_a] - mdi.md_phi[md_b]).abs(),
                        ],
                        dim=1,
                    )
                )
        pair_layers = torch.cat(pair_layers)
        pair_absDeltas = torch.cat(pair_absDeltas)
        print_and_log(
            line=(
                f"[windows] n_events={len(paths)} window_quantile={config['window_quantile']} "
                f"window_scale={config['window_scale']}"
            ),
            config=config,
        )
        eta = torch.zeros(N_LAYERS, N_LAYERS)
        phi = torch.zeros(N_LAYERS, N_LAYERS)
        for low in range(N_LAYERS):
            for high in range(low, N_LAYERS):
                abs_deltas = pair_absDeltas[
                    (pair_layers[:, 0] == low) & (pair_layers[:, 1] == high)
                ]
                if len(abs_deltas) >= MIN_WINDOW_PAIRS:
                    eta_window, phi_window = config["window_scale"] * torch.quantile(
                        abs_deltas,
                        q=config["window_quantile"],
                        dim=0,
                    )
                    eta[low, high] = eta[high, low] = eta_window
                    phi[low, high] = phi[high, low] = phi_window
                print_and_log(
                    line=(
                        f"[window] layer_low={low + FIRST_LAYER} layer_high={high + FIRST_LAYER} "
                        f"n_pairs={len(abs_deltas)} eta={eta[low, high]:.4f} phi={phi[low, high]:.4f}"
                    ),
                    config=config,
                )
        return cls(eta=eta, phi=phi)


# ---------------------------------------------------------------- data


@dataclass
class MDInputs:
    """
    One event's MDs prepared for the network (n_md rows each): the per-MD inputs that patches are cut from, the
    (eta, phi) that patches select on, and the training targets.
    """

    # (n_md, N_LAYERS + len(FEATURE_SCALES)): layer one-hot, scaled features
    md_inputs: torch.Tensor
    md_eta: torch.Tensor
    md_phi: torch.Tensor
    md_layer: torch.Tensor  # layer - FIRST_LAYER
    md_simIdx: torch.Tensor
    md_simIsFake: torch.Tensor  # no sim, or sim pT < pt_min
    # (n_md, 3): sim q/pT, eta, phi; nan where md_simIdx < 0
    md_simKinematics: torch.Tensor
    # (n_md, 2): asinh(sim_vz / SIM_VZ_SCALE), log1p(sim_vtxperp / SIM_VTXPERP_SCALE); nan where md_simIdx < 0
    md_simVertex: torch.Tensor

    @classmethod
    def from_mds(cls, mds: MDsFromOneEvent, pt_min: float) -> "MDInputs":
        """The MDs of mds, with layer one-hot encoded and features divided by FEATURE_SCALES."""
        md_features = mds.md_features
        md_simFeatures = mds.md_simFeatures
        md_layer = md_features[:, MD_FEATURES.index("layer")].long() - FIRST_LAYER
        md_scaled = [
            md_features[:, MD_FEATURES.index(name)] / scale
            for name, scale in FEATURE_SCALES.items()
        ]
        md_simPt = md_simFeatures[:, SIM_FEATURES.index("sim_pt")]
        md_simKinematics = torch.stack(
            [
                md_simFeatures[:, SIM_FEATURES.index("sim_q")] / md_simPt,
                md_simFeatures[:, SIM_FEATURES.index("sim_eta")],
                md_simFeatures[:, SIM_FEATURES.index("sim_phi")],
            ],
            dim=1,
        )
        md_simVertex = torch.stack(
            [
                torch.asinh(
                    md_simFeatures[:, SIM_FEATURES.index("sim_vz")] / SIM_VZ_SCALE
                ),
                torch.log1p(
                    md_simFeatures[:, SIM_FEATURES.index("sim_vtxperp")]
                    / SIM_VTXPERP_SCALE
                ),
            ],
            dim=1,
        )
        # The .pt files fill every sim feature with -999 when there is no sim; nan makes any use of them obvious
        md_simKinematics[mds.md_simIdx < 0] = torch.nan
        md_simVertex[mds.md_simIdx < 0] = torch.nan
        return cls(
            md_inputs=torch.cat(
                [
                    nn.functional.one_hot(md_layer, N_LAYERS).float(),
                    torch.stack(md_scaled, dim=1),
                ],
                dim=1,
            ),
            md_eta=md_features[:, MD_FEATURES.index("eta")],
            md_phi=torch.atan2(
                md_features[:, MD_FEATURES.index("sinphi")],
                md_features[:, MD_FEATURES.index("cosphi")],
            ),
            md_layer=md_layer,
            md_simIdx=mds.md_simIdx,
            md_simIsFake=(mds.md_simIdx < 0) | (md_simPt < pt_min),
            md_simKinematics=md_simKinematics,
            md_simVertex=md_simVertex,
        )


@dataclass
class Patch:
    """The MDs of one patch and its margin, as indices into the event and the tensors the network and loss need."""

    md_idx: torch.Tensor  # (n,): index of each MD in the event
    md_inputs: torch.Tensor  # (n, n_inputs)
    md_layer: torch.Tensor
    # (n, 2): (deta, dphi) to the centre, unscaled, for the attention windows
    md_pos: torch.Tensor
    # In the margin: context only, not in the loss or predictions
    md_isMargin: torch.Tensor
    md_simIsFake: torch.Tensor
    md_simIdx: torch.Tensor
    md_simKinematics: torch.Tensor  # (n, 4): SIM_KINEMATICS, phi relative to the centre
    md_simVertex: torch.Tensor  # (n, 2): as MDInputs.md_simVertex

    @classmethod
    def from_mdinputs(
        cls,
        mdi: MDInputs,
        eta_centre: float,
        phi_centre: float,
        config: dict,
    ) -> "Patch":
        """
        The MDs of mdi within half-widths (patch_eta + margin_eta, patch_phi + margin_phi) of the centre; those
        outside (patch_eta, patch_phi) are the margin. Inputs: mdi.md_inputs, then (deta, dphi) to the centre
        divided by the half-widths of patch + margin, then (if absolute_phi) each MD's absolute (cos phi, sin phi).
        """
        eta_half = config["patch_eta"] + config["margin_eta"]
        phi_half = config["patch_phi"] + config["margin_phi"]
        md_deta = mdi.md_eta - eta_centre
        md_dphi = wrap_phi(mdi.md_phi - phi_centre)
        md_idx = torch.nonzero(
            (md_deta.abs() <= eta_half) & (md_dphi.abs() <= phi_half)
        ).squeeze(1)
        md_deta = md_deta[md_idx]
        md_dphi = md_dphi[md_idx]
        md_inputs = [
            mdi.md_inputs[md_idx],
            torch.stack([md_deta / eta_half, md_dphi / phi_half], dim=1),
        ]
        if config["absolute_phi"]:
            md_phi = mdi.md_phi[md_idx]
            md_inputs.append(torch.stack([torch.cos(md_phi), torch.sin(md_phi)], dim=1))
        md_simKinematics = mdi.md_simKinematics[md_idx]
        # Not wrapped: only its cos and sin are used
        md_simDphi = md_simKinematics[:, 2] - phi_centre
        return cls(
            md_idx=md_idx,
            md_inputs=torch.cat(md_inputs, dim=1),
            md_layer=mdi.md_layer[md_idx],
            md_pos=torch.stack([md_deta, md_dphi], dim=1),
            md_isMargin=(md_deta.abs() > config["patch_eta"])
            | (md_dphi.abs() > config["patch_phi"]),
            md_simIsFake=mdi.md_simIsFake[md_idx],
            md_simIdx=mdi.md_simIdx[md_idx],
            md_simKinematics=torch.stack(
                [
                    md_simKinematics[:, 0],
                    md_simKinematics[:, 1],
                    torch.cos(md_simDphi),
                    torch.sin(md_simDphi),
                ],
                dim=1,
            ),
            md_simVertex=mdi.md_simVertex[md_idx],
        )


def wrap_phi(dphi: torch.Tensor) -> torch.Tensor:
    """dphi wrapped into [-pi, pi)."""
    return torch.remainder(dphi + np.pi, 2 * np.pi) - np.pi


@dataclass
class PatchBatch:
    """
    Patches padded to (n_patches, n_md_max, ...); md_isValid marks real MDs. Lightning moves it to the device.
    """

    md_inputs: torch.Tensor
    md_layer: torch.Tensor
    md_pos: torch.Tensor
    md_isValid: torch.Tensor
    md_isMargin: torch.Tensor
    md_simIsFake: torch.Tensor
    md_simIdx: torch.Tensor
    md_simKinematics: torch.Tensor
    md_simVertex: torch.Tensor


def collate_patches(patches: list[Patch]) -> PatchBatch:
    """
    Pad each field to the largest patch. Padding is 0 or False, except md_simIdx: -1 (no sim) and
    md_simKinematics, md_simVertex: nan, like MDs with no sim. md_inputs and md_pos must stay finite, since padded MDs still go
    through the network.
    """

    def pad(tensors: list[torch.Tensor], value: float) -> torch.Tensor:
        return nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=value)

    return PatchBatch(
        md_inputs=pad([pat.md_inputs for pat in patches], value=0),
        md_layer=pad([pat.md_layer for pat in patches], value=0),
        md_pos=pad([pat.md_pos for pat in patches], value=0),
        md_isValid=pad(
            [torch.ones(len(pat.md_idx), dtype=torch.bool) for pat in patches],
            value=False,
        ),
        md_isMargin=pad([pat.md_isMargin for pat in patches], value=False),
        md_simIsFake=pad([pat.md_simIsFake for pat in patches], value=False),
        md_simIdx=pad([pat.md_simIdx for pat in patches], value=-1),
        md_simKinematics=pad(
            [pat.md_simKinematics for pat in patches],
            value=torch.nan,
        ),
        md_simVertex=pad(
            [pat.md_simVertex for pat in patches],
            value=torch.nan,
        ),
    )


class PatchDataset(torch.utils.data.Dataset):
    """
    Each item is one event's patches_per_event patches, centred at random MDs. Validation centres are seeded
    per event, so the validation loss is comparable between epochs and runs.
    """

    def __init__(self, paths: list[str], config: dict, train: bool) -> None:
        self.paths = paths
        self.config = config
        self.train = train

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, item: int) -> list[Patch]:
        mdi = MDInputs.from_mds(
            mds=read_event(path=self.paths[item]),
            pt_min=self.config["pt_min"],
        )
        generator = None if self.train else torch.Generator().manual_seed(item)
        md_centres = torch.randint(
            len(mdi.md_eta),
            (self.config["patches_per_event"],),
            generator=generator,
        )
        return [
            Patch.from_mdinputs(
                mdi=mdi,
                eta_centre=float(mdi.md_eta[md]),
                phi_centre=float(mdi.md_phi[md]),
                config=self.config,
            )
            for md in md_centres.tolist()
        ]


def make_loader(
    paths: list[str],
    config: dict,
    train: bool,
) -> torch.utils.data.DataLoader:
    """
    One event per step, its patches collated into one PatchBatch. batch_size=None turns off the DataLoader's own
    batching, so collate_patches gets one item: one event's patches.
    """
    return torch.utils.data.DataLoader(
        PatchDataset(paths=paths, config=config, train=train),
        batch_size=None,
        shuffle=train,
        num_workers=config["num_workers"],
        collate_fn=collate_patches,
        persistent_workers=config["num_workers"] > 0,
    )


# ---------------------------------------------------------------- algorithm


@dataclass
class MDPredictions:
    """One event's network outputs (MDTransformerNet.predict); entry i of every field belongs to MD i."""

    md_predProbFake: np.ndarray  # (n_md,)
    # (n_md, 4): q/pT, eta, cos phi, sin phi, with absolute phi
    md_predKinematics: np.ndarray
    # (n_md, 2): sim_vz, sim_vtxperp in cm, back from the regressed asinh and log1p
    md_predVertex: np.ndarray
    md_predEmbedding: np.ndarray  # (n_md, embed_dim)


class MDTransformerNet:
    """
    The whole algorithm: an MDTransformerModule plus the working point that turns its output into track
    candidates. fake_cut: MDs with p(fake) below it are clustered (chosen to keep fake_keep of the real MDs).
    eps: DBSCAN radius in cluster_space (quantile eps_quantile, see choose_eps). Both are None until chosen or
    loaded.
    """

    def __init__(self, config: dict, windows: LayerPairWindows) -> None:
        self.config = config
        self.module = MDTransformerModule(config=config, windows=windows).to(
            config["device"]
        )
        self.fake_cut = None
        self.eps = None

    def train(self, train_paths: list[str], val_paths: list[str]) -> None:
        """
        Train the network with an MDTransformerLightningModule (make_trainer). self.module is trained in place
        and keeps the weights of the last epoch.
        """
        make_trainer(config=self.config).fit(
            MDTransformerLightningModule(config=self.config, module=self.module),
            train_dataloaders=make_loader(
                paths=train_paths,
                config=self.config,
                train=True,
            ),
            val_dataloaders=make_loader(
                paths=val_paths,
                config=self.config,
                train=False,
            ),
        )
        self.module.to(self.config["device"])

    def predict(
        self,
        mds: MDsFromOneEvent,
    ) -> MDPredictions:
        """
        The MDPredictions of every MD in mds. The patch grid (patch_grid) covers the event; each grid
        patch gets its margin, and patches_per_batch of them go through the network at a time. Each MD takes the
        prediction of the grid patch that holds it.
        """
        self.module.eval()
        device = next(self.module.parameters()).device
        mdi = MDInputs.from_mds(mds=mds, pt_min=self.config["pt_min"])
        n_eta, n_phi = patch_grid(config=self.config)
        eta_width = 2 * MD_ETA_MAX / n_eta
        phi_width = 2 * np.pi / n_phi
        # The grid patch (row in eta, column in phi) that holds each MD
        md_gridRow = torch.clamp(
            torch.floor((mdi.md_eta + MD_ETA_MAX) / eta_width).long(),
            0,
            n_eta - 1,
        )
        md_gridCol = torch.floor((mdi.md_phi + np.pi) / phi_width).long() % n_phi
        n_md = len(mdi.md_eta)
        md_predProbFake = torch.full((n_md,), torch.nan)
        md_predKinematics = torch.full((n_md, len(SIM_KINEMATICS)), torch.nan)
        md_predVertex = torch.full((n_md, len(SIM_VERTEX_SCALES)), torch.nan)
        md_predEmbedding = torch.full((n_md, self.config["embed_dim"]), torch.nan)
        grid = [(row, col) for row in range(n_eta) for col in range(n_phi)]
        for start in range(0, len(grid), self.config["patches_per_batch"]):
            cells = grid[start : start + self.config["patches_per_batch"]]
            phi_centres = [-np.pi + (col + 0.5) * phi_width for _, col in cells]
            patches = [
                Patch.from_mdinputs(
                    mdi=mdi,
                    eta_centre=-MD_ETA_MAX + (row + 0.5) * eta_width,
                    phi_centre=phi_centre,
                    config=self.config,
                )
                for (row, _), phi_centre in zip(cells, phi_centres)
            ]
            batch = collate_patches(patches=patches)
            batch = PatchBatch(
                **{name: val.to(device) for name, val in vars(batch).items()}
            )
            with torch.no_grad():
                logits, kinematics, vertex, embeddings = self.module(batch=batch)
            for idx, ((row, col), phi_centre, pat) in enumerate(
                zip(cells, phi_centres, patches)
            ):
                n_pat = len(pat.md_idx)
                is_in_cell = (md_gridRow[pat.md_idx] == row) & (
                    md_gridCol[pat.md_idx] == col
                )
                md_idx = pat.md_idx[is_in_cell]
                md_predProbFake[md_idx] = torch.sigmoid(
                    logits[idx, :n_pat][is_in_cell]
                ).cpu()
                relative = kinematics[idx, :n_pat][is_in_cell].cpu()
                # Back from phi relative to the patch centre to absolute phi
                phi = torch.atan2(relative[:, 3], relative[:, 2]) + phi_centre
                md_predKinematics[md_idx] = torch.stack(
                    [relative[:, 0], relative[:, 1], torch.cos(phi), torch.sin(phi)],
                    dim=1,
                )
                transformed = vertex[idx, :n_pat][is_in_cell].cpu()
                md_predVertex[md_idx] = torch.stack(
                    [
                        SIM_VZ_SCALE * torch.sinh(transformed[:, 0]),
                        SIM_VTXPERP_SCALE * torch.expm1(transformed[:, 1]),
                    ],
                    dim=1,
                )
                md_predEmbedding[md_idx] = embeddings[idx, :n_pat][is_in_cell].cpu()
        return MDPredictions(
            md_predProbFake=md_predProbFake.numpy(),
            md_predKinematics=md_predKinematics.numpy(),
            md_predVertex=md_predVertex.numpy(),
            md_predEmbedding=md_predEmbedding.numpy(),
        )

    def choose_fake_cut(self, paths: list[str]) -> None:
        """Set self.fake_cut to quantile fake_keep of the real MDs' p(fake). Logs a [fake_cut] line."""
        real_probs = []
        start = time.time()
        for idx, path in enumerate(paths):
            print_and_log(
                line=f"[fake_cut] event={idx + 1} n_events={len(paths)} seconds={time.time() - start:.0f}",
                config=self.config,
            )
            mds = read_event(path=path)
            pred = self.predict(mds=mds)
            mdi = MDInputs.from_mds(mds=mds, pt_min=self.config["pt_min"])
            real_probs.append(pred.md_predProbFake[~mdi.md_simIsFake.numpy()])
        real_probs = np.concatenate(real_probs)
        self.fake_cut = float(np.quantile(real_probs, self.config["fake_keep"]))
        print_and_log(
            line=(
                f"[fake_cut] fake_cut={self.fake_cut:.4f} fake_keep={self.config['fake_keep']} "
                f"n_events={len(paths)} n_real_mds={len(real_probs)}"
            ),
            config=self.config,
        )

    def choose_eps(self, paths: list[str]) -> None:
        """
        Set self.eps to quantile eps_quantile, over the kept real MDs, of the cluster_space distance to the
        (MIN_MDS - 1)-th nearest other kept real MD of the same sim. DBSCAN counts an MD among its own MIN_MDS,
        so at this eps that fraction of the kept real MDs could be core points from their own sim alone. Logs an
        [eps] line.
        """
        distances = []
        start = time.time()
        for idx, path in enumerate(paths):
            print_and_log(
                line=f"[eps] event={idx + 1} n_events={len(paths)} seconds={time.time() - start:.0f}",
                config=self.config,
            )
            mds = read_event(path=path)
            pred = self.predict(mds=mds)
            md_coordinates = self.cluster_coordinates(pred=pred)
            mdi = MDInputs.from_mds(mds=mds, pt_min=self.config["pt_min"])
            md_isKeptReal = (
                pred.md_predProbFake < self.fake_cut
            ) & ~mdi.md_simIsFake.numpy()
            kept_simIdx = mdi.md_simIdx.numpy()[md_isKeptReal]
            kept_coordinates = md_coordinates[md_isKeptReal]
            for sim in np.unique(kept_simIdx):
                sim_coordinates = kept_coordinates[kept_simIdx == sim]
                if len(sim_coordinates) < MIN_MDS:
                    # Too few kept MDs for any of them to be a core point from this sim alone
                    distances.extend([np.inf] * len(sim_coordinates))
                    continue
                # The nearest neighbour of each MD is itself, at distance 0
                dist, _ = (
                    NearestNeighbors(n_neighbors=MIN_MDS)
                    .fit(sim_coordinates)
                    .kneighbors(sim_coordinates)
                )
                distances.extend(dist[:, MIN_MDS - 1])
        self.eps = float(np.quantile(distances, self.config["eps_quantile"]))
        n_too_few = int(np.isinf(distances).sum())
        print_and_log(
            line=(
                f"[eps] eps={self.eps:.4f} eps_quantile={self.config['eps_quantile']} "
                f"cluster_space={self.config['cluster_space']} n_events={len(paths)} "
                f"n_kept_real_mds={len(distances)} n_in_sims_with_too_few={n_too_few}"
            ),
            config=self.config,
        )

    def trackcands(self, paths: list[str]) -> list[TrackCandidatesFromOneEvent]:
        """
        The track candidates of each event in paths, in the same order (the harness calls this). DBSCAN runs in
        cluster_space; the track candidates' pT, eta and phi always come from the predicted kinematics.
        """
        event_trackcands = []
        for path in paths:
            mds = read_event(path=path)
            pred = self.predict(mds=mds)
            md_mdclusterIdx = find_mdclusters(
                md_coordinates=self.cluster_coordinates(pred=pred),
                md_isKept=pred.md_predProbFake < self.fake_cut,
                eps=self.eps,
            )
            event_trackcands.append(
                trackcands_from_mdclusters(
                    mds=mds,
                    md_mdclusterIdx=md_mdclusterIdx,
                    md_predProbFake=pred.md_predProbFake,
                    md_predKinematics=pred.md_predKinematics,
                )
            )
        return event_trackcands

    def cluster_coordinates(self, pred: MDPredictions) -> np.ndarray:
        """The per-MD coordinates DBSCAN clusters, chosen by cluster_space: kinematics or embedding."""
        if self.config["cluster_space"] == "kinematics":
            return pred.md_predKinematics
        return pred.md_predEmbedding

    def save(self, path: str) -> None:
        """The config, the network's state_dict (windows included), fake_cut and eps, as one dict."""
        torch.save(
            {
                "config": self.config,
                "state_dict": self.module.state_dict(),
                "fake_cut": self.fake_cut,
                "eps": self.eps,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "MDTransformerNet":
        """The algorithm saved at path, on device (which replaces the saved config's device)."""
        saved = torch.load(path, map_location=device, weights_only=True)
        config = dict(saved["config"], device=device)
        # Placeholder windows: load_state_dict overwrites them with the saved ones
        windows = LayerPairWindows(
            eta=torch.zeros(N_LAYERS, N_LAYERS),
            phi=torch.zeros(N_LAYERS, N_LAYERS),
        )
        algo = cls(config=config, windows=windows)
        algo.module.load_state_dict(saved["state_dict"])
        algo.fake_cut = saved["fake_cut"]
        algo.eps = saved["eps"]
        return algo


def patch_grid(config: dict) -> tuple[int, int]:
    """
    Number of grid patches in eta (over |eta| <= MD_ETA_MAX) and phi: patches with margin 0 that cover the event
    without overlap, no wider than (patch_eta, patch_phi).
    """
    n_eta = int(np.ceil(MD_ETA_MAX / config["patch_eta"]))
    n_phi = int(np.ceil(np.pi / config["patch_phi"]))
    return n_eta, n_phi


# ---------------------------------------------------------------- training


@dataclass
class LossTerms:
    """The unweighted loss terms of one batch, or their mean over validation batches."""

    bce: torch.Tensor
    mse: torch.Tensor
    sim_vz: torch.Tensor
    sim_vtxperp: torch.Tensor
    contrastive: torch.Tensor

    def total(self, config: dict) -> torch.Tensor:
        """The weighted sum: bce weighs 1, the others their *_weight from config."""
        return (
            self.bce
            + config["mse_weight"] * self.mse
            + config["sim_vz_weight"] * self.sim_vz
            + config["sim_vtxperp_weight"] * self.sim_vtxperp
            + config["contrastive_weight"] * self.contrastive
        )

    def log_text(self) -> str:
        """loss_<term>=value for every term, for the [train] and [val] lines."""
        return " ".join(
            f"loss_{name}={term.item():.4f}" for name, term in vars(self).items()
        )

    @classmethod
    def mean(cls, terms: list["LossTerms"]) -> "LossTerms":
        return cls(
            **{
                name: torch.stack([getattr(term, name) for term in terms]).mean()
                for name in vars(terms[0])
            }
        )


class MDTransformerLightningModule(pl.LightningModule):
    """
    Trains an MDTransformerModule: BCE on the fake logit of MDs outside the margin, plus mse_weight times the MSE
    of SIM_KINEMATICS on real MDs outside the margin, plus sim_vz_weight and sim_vtxperp_weight times the MSE of
    each vertex output (as MDInputs.md_simVertex) on the same MDs, plus (if embed_dim > 0) contrastive_weight
    times contrastive_loss. AdamW with a one-cycle learning rate. Logs [train] every log_every steps and [val]
    per epoch; the hyperparameters (the config) go to hparams.yaml.
    """

    def __init__(self, config: dict, module: "MDTransformerModule") -> None:
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.module = module
        # One LossTerms per validation event, averaged at the end of the epoch
        self.val_losses = []
        self.start = time.time()

    def losses(self, batch: PatchBatch) -> LossTerms:
        """The unweighted loss terms; contrastive is 0 if embed_dim is 0."""
        md_predIsFakeLogit, md_predKinematics, md_predVertex, md_predEmbedding = (
            self.module(batch=batch)
        )
        md_isInLoss = batch.md_isValid & ~batch.md_isMargin
        bce = nn.functional.binary_cross_entropy_with_logits(
            md_predIsFakeLogit[md_isInLoss],
            batch.md_simIsFake[md_isInLoss].float(),
        )
        md_isReal = md_isInLoss & ~batch.md_simIsFake
        mse = nn.functional.mse_loss(
            md_predKinematics[md_isReal],
            batch.md_simKinematics[md_isReal],
        )
        md_predVertex = md_predVertex[md_isReal]
        md_simVertex = batch.md_simVertex[md_isReal]
        if self.config["embed_dim"] > 0:
            contrastive = contrastive_loss(
                md_embedding=md_predEmbedding,
                batch=batch,
                contrastive_margin=self.config["contrastive_margin"],
            )
        else:
            contrastive = torch.zeros((), device=bce.device)
        return LossTerms(
            bce=bce,
            mse=mse,
            sim_vz=nn.functional.mse_loss(md_predVertex[:, 0], md_simVertex[:, 0]),
            sim_vtxperp=nn.functional.mse_loss(md_predVertex[:, 1], md_simVertex[:, 1]),
            contrastive=contrastive,
        )

    def training_step(self, batch: PatchBatch, batch_idx: int) -> torch.Tensor:
        terms = self.losses(batch=batch)
        loss = terms.total(config=self.config)
        # To metrics.csv, every log_every steps (make_trainer)
        self.log("train_loss", loss, batch_size=1)
        if batch_idx % self.config["log_every"] == 0:
            gpu_mem = (
                torch.cuda.max_memory_allocated() / 1e9
                if torch.cuda.is_available()
                else 0.0
            )
            print_and_log(
                line=(
                    f"[train] epoch={self.current_epoch} step={batch_idx} loss={loss.item():.4f} "
                    f"{terms.log_text()} n_patches={batch.md_inputs.shape[0]} "
                    f"n_md_max={batch.md_inputs.shape[1]} gpu_mem_peak_gb={gpu_mem:.2f} "
                    f"seconds={time.time() - self.start:.0f}"
                ),
                config=self.config,
            )
        return loss

    def validation_step(self, batch: PatchBatch, batch_idx: int) -> None:
        self.val_losses.append(self.losses(batch=batch))

    def on_validation_epoch_end(self) -> None:
        terms = LossTerms.mean(terms=self.val_losses)
        n_events = len(self.val_losses)
        self.val_losses.clear()
        loss = terms.total(config=self.config)
        # val_loss picks the best checkpoint (make_trainer)
        self.log("val_loss", loss)
        print_and_log(
            line=(
                f"[val] epoch={self.current_epoch} n_events={n_events} loss={loss.item():.4f} "
                f"{terms.log_text()} seconds={time.time() - self.start:.0f}"
            ),
            config=self.config,
        )

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
        )
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config["lr"],
            total_steps=total_steps,
            # 5% warm-up, but at least 3 steps, so that short test runs work
            pct_start=max(0.05, 3 / total_steps),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def contrastive_loss(
    md_embedding: torch.Tensor,
    batch: PatchBatch,
    contrastive_margin: float,
) -> torch.Tensor:
    """
    Per patch, the reference MDs are the real MDs outside the margin. For each, every other valid MD (margin
    included) is a positive (real, same sim) or a negative. Loss: mean d^2 over positives plus mean relu(contrastive_margin - d)^2
    over negatives, with d the embedding distance, averaged over patches with at least one reference MD.
    Fake MDs (no sim, or a sim below pt_min) are negatives of every reference MD.
    """
    terms = []
    for patch in range(md_embedding.shape[0]):
        md_isValid = batch.md_isValid[patch]
        md_isReal = md_isValid & ~batch.md_simIsFake[patch]
        reference_mds = torch.nonzero(md_isReal & ~batch.md_isMargin[patch]).squeeze(1)
        if len(reference_mds) == 0:
            continue
        other_mds = torch.nonzero(md_isValid).squeeze(1)
        # (n_reference_mds, n_other_mds)
        dist = torch.cdist(
            md_embedding[patch, reference_mds],
            md_embedding[patch, other_mds],
        )
        md_simIdx = batch.md_simIdx[patch]
        not_self = reference_mds[:, None] != other_mds[None, :]
        is_positive = (
            (md_simIdx[reference_mds][:, None] == md_simIdx[other_mds][None, :])
            & md_isReal[other_mds][None, :]
            & not_self
        )
        is_negative = ~is_positive & not_self
        # A patch can have reference MDs but no positives (each real sim with a single MD)
        loss_positive = (
            (dist[is_positive] ** 2).mean() if is_positive.any() else dist.new_zeros(())
        )
        loss_negative = (
            nn.functional.relu(contrastive_margin - dist[is_negative]) ** 2
        ).mean()
        terms.append(loss_positive + loss_negative)
    return torch.stack(terms).mean()


def make_trainer(config: dict) -> pl.Trainer:
    """
    Lightning's progress bar off (it redraws lines with carriage returns): PrintProgress prints plain lines
    instead. CSVLogger and checkpoints in the run directory.
    """
    return pl.Trainer(
        max_epochs=config["epochs"],
        accelerator="gpu" if config["device"] == "cuda" else "cpu",
        devices=1,
        precision=config["precision"],
        gradient_clip_val=GRADIENT_CLIP_NORM,
        enable_progress_bar=False,
        enable_model_summary=False,
        # hparams.yaml and metrics.csv in the run directory: its name (the UTC time) is the version
        logger=pl.loggers.CSVLogger(
            save_dir=config["output_dir"],
            name="",
            version=os.path.basename(config["run_dir"]),
        ),
        log_every_n_steps=config["log_every"],
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=config["run_dir"],
                filename="epoch{epoch}_step{step}",
                auto_insert_metric_name=False,
                monitor="val_loss",
                save_last=True,
            ),
            PrintProgress(config=config),
        ],
        num_sanity_val_steps=0,
    )


class PrintProgress(pl.Callback):
    """
    A progress bar without carriage returns: one [progress] line per training and validation batch (one event),
    with how far the epoch is and an estimate of the seconds left in it (also in train.log, see print_and_log).
    """

    def __init__(self, config: dict) -> None:
        self.config = config

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self.start = time.time()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict,
        batch: PatchBatch,
        batch_idx: int,
    ) -> None:
        self.print_line(
            stage="train",
            epoch=trainer.current_epoch,
            batch_idx=batch_idx,
            n_batches=trainer.num_training_batches,
            loss=outputs["loss"].item(),
        )

    def on_validation_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        self.start = time.time()

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: None,
        batch: PatchBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.print_line(
            stage="val",
            epoch=trainer.current_epoch,
            batch_idx=batch_idx,
            n_batches=trainer.num_val_batches[dataloader_idx],
            loss=None,
        )

    def print_line(
        self,
        stage: str,
        epoch: int,
        batch_idx: int,
        n_batches: int,
        loss: float | None,
    ) -> None:
        n_done = batch_idx + 1
        seconds = time.time() - self.start
        seconds_left = seconds / n_done * (n_batches - n_done)
        loss_text = "" if loss is None else f" loss={loss:.4f}"
        line = (
            f"[progress] stage={stage} epoch={epoch} batch={n_done} n_batches={n_batches} "
            f"percent={100 * n_done / n_batches:.1f} seconds={seconds:.0f} "
            f"seconds_left={seconds_left:.0f}{loss_text}"
        )
        print_and_log(line=line, config=self.config)


# ---------------------------------------------------------------- network


class MDTransformerModule(nn.Module):
    """
    A per-MD input MLP, n_blocks MDTransformerBlocks with the layer-pair window mask, and a head -> (fake logit,
    SIM_KINEMATICS, vertex, embedding). The head's hidden layer is
    shared; only its last linear layer has separate rows per output. The windows are buffers, so they are saved
    with the state_dict.
    """

    def __init__(self, config: dict, windows: LayerPairWindows) -> None:
        super().__init__()
        hidden = config["hidden"]
        n_inputs = N_INPUTS + (N_ABSOLUTE_PHI_INPUTS if config["absolute_phi"] else 0)
        self.input_mlp = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.blocks = nn.ModuleList(
            [
                MDTransformerBlock(
                    hidden=hidden,
                    n_heads=config["n_heads"],
                    dropout=config["dropout"],
                )
                for _ in range(config["n_blocks"])
            ]
        )
        # Pre-norm blocks leave their output unnormalised
        self.norm = nn.LayerNorm(hidden)
        # Outputs: fake logit, SIM_KINEMATICS, vertex (as MDInputs.md_simVertex), embedding
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(
                hidden,
                1 + len(SIM_KINEMATICS) + len(SIM_VERTEX_SCALES) + config["embed_dim"],
            ),
        )
        self.register_buffer("window_eta", windows.eta.clone())
        self.register_buffer("window_phi", windows.phi.clone())

    def forward(
        self,
        batch: PatchBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        md_predIsFakeLogit (n_patches, n_md_max), md_predKinematics (n_patches, n_md_max, 4) with phi relative
        to the patch centre, md_predVertex (n_patches, n_md_max, 2) as MDInputs.md_simVertex, and
        md_predEmbedding (n_patches, n_md_max, embed_dim).
        """
        mask = self.attention_mask(batch=batch)
        hid = self.input_mlp(batch.md_inputs)
        for block in self.blocks:
            hid = block(hid=hid, mask=mask)
        out = self.head(self.norm(hid))
        end_kinematics = 1 + len(SIM_KINEMATICS)
        end_vertex = end_kinematics + len(SIM_VERTEX_SCALES)
        return (
            out[..., 0],
            out[..., 1:end_kinematics],
            out[..., end_kinematics:end_vertex],
            out[..., end_vertex:],
        )

    def attention_mask(self, batch: PatchBatch) -> torch.Tensor:
        """
        (n_patches, 1, n_md_max, n_md_max), True where a query (dim 2) may attend to a key (dim 3), as
        scaled_dot_product_attention wants; the 1 broadcasts over heads. Allowed: a valid key inside the query's
        layer-pair window, strictly |deta| < window and |dphi| < window, so zero-width windows (window_scale 0)
        leave only self-attention: MDs sharing an anchor hit have identical (eta, phi). Every MD attends to
        itself, so no row is empty.
        """
        md_pos = batch.md_pos
        # (n_patches, n_md_max, n_md_max): query along dim 1, key along dim 2
        layer_query = batch.md_layer[:, :, None]
        layer_key = batch.md_layer[:, None, :]
        abs_deta = (md_pos[:, :, None, 0] - md_pos[:, None, :, 0]).abs()
        abs_dphi = wrap_phi(md_pos[:, :, None, 1] - md_pos[:, None, :, 1]).abs()
        is_allowed = (
            batch.md_isValid[:, None, :]
            & (abs_deta < self.window_eta[layer_query, layer_key])
            & (abs_dphi < self.window_phi[layer_query, layer_key])
        )
        n_md_max = md_pos.shape[1]
        is_allowed |= torch.eye(n_md_max, dtype=torch.bool, device=md_pos.device)
        return is_allowed[:, None]


class MDTransformerBlock(nn.Module):
    """
    A pre-norm transformer block: masked multi-head self-attention, then a feed-forward layer (width 2 * hidden),
    each in a residual. Attention is scaled_dot_product_attention with one boolean mask per patch, shared by all
    heads (see the README decision on nn.TransformerEncoderLayer).
    """

    def __init__(self, hidden: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(hidden)
        # Query, key and value of every head in one linear layer
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.out = nn.Linear(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, 2 * hidden),
            nn.GELU(),
            nn.Linear(2 * hidden, hidden),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, hid: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """hid (n_patches, n_md_max, hidden) -> same shape; mask from MDTransformerModule.attention_mask."""
        n_patches, n_md_max, hidden = hid.shape
        qkv = self.qkv(self.norm1(hid)).view(
            n_patches, n_md_max, 3, self.n_heads, hidden // self.n_heads
        )
        # Each (n_patches, n_heads, n_md_max, hidden // n_heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        att = nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
        )
        att = att.transpose(1, 2).reshape(n_patches, n_md_max, hidden)
        hid = hid + self.drop(self.out(att))
        return hid + self.drop(self.ff(self.norm2(hid)))


# ---------------------------------------------------------------- per-MD quality


def log_md_quality(algo: MDTransformerNet, paths: list[str]) -> None:
    """
    One [md_quality] line over every MD of paths: the fraction of fake MDs rejected when 90% of real MDs are kept,
    and the IQR of the q/pT, eta, phi, sim_vz and sim_vtxperp (cm) residuals of real MDs.
    """
    probs, is_fake, kinematics, sim_kinematics = [], [], [], []
    vertex, sim_vertex = [], []
    start = time.time()
    for idx, path in enumerate(paths):
        print_and_log(
            line=f"[md_quality] event={idx + 1} n_events={len(paths)} seconds={time.time() - start:.0f}",
            config=algo.config,
        )
        mds = read_event(path=path)
        pred = algo.predict(mds=mds)
        mdi = MDInputs.from_mds(mds=mds, pt_min=algo.config["pt_min"])
        probs.append(pred.md_predProbFake)
        is_fake.append(mdi.md_simIsFake.numpy())
        kinematics.append(pred.md_predKinematics)
        sim_kinematics.append(mdi.md_simKinematics.numpy())
        vertex.append(pred.md_predVertex)
        sim_vertex.append(
            mds.md_simFeatures[
                :,
                [SIM_FEATURES.index("sim_vz"), SIM_FEATURES.index("sim_vtxperp")],
            ].numpy()
        )
    probs = np.concatenate(probs)
    is_fake = np.concatenate(is_fake)
    is_real = ~is_fake
    # Real MDs only: (q/pT, eta, cos phi, sin phi) and the sim's (q/pT, eta, phi)
    kinematics = np.concatenate(kinematics)[is_real]
    sim_kinematics = np.concatenate(sim_kinematics)[is_real]
    # Real MDs only: (sim_vz, sim_vtxperp) in cm
    vertex_residual = (
        np.concatenate(vertex)[is_real] - np.concatenate(sim_vertex)[is_real]
    )
    cut_keep90 = np.quantile(probs[is_real], 0.9)
    phi = np.arctan2(kinematics[:, 3], kinematics[:, 2])
    phi_residual = wrap_phi(torch.from_numpy(phi - sim_kinematics[:, 2])).numpy()
    print_and_log(
        line=(
            f"[md_quality] n_events={len(paths)} n_md={len(probs)} frac_real={is_real.mean():.4f} "
            f"frac_fake_rejected_at_real_keep_0.9={np.mean(probs[is_fake] >= cut_keep90):.4f} "
            f"qoverpt_res_iqr={iqr(kinematics[:, 0] - sim_kinematics[:, 0]):.4f} "
            f"eta_res_iqr={iqr(kinematics[:, 1] - sim_kinematics[:, 1]):.4f} "
            f"phi_res_iqr={iqr(phi_residual):.4f} "
            f"vz_res_iqr={iqr(vertex_residual[:, 0]):.4f} "
            f"vtxperp_res_iqr={iqr(vertex_residual[:, 1]):.4f}"
        ),
        config=algo.config,
    )


def iqr(values: np.ndarray) -> float:
    """Interquartile range: the 75th minus the 25th percentile."""
    return float(np.quantile(values, 0.75) - np.quantile(values, 0.25))


if __name__ == "__main__":
    main()
