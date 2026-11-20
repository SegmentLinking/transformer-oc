"""
MDTransformerNet: a tracking algorithm from outer-tracker MDs to track candidates.

A transformer (MDTransformerModule) sees the MDs of one (eta, phi) patch at a time and predicts, for each MD,
whether it is fake and the kinematics of its sim, SIM_KINEMATICS = (q/pT, eta, cos phi, sin phi). Optionally
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

From the repo root, train, choose the working point and log the per-MD quality into one new run directory:
apptainer exec --nv ml-tracking.sif python3 -m mdml.md_transformer --config mdml/md_transformer.yaml

The run directory (config "output") must be new. It gets hparams.yaml and metrics.csv (Lightning's CSVLogger),
Lightning checkpoints, and md_transformer.pt (the whole algorithm, see MDTransformerNet.save).
"""

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
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
# The patch grid covers |eta| <= MD_ETA_MAX; the largest |md eta| seen is 2.78
MD_ETA_MAX = 2.8
# Layer pairs with fewer same-sim MD pairs get a zero-width window (no attention between them)
MIN_WINDOW_PAIRS = 50


def main() -> None:
    config = parse_args()
    pl.seed_everything(config["seed"])
    os.makedirs(config["output"])
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
    algo.save(path=os.path.join(config["output"], "md_transformer.pt"))
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
        raise NotImplementedError()


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
        # The .pt files fill every sim feature with -999 when there is no sim; nan makes any use of them obvious
        md_simKinematics[mds.md_simIdx < 0] = torch.nan
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


def collate_patches(patches: list[Patch]) -> PatchBatch:
    """
    Pad each field to the largest patch. Padding is 0 or False, except md_simIdx: -1 (no sim) and
    md_simKinematics: nan, like MDs with no sim. md_inputs and md_pos must stay finite, since padded MDs still go
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


class MDTransformerNet:
    """
    The whole algorithm: an MDTransformerModule plus the working point that turns its output into track
    candidates. fake_cut: MDs with p(fake) below it are clustered (chosen to keep fake_keep of the real MDs).
    eps: DBSCAN radius in cluster_space (quantile eps_quantile, see choose_eps). Both are None until chosen or
    loaded.
    """

    def __init__(self, config: dict, windows: LayerPairWindows) -> None:
        raise NotImplementedError()

    def train(self, train_paths: list[str], val_paths: list[str]) -> None:
        """Train the network with an MDTransformerLightningModule (make_trainer)."""
        raise NotImplementedError()

    def predict(
        self,
        mds: MDsFromOneEvent,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        md_predProbFake (n_md,), md_predKinematics (n_md, 4: q/pT, eta, cos phi, sin phi with absolute phi) and
        md_predEmbedding (n_md, embed_dim) for every MD. The patch grid (patch_grid) covers the event; each grid
        patch gets its margin, and patches_per_batch of them go through the network at a time. Each MD takes the
        prediction of the grid patch that holds it.
        """
        raise NotImplementedError()

    def choose_fake_cut(self, paths: list[str]) -> None:
        """Set self.fake_cut to quantile fake_keep of the real MDs' p(fake). Logs a [fake_cut] line."""
        raise NotImplementedError()

    def choose_eps(self, paths: list[str]) -> None:
        """
        Set self.eps to quantile eps_quantile, over the kept real MDs, of the cluster_space distance to the
        (MIN_MDS - 1)-th nearest other kept real MD of the same sim. DBSCAN counts an MD among its own MIN_MDS,
        so at this eps that fraction of the kept real MDs could be core points from their own sim alone. Logs an
        [eps] line.
        """
        raise NotImplementedError()

    def trackcands(self, paths: list[str]) -> list[TrackCandidatesFromOneEvent]:
        """
        The track candidates of each event in paths, in the same order (the harness calls this). DBSCAN runs in
        cluster_space; the track candidates' pT, eta and phi always come from the predicted kinematics.
        """
        raise NotImplementedError()

    def save(self, path: str) -> None:
        """The config, the network's state_dict (windows included), fake_cut and eps, as one dict."""
        raise NotImplementedError()

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "MDTransformerNet":
        raise NotImplementedError()


def patch_grid(config: dict) -> tuple[int, int]:
    """
    Number of grid patches in eta (over |eta| <= MD_ETA_MAX) and phi: patches with margin 0 that cover the event
    without overlap, no wider than (patch_eta, patch_phi).
    """
    raise NotImplementedError()


# ---------------------------------------------------------------- training


class MDTransformerLightningModule(pl.LightningModule):
    """
    Trains an MDTransformerModule: BCE on the fake logit of MDs outside the margin, plus mse_weight times the MSE
    of SIM_KINEMATICS on real MDs outside the margin, plus (if embed_dim > 0) contrastive_weight times
    contrastive_loss. AdamW with a one-cycle learning rate. Logs [train] every log_every steps and [val] per
    epoch; the hyperparameters (the config) go to hparams.yaml.
    """

    def __init__(self, config: dict, module: "MDTransformerModule") -> None:
        raise NotImplementedError()

    def losses(
        self,
        batch: PatchBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(bce, mse, contrastive), each unweighted; contrastive is 0 if embed_dim is 0."""
        raise NotImplementedError()

    def training_step(self, batch: PatchBatch, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError()

    def validation_step(self, batch: PatchBatch, batch_idx: int) -> None:
        raise NotImplementedError()

    def on_validation_epoch_end(self) -> None:
        raise NotImplementedError()

    def configure_optimizers(self) -> dict:
        raise NotImplementedError()


def contrastive_loss(
    md_embedding: torch.Tensor,
    batch: PatchBatch,
    contrastive_margin: float,
) -> torch.Tensor:
    """
    Per patch, anchors are the real MDs outside the margin, and every other valid MD (margin included) is a
    positive (real, same sim) or a negative. Loss: mean d^2 over positives plus mean relu(contrastive_margin - d)^2
    over negatives, with d the embedding distance, averaged over patches with at least one anchor.
    """
    raise NotImplementedError()


def make_trainer(config: dict) -> pl.Trainer:
    """No progress bar (plain [train]/[val] lines), CSVLogger and checkpoints in the run directory."""
    raise NotImplementedError()


# ---------------------------------------------------------------- network


class MDTransformerModule(nn.Module):
    """
    A per-MD input MLP, n_blocks pre-norm transformer encoder layers (nn.TransformerEncoderLayer) with the
    layer-pair window mask, and a head -> (fake logit, SIM_KINEMATICS, embedding). The head's hidden layer is
    shared; only its last linear layer has separate rows per output. The windows are buffers, so they are saved
    with the state_dict.
    """

    def __init__(self, config: dict, windows: LayerPairWindows) -> None:
        raise NotImplementedError()

    def forward(
        self,
        batch: PatchBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        md_predIsFakeLogit (n_patches, n_md_max), md_predKinematics (n_patches, n_md_max, 4) and
        md_predEmbedding (n_patches, n_md_max, embed_dim), with phi relative to the patch centre.
        """
        raise NotImplementedError()

    def attention_mask(self, batch: PatchBatch) -> torch.Tensor:
        """
        (n_patches * n_heads, n_md_max, n_md_max), True where attention is blocked: a padded key, or a key outside
        the query's layer-pair window. Inside means strictly |deta| < window and |dphi| < window, so zero-width
        windows (window_scale 0) leave only self-attention: MDs sharing an anchor hit have identical (eta, phi).
        Every MD attends to itself, so no row is fully blocked.
        """
        raise NotImplementedError()


# ---------------------------------------------------------------- per-MD quality


def log_md_quality(algo: MDTransformerNet, paths: list[str]) -> None:
    """
    One [md_quality] line over every MD of paths: the fraction of fake MDs rejected when 90% of real MDs are kept,
    and the IQR of the q/pT, eta and phi residuals of real MDs.
    """
    raise NotImplementedError()


if __name__ == "__main__":
    main()
