"""
MDTransformerNet: a tracking algorithm from outer-tracker MDs to track candidates.

A transformer (MDTransformerModule) sees the MDs of one (eta, phi) patch at a time and predicts, for each MD,
whether it is fake and the kinematics of its sim, SIM_KINEMATICS = (q/pT, eta, cos phi, sin phi).

- Patches: a patch is the MDs within half-widths (patch_eta, patch_phi) of a centre. Around it, a margin of
  half-widths (margin_eta, margin_phi) adds context: margin MDs are seen by attention but are not in the loss or
  the predictions. phi is relative to the patch centre in the inputs and targets, so the network is phi-symmetric
  unless absolute_phi also gives it each MD's absolute phi.
- Attention: an MD attends to another only if their |deta| and |dphi| are within the window of their layer pair:
  a quantile of the |deta| and |dphi| of same-sim MD pairs in training events (layer_pair_windows).
- Inference: a patch grid (patches with margin 0) covers the event, so each MD is in exactly one grid patch; each
  grid patch gets its margin and goes through the network.
- Decoding: MDs with p(fake) >= fake_cut are dropped, DBSCAN (min_samples=MIN_MDS) clusters the rest in the
  predicted SIM_KINEMATICS with radius eps, and each cluster is a track candidate.

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
    windows = layer_pair_windows(
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


def layer_pair_windows(paths: list[str], config: dict) -> LayerPairWindows:
    """
    For each layer pair, quantile window_quantile of |deta| and |dphi| over the same-sim MD pairs (sim pT >=
    pt_min) in paths, times window_scale. Pairs with fewer than MIN_WINDOW_PAIRS get width 0. Logs one [window]
    line per layer pair.
    """
    raise NotImplementedError()


# ---------------------------------------------------------------- data


@dataclass
class EventInputs:
    """One event's per-MD tensors that patches are cut from (n_md rows each)."""

    # (n_md, N_LAYERS + len(FEATURE_SCALES)): layer one-hot, scaled features
    md_inputs: torch.Tensor
    md_eta: torch.Tensor
    md_phi: torch.Tensor
    md_layer: torch.Tensor  # layer - FIRST_LAYER
    md_simIdx: torch.Tensor
    md_simIsFake: torch.Tensor  # no sim, or sim pT < pt_min
    md_simKinematics: torch.Tensor  # (n_md, 3): sim q/pT, eta, phi


def event_inputs(mds: MDsFromOneEvent, pt_min: float) -> EventInputs:
    raise NotImplementedError()


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


def make_patch(
    ev: EventInputs,
    eta_centre: float,
    phi_centre: float,
    config: dict,
) -> Patch:
    raise NotImplementedError()


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
    md_simKinematics: torch.Tensor


def collate_patches(patches: list[Patch]) -> PatchBatch:
    raise NotImplementedError()


class PatchDataset(torch.utils.data.Dataset):
    """
    Each item is one event's patches_per_event patches, centred at random MDs. Validation centres are seeded
    per event, so the validation loss is comparable between epochs and runs.
    """

    def __init__(self, paths: list[str], config: dict, train: bool) -> None:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

    def __getitem__(self, item: int) -> list[Patch]:
        raise NotImplementedError()


def make_loader(
    paths: list[str],
    config: dict,
    train: bool,
) -> torch.utils.data.DataLoader:
    """One event per step (batch_size 1), its patches collated into one PatchBatch."""
    raise NotImplementedError()


# ---------------------------------------------------------------- algorithm


class MDTransformerNet:
    """
    The whole algorithm: an MDTransformerModule plus the working point that turns its output into track
    candidates. fake_cut: MDs with p(fake) below it are clustered (chosen to keep fake_keep of the real MDs).
    eps: DBSCAN radius in predicted SIM_KINEMATICS (quantile eps_quantile, see choose_eps). Both are None until
    chosen or loaded.
    """

    def __init__(self, config: dict, windows: LayerPairWindows) -> None:
        raise NotImplementedError()

    def train(self, train_paths: list[str], val_paths: list[str]) -> None:
        """Train the network with an MDTransformerLightningModule (make_trainer)."""
        raise NotImplementedError()

    def predict(self, mds: MDsFromOneEvent) -> tuple[np.ndarray, np.ndarray]:
        """
        md_predProbFake (n_md,) and md_predKinematics (n_md, 4: q/pT, eta, cos phi, sin phi with absolute phi)
        for every MD. The patch grid (patch_grid) covers the event; each grid patch gets its margin, and
        patches_per_batch of them go through the network at a time. Each MD takes the prediction of the grid
        patch that holds it.
        """
        raise NotImplementedError()

    def choose_fake_cut(self, paths: list[str]) -> None:
        """Set self.fake_cut to quantile fake_keep of the real MDs' p(fake). Logs a [fake_cut] line."""
        raise NotImplementedError()

    def choose_eps(self, paths: list[str]) -> None:
        """
        Set self.eps to quantile eps_quantile, over the kept real MDs, of the predicted-kinematics distance to the
        (MIN_MDS - 1)-th nearest other kept real MD of the same sim. DBSCAN counts an MD among its own MIN_MDS,
        so at this eps that fraction of the kept real MDs could be core points from their own sim alone. Logs an
        [eps] line.
        """
        raise NotImplementedError()

    def trackcands(self, paths: list[str]) -> list[TrackCandidatesFromOneEvent]:
        """The track candidates of each event in paths, in the same order (the harness calls this)."""
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
    of SIM_KINEMATICS on real MDs outside the margin. AdamW with a one-cycle learning rate. Logs [train] every
    log_every steps and [val] per epoch; the hyperparameters (the config) go to hparams.yaml.
    """

    def __init__(self, config: dict, module: "MDTransformerModule") -> None:
        raise NotImplementedError()

    def losses(self, batch: PatchBatch) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def training_step(self, batch: PatchBatch, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError()

    def validation_step(self, batch: PatchBatch, batch_idx: int) -> None:
        raise NotImplementedError()

    def on_validation_epoch_end(self) -> None:
        raise NotImplementedError()

    def configure_optimizers(self) -> dict:
        raise NotImplementedError()


def make_trainer(config: dict) -> pl.Trainer:
    """No progress bar (plain [train]/[val] lines), CSVLogger and checkpoints in the run directory."""
    raise NotImplementedError()


# ---------------------------------------------------------------- network


class MDTransformerModule(nn.Module):
    """
    Per-MD MLP embedding, n_blocks pre-norm transformer encoder layers (nn.TransformerEncoderLayer) with the
    layer-pair window mask, and a head -> (fake logit, SIM_KINEMATICS). The windows are buffers, so they are
    saved with the state_dict.
    """

    def __init__(self, config: dict, windows: LayerPairWindows) -> None:
        raise NotImplementedError()

    def forward(self, batch: PatchBatch) -> tuple[torch.Tensor, torch.Tensor]:
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
