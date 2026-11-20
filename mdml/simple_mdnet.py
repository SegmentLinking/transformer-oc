"""
SimpleMDNet: a whole tracking algorithm from outer-tracker MDs to track candidates.

An MLP (MDNet) sees one MD at a time and predicts whether it is fake and the kinematics of its sim,
SIM_KINEMATICS = (q/pT, eta, cos phi, sin phi). MDs with p(fake) >= fake_cut are dropped, DBSCAN
(min_samples=MIN_MDS) clusters the rest in the predicted kinematics with radius eps, and each cluster is a
track candidate. SimpleMDNet owns the MDNet and the working point (fake_cut, eps), and trains and chooses
them itself.

From the repo root, train and choose the working point into one new run directory:
apptainer exec --nv /ceph/users/atuna/ml-tracking.sif python3 -u -m mdml.simple_mdnet \
    --train-dir ../transformer-datasets/samples/ttbar/train \
    --val-dir ../transformer-datasets/samples/ttbar/val \
    --output output/mdml/run_$(date +%Y_%m_%d_%H%M%S)

--output must be a new directory; the date gives one per run, e.g. output/mdml/run_2026_11_17_183005. It gets
mdnet_epoch_<N>.pt (the MDNet after each epoch) and simple_mdnet.pt (the whole algorithm, see SimpleMDNet.save).
"""

import argparse
import os
import random
import time

import numpy as np
import torch
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
# Fixed input scales, about the largest |value| of each feature (scratch input_scales.py, 5 training events)
FEATURE_SCALES = {
    "rt": 110.0,  # cm
    "cosphi": 1.0,
    "sinphi": 1.0,
    "z": 270.0,  # cm
    "drt": 5.0,  # cm
    "dphi": 0.01,
    "dz": 5.0,  # cm
    "eta": 2.5,
    "qoverpt": 1.0,  # GeV^-1
}
N_INPUTS = N_LAYERS + len(FEATURE_SCALES)
# What the network regresses for each non-fake MD
SIM_KINEMATICS = ["sim_qoverpt", "sim_eta", "sim_cosphi", "sim_sinphi"]
# The fake logit, then SIM_KINEMATICS
N_TARGETS = 1 + len(SIM_KINEMATICS)
N_HIDDEN = 64


def main() -> None:
    args = parse_args()
    train_paths = event_paths(pt_dir=args.train_dir)[: args.n_events]
    val_paths = event_paths(pt_dir=args.val_dir)[: args.n_val_events]
    # A new directory per run, so a run never overwrites another's files
    os.makedirs(args.output)
    # Seeds the initial weights; the event order is seeded per epoch in train_mdnet
    torch.manual_seed(args.seed)
    simple_mdnet = SimpleMDNet(device=args.device, pt_min=args.pt_min)
    simple_mdnet.train_mdnet(
        train_paths=train_paths,
        val_paths=val_paths,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        log_every=args.log_every,
        output=args.output,
    )
    # The working point is chosen on validation events, never on training events
    simple_mdnet.choose_fake_cut(paths=val_paths)
    simple_mdnet.choose_eps(paths=val_paths)
    simple_mdnet.save(path=os.path.join(args.output, "simple_mdnet.pt"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SimpleMDNet and choose its working point."
    )
    parser.add_argument(
        "--train-dir",
        required=True,
        help="directory of training .pt files",
    )
    parser.add_argument(
        "--val-dir",
        required=True,
        help="directory of validation .pt files",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="new directory for the run",
    )
    parser.add_argument(
        "--n-events",
        type=int,
        default=None,
        help="use only the first N training events",
    )
    parser.add_argument(
        "--n-val-events",
        type=int,
        default=None,
        help="use only the first N validation events",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--pt-min",
        type=float,
        default=0.8,
        help="MDs of sims below this pT (GeV) count as fake",
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed for the initial weights and the order of training events",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="log a [train] line every N steps",
    )
    return parser.parse_args()


class SimpleMDNet:
    """
    The whole algorithm: an MDNet plus the working point that turns its output into track candidates.

    fake_cut: MDs with predicted p(fake) below it are clustered. eps: DBSCAN radius in the predicted
    SIM_KINEMATICS space. Both are None until chosen (choose_fake_cut, choose_eps) or loaded.
    pt_min: MDs of sims below this pT (GeV) count as fake in training.
    """

    def __init__(
        self,
        device: str = "cuda",
        pt_min: float = 0.8,
    ) -> None:
        self.device = device
        self.pt_min = pt_min
        self.mdnet = MDNet().to(device)
        self.fake_cut = None
        self.eps = None

    def train_mdnet(
        self,
        train_paths: list[str],
        val_paths: list[str],
        epochs: int,
        lr: float,
        seed: int,
        log_every: int,
        output: str,
    ) -> None:
        """
        Train self.mdnet with Adam, one event per step on loss_bce + loss_mse, in an order shuffled by seed and
        epoch. Logs [train] lines every log_every steps and a [val] line per epoch, and saves the MDNet's
        state_dict as <output>/mdnet_epoch_<N>.pt after each epoch. (The loop of the former train.py.)
        """
        optimizer = torch.optim.Adam(self.mdnet.parameters(), lr=lr)
        for epoch in range(epochs):
            self.train_one_epoch(
                optimizer=optimizer,
                paths=train_paths,
                epoch=epoch,
                seed=seed,
                log_every=log_every,
            )
            self.validate(paths=val_paths, epoch=epoch)
            path = os.path.join(output, f"mdnet_epoch_{epoch}.pt")
            torch.save(self.mdnet.state_dict(), path)
            print(f"[checkpoint] path={path}")

    def train_one_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        paths: list[str],
        epoch: int,
        seed: int,
        log_every: int,
    ) -> None:
        """
        One pass over the training events in an order shuffled by seed and epoch, one optimizer step per event on
        loss_bce + loss_mse. Logs a [train] line every log_every steps, with the mean losses since the previous
        line.
        """
        self.mdnet.train()
        shuffled_paths = list(paths)
        random.Random(seed + epoch).shuffle(shuffled_paths)
        sum_bce, sum_mse, n_summed = 0.0, 0.0, 0
        start = time.time()
        for step, path in enumerate(shuffled_paths, start=1):
            loss_bce, loss_mse = self.event_losses(path=path)
            optimizer.zero_grad()
            (loss_bce + loss_mse).backward()
            optimizer.step()
            sum_bce += loss_bce.item()
            sum_mse += loss_mse.item()
            n_summed += 1
            if step % log_every == 0 or step == len(shuffled_paths):
                print(
                    f"[train] epoch={epoch} step={step} n_steps={len(shuffled_paths)}"
                    f" loss_bce={sum_bce / n_summed:.4f} loss_mse={sum_mse / n_summed:.4f}"
                    f" seconds={time.time() - start:.1f}"
                )
                sum_bce, sum_mse, n_summed = 0.0, 0.0, 0

    def validate(
        self,
        paths: list[str],
        epoch: int,
    ) -> None:
        """The mean losses over the validation events, without gradients, logged as one [val] line."""
        self.mdnet.eval()
        sum_bce, sum_mse = 0.0, 0.0
        with torch.no_grad():
            for path in paths:
                loss_bce, loss_mse = self.event_losses(path=path)
                sum_bce += loss_bce.item()
                sum_mse += loss_mse.item()
        print(
            f"[val] epoch={epoch} n_events={len(paths)}"
            f" loss_bce={sum_bce / len(paths):.4f} loss_mse={sum_mse / len(paths):.4f}"
        )

    def event_losses(self, path: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        loss_bce and loss_mse of one event: read it, make its inputs and targets, run the MDNet, and compare.
        Shared by training and validation.
        """
        mds = read_event(path=path)
        md_simIsFake, md_simKinematics = md_targets(mds=mds, pt_min=self.pt_min)
        md_predIsFakeLogit, md_predKinematics = self.mdnet(
            md_inputs(md_features=mds.md_features).to(self.device)
        )
        return loss_terms(
            md_predIsFakeLogit=md_predIsFakeLogit,
            md_predKinematics=md_predKinematics,
            md_simIsFake=md_simIsFake.to(self.device),
            md_simKinematics=md_simKinematics.to(self.device),
        )

    def choose_fake_cut(self, paths: list[str]) -> None:
        """
        Set self.fake_cut to the cut that keeps 50% of the non-fake MDs (md_simIsFake False) in paths: the median
        of their predicted p(fake). Logs a [fake_cut] line.
        """
        real_probs = []
        for path in paths:
            mds = read_event(path=path)
            md_predProbFake, _ = self.predict(mds=mds)
            md_simIsFake, _ = md_targets(mds=mds, pt_min=self.pt_min)
            real_probs.append(md_predProbFake[~md_simIsFake.numpy()])
        real_probs = np.concatenate(real_probs)
        self.fake_cut = float(np.median(real_probs))
        print(
            f"[fake_cut] fake_cut={self.fake_cut:.4f} n_events={len(paths)}"
            f" n_real_mds={len(real_probs)}"
        )

    def choose_eps(self, paths: list[str]) -> None:
        """
        Set self.eps so that 50% of the non-fake MDs passing self.fake_cut in paths would be DBSCAN core points
        from their own sim alone: the median, over those MDs, of the predicted-kinematics distance to the
        (MIN_MDS - 1)-th nearest other kept non-fake MD of the same sim (DBSCAN counts the MD itself among its
        MIN_MDS). MDs whose sim has fewer than MIN_MDS such MDs count as never core (distance infinity).
        Logs an [eps] line.
        """
        distances = []
        for path in paths:
            mds = read_event(path=path)
            md_predProbFake, md_predKinematics = self.predict(mds=mds)
            md_simIsFake, _ = md_targets(mds=mds, pt_min=self.pt_min)
            md_isKeptReal = (md_predProbFake < self.fake_cut) & ~md_simIsFake.numpy()
            kept_sims = mds.md_simIdx.numpy()[md_isKeptReal]
            kept_kinematics = md_predKinematics[md_isKeptReal]
            for sim in np.unique(kept_sims):
                sim_kinematics = kept_kinematics[kept_sims == sim]
                if len(sim_kinematics) < MIN_MDS:
                    distances.extend([np.inf] * len(sim_kinematics))
                    continue
                # Each MD is its own nearest neighbour (column 0), so column MIN_MDS - 1 is the
                # (MIN_MDS - 1)-th nearest other MD
                neighbour_distances, _ = (
                    NearestNeighbors(n_neighbors=MIN_MDS)
                    .fit(sim_kinematics)
                    .kneighbors(sim_kinematics)
                )
                distances.extend(neighbour_distances[:, MIN_MDS - 1])
        self.eps = float(np.median(distances))
        print(
            f"[eps] eps={self.eps:.4f} n_events={len(paths)} n_kept_real_mds={len(distances)}"
        )

    def predict(self, mds: MDsFromOneEvent) -> tuple[np.ndarray, np.ndarray]:
        """
        Each MD's predicted fake probability, md_predProbFake (n_md,) = sigmoid(md_predIsFakeLogit), and
        md_predKinematics (n_md, len(SIM_KINEMATICS)), as numpy arrays on the CPU. No gradients.
        """
        self.mdnet.eval()
        with torch.no_grad():
            md_predIsFakeLogit, md_predKinematics = self.mdnet(
                md_inputs(md_features=mds.md_features).to(self.device)
            )
        md_predProbFake = torch.sigmoid(md_predIsFakeLogit)
        return md_predProbFake.cpu().numpy(), md_predKinematics.cpu().numpy()

    def trackcands(self, paths: list[str]) -> list[TrackCandidatesFromOneEvent]:
        """
        The track candidates of each event in paths, in the same order: predict, cut at self.fake_cut, cluster
        with self.eps (find_mdclusters), and build them (trackcands_from_mdclusters).
        """
        event_trackcands = []
        for path in paths:
            mds = read_event(path=path)
            md_predProbFake, md_predKinematics = self.predict(mds=mds)
            md_mdclusterIdx = find_mdclusters(
                md_coordinates=md_predKinematics,
                md_isKept=md_predProbFake < self.fake_cut,
                eps=self.eps,
            )
            event_trackcands.append(
                trackcands_from_mdclusters(
                    mds=mds,
                    md_mdclusterIdx=md_mdclusterIdx,
                    md_predProbFake=md_predProbFake,
                    md_predKinematics=md_predKinematics,
                )
            )
        return event_trackcands

    def save(self, path: str) -> None:
        """
        Save the whole algorithm at path: the MDNet's state_dict, fake_cut, eps and pt_min, as one dict
        (readable with torch.load(path, weights_only=True)). Logs a [save] line.
        """
        saved = {
            "mdnet_state_dict": self.mdnet.state_dict(),
            "fake_cut": self.fake_cut,
            "eps": self.eps,
            "pt_min": self.pt_min,
        }
        torch.save(saved, path)
        print(
            f"[save] path={path} fake_cut={self.fake_cut:.4f} eps={self.eps:.4f} pt_min={self.pt_min}"
        )

    @classmethod
    def load(
        cls,
        path: str,
        device: str = "cuda",
    ) -> "SimpleMDNet":
        """A SimpleMDNet saved with save, on device."""
        saved = torch.load(path, weights_only=True, map_location=device)
        simple_mdnet = cls(device=device, pt_min=saved["pt_min"])
        simple_mdnet.mdnet.load_state_dict(saved["mdnet_state_dict"])
        simple_mdnet.fake_cut = saved["fake_cut"]
        simple_mdnet.eps = saved["eps"]
        return simple_mdnet


def md_inputs(md_features: torch.Tensor) -> torch.Tensor:
    """
    The network inputs of each MD, shape (n_md, N_INPUTS): the layer one-hot, then each FEATURE_SCALES feature
    divided by its scale.
    """
    layer = md_features[:, MD_FEATURES.index("layer")].long()
    # Layers 1-11 -> one-hot columns 0-10
    layer_onehot = nn.functional.one_hot(
        layer - FIRST_LAYER,
        num_classes=N_LAYERS,
    ).float()
    scaled = [
        md_features[:, MD_FEATURES.index(name)] / scale
        for name, scale in FEATURE_SCALES.items()
    ]
    return torch.cat([layer_onehot, torch.stack(scaled, dim=1)], dim=1)


def md_targets(
    mds: MDsFromOneEvent,
    pt_min: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Each MD's targets, from its sim: md_simIsFake, shape (n_md,), and md_simKinematics, shape
    (n_md, len(SIM_KINEMATICS)).

    An MD is fake if it is not matched to a sim (md_simIdx < 0) or its sim has pT < pt_min. md_simKinematics of
    fake MDs are meaningless (computed from UNMATCHED sim features) and must be left out of the loss.
    """
    md_sim_pt = mds.md_simFeatures[:, SIM_FEATURES.index("sim_pt")]
    md_sim_eta = mds.md_simFeatures[:, SIM_FEATURES.index("sim_eta")]
    md_sim_phi = mds.md_simFeatures[:, SIM_FEATURES.index("sim_phi")]
    md_sim_q = mds.md_simFeatures[:, SIM_FEATURES.index("sim_q")]
    md_simIsFake = (mds.md_simIdx < 0) | (md_sim_pt < pt_min)
    md_simKinematics = torch.stack(
        [
            md_sim_q / md_sim_pt,
            md_sim_eta,
            torch.cos(md_sim_phi),
            torch.sin(md_sim_phi),
        ],
        dim=1,
    )
    return md_simIsFake, md_simKinematics


class MDNet(nn.Module):
    """
    MLP on one MD's inputs, N_INPUTS -> N_HIDDEN -> N_HIDDEN -> N_HIDDEN -> N_TARGETS.

    forward returns the predicted fake logit, shape (n_md,), and the predicted SIM_KINEMATICS, shape
    (n_md, len(SIM_KINEMATICS)).
    """

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(N_INPUTS, N_HIDDEN),
            nn.ReLU(),
            nn.Linear(N_HIDDEN, N_HIDDEN),
            nn.ReLU(),
            nn.Linear(N_HIDDEN, N_HIDDEN),
            nn.ReLU(),
            nn.Linear(N_HIDDEN, N_TARGETS),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.mlp(inputs)
        # Column 0 is the fake logit, the rest are SIM_KINEMATICS
        return output[:, 0], output[:, 1:]


def loss_terms(
    md_predIsFakeLogit: torch.Tensor,
    md_predKinematics: torch.Tensor,
    md_simIsFake: torch.Tensor,
    md_simKinematics: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    The BCE between the predicted fake logit (md_predIsFakeLogit) and md_simIsFake, averaged over every MD, and
    the MSE between md_predKinematics and md_simKinematics, averaged over the MDs that are not fake (from the
    sim) and over SIM_KINEMATICS. Training minimizes their sum.
    """
    loss_bce = nn.functional.binary_cross_entropy_with_logits(
        md_predIsFakeLogit,
        md_simIsFake.float(),
    )
    md_simIsReal = ~md_simIsFake
    loss_mse = nn.functional.mse_loss(
        md_predKinematics[md_simIsReal],
        md_simKinematics[md_simIsReal],
    )
    return loss_bce, loss_mse


if __name__ == "__main__":
    main()
