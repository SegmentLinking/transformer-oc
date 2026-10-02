"""
Make a few plots of T3s arising from a common truth particle (sibling T3s).

NB: we do truth-matching based on the most popular sim index for each T3 among its MDs.
"""
from glob import glob
import argparse
import logging
logger = logging.getLogger(__name__)

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams

MIN_PARENT_FRACTION = 0.5
BINS = {
    "pt": np.linspace(0, 100, 201),
    "1overpt": np.linspace(0, 2.0, 201),
    "eta": np.linspace(-3.0, 3.0, 301),
    "phi": np.linspace(-3.2, 3.2, 321),
    "dpt": np.linspace(0, 20, 201),
    "d1_over_pt": np.linspace(0, 1.5, 301),
    "deta": np.linspace(0, 0.5, 201),
    "dphi": np.linspace(0, 0.5, 201),
    "dr": np.linspace(0, 0.5, 201),
}
LABEL = {
    "pt": r"T3 $p_T$",
    "1overpt": r"T3 $1/p_T$",
    "eta": r"T3 $\eta$",
    "phi": r"T3 $\phi$",
    "dpt": r"$\Delta p_T$",
    "d1_over_pt": r"$\Delta (1/p_T)$",
    "deta": r"$\Delta \eta$",
    "dphi": r"$\Delta \phi$",
    "dr": r"$\Delta R$",
}

def main():
    args = arguments()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    plotter = T3SiblingPlotter(args.input, args.output, args.num_events)
    plotter.load()
    plotter.plot()


def arguments():
    default_input = "/ceph/users/atuna/work/cms_tracking_ml/transformer-oc/data/pu200_t3/train/*.pt"
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=str, default=default_input, help="Input globbable file path")
    parser.add_argument("--output", type=str, default="t3_sibling_plots.pdf", help="Output PDF file for the plots")
    parser.add_argument("--num-events", type=int, default=1, help="Maximum number of events to plot")
    return parser.parse_args()


class T3SiblingPlotter:

    def __init__(self, input_glob, output_pdf, num_events):
        self.input_glob = input_glob
        self.output_pdf = output_pdf
        self.num_events = num_events
        self.files = []


    def load(self):
        self.files = glob(self.input_glob)[:self.num_events]
        t3_features_list = []
        sibling_features_list = []
        for fname in self.files:
            t3_features, sibling_features = self.load_one(fname)
            t3_features_list.append(t3_features)
            sibling_features_list.append(sibling_features)
        self.t3_features = np.concatenate(t3_features_list, axis=0)
        self.sibling_features = np.concatenate(sibling_features_list, axis=0)
        self.derive_features()


    def load_one(self, fname: str) -> np.ndarray:
        logger.info(f"Loading file {fname} ...")
        data = torch.load(fname, weights_only=False)
        t3_features = data.x.numpy()
        parent_simIdx, parent_fraction = self.choose_parent(data.md_simIdx.numpy())
        # t3_layers = self.find_unique_layers(data.md_layer.numpy())
        sibling_pairs = self.find_sibling_pairs(parent_simIdx, parent_fraction)
        sibling_features = self.make_sibling_features(data.x, sibling_pairs)
        memory = sibling_features.nbytes / (1024**2) # memory usage in MB
        logger.info(f"Found {len(sibling_pairs)} sibling pairs out of {len(parent_simIdx)} T3s, using {memory:.1f} MB of memory")
        return t3_features, sibling_features


    def choose_parent(self, md_simIdx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Choose the parent T3 sim index, given a list of MD sim indices for each T3

        Args:
            md_simIdx (np.ndarray): [num_t3s, num_md] array of MD sim indices for each T3.

        Returns:
            tuple[np.ndarray, np.ndarray]: The most-popular MD sim index for each T3,
                                            and the fraction of MDs that agree with it.
        """
        num_t3s, num_md = md_simIdx.shape
        parent_simIdx = np.zeros(num_t3s, dtype=int)
        parent_fraction = np.zeros(num_t3s, dtype=float)
        for i_t3 in range(num_t3s):
            values, counts = np.unique(md_simIdx[i_t3], return_counts=True)
            max_idx = np.argmax(counts)
            parent_simIdx[i_t3] = values[max_idx]
            parent_fraction[i_t3] = counts[max_idx] / num_md
        return parent_simIdx, parent_fraction


    def find_unique_layers(self, md_layer: np.ndarray) -> np.ndarray:
        t3_layer = np.sort(np.unique(md_layer, axis=1), axis=1).astype(int)
        return t3_layer


    def find_sibling_pairs(self, parent_simIdx: np.ndarray, parent_fraction: np.ndarray) -> np.ndarray:
        """
        Find pairs of T3s that share the same parent sim index.

        Args:
            parent_simIdx (np.ndarray): [num_t3s] array of parent sim indices for each T3.
            parent_fraction (np.ndarray): [num_t3s] array of fractions of MDs that agree with the parent sim index for each T3.

        Returns:
            np.ndarray: [num_sibling_pairs, 2] array of indices of sibling T3 pairs.
        """
        logger.info("Finding sibling pairs ...")
        sibling_pairs = []
        valid = (parent_fraction >= MIN_PARENT_FRACTION) & (parent_simIdx >= 0)
        unique_parents = np.sort(np.unique(parent_simIdx[valid]))
        for parent in unique_parents:
            siblings = np.argwhere((parent_simIdx == parent) & valid).flatten()
            i, j = np.triu_indices(len(siblings), k=1)
            sibling_pairs.append(np.column_stack((siblings[i], siblings[j])))
        if not sibling_pairs:
            return np.empty((0, 2), dtype=int)
        return np.concatenate(sibling_pairs)


    def make_sibling_features(self, x: np.ndarray, sibling_pairs: np.ndarray) -> np.ndarray:
        """
        Create features for sibling pairs. Only consider three features: pt, eta, phi.

        Args:
            x (np.ndarray): [num_t3s, num_features] array of features for each T3.
            sibling_pairs (np.ndarray): [num_sibling_pairs, 2] array of indices of sibling T3 pairs.

        Returns:
            np.ndarray: [num_sibling_pairs, 2 * num_features] array of concatenated features for each sibling pair.
        """
        logger.info("Creating sibling features ...")
        feats = [t3.pt, t3.eta, t3.phi]
        subset = x[:, feats]
        return np.concatenate([subset[sibling_pairs[:, 0]], subset[sibling_pairs[:, 1]]], axis=1)


    def derive_features(self):
        logger.info("Deriving sibling features ...")
        self.derived_features = {}
        self.derived_features["pt"] = self.sibling_features[:, t3t3.pt0]
        self.derived_features["1overpt"] = 1 / self.sibling_features[:, t3t3.pt0]
        self.derived_features["eta"] = self.sibling_features[:, t3t3.eta0]
        self.derived_features["phi"] = self.sibling_features[:, t3t3.phi0]
        self.derived_features["dpt"] = np.abs(self.sibling_features[:, t3t3.pt0] - self.sibling_features[:, t3t3.pt1])
        self.derived_features["d1_over_pt"] = np.abs(1 / self.sibling_features[:, t3t3.pt0] - 1 / self.sibling_features[:, t3t3.pt1])
        self.derived_features["deta"] = np.abs(self.sibling_features[:, t3t3.eta0] - self.sibling_features[:, t3t3.eta1])
        self.derived_features["dphi"] = self.sibling_features[:, t3t3.phi0] - self.sibling_features[:, t3t3.phi1]
        self.derived_features["dphi"] = np.abs(np.mod(self.derived_features["dphi"] + np.pi, 2 * np.pi) - np.pi)
        self.derived_features["dr"] = np.sqrt(self.derived_features["deta"]**2 + self.derived_features["dphi"]**2)
        memory = self.sibling_features.nbytes / (1024**2)  # memory usage in MB
        logger.info(f"Derived sibling features using {memory:.1f} MB of memory")


    def plot(self):
        logger.info(f"Writing plots to {self.output_pdf} ...")
        feats = ["pt", "1overpt", "eta", "phi"]
        diffs = ["dpt", "d1_over_pt", "deta", "dphi", "dr"]
        with PdfPages(self.output_pdf) as pdf:
            for feat in feats:
                self.plot_feature(feat, pdf)
            for diff in diffs:
                self.plot_difference(diff, pdf)


    def plot_feature(self, feat: str, pdf: PdfPages):
        arr = self.derived_features[feat]
        fig, ax = plt.subplots()
        ax.hist(arr, bins=BINS[feat])
        ax.set_title(f"N = {len(arr)}")
        ax.set_xlabel(LABEL[feat])
        ax.set_ylabel("T3s")
        pdf.savefig(fig)
        plt.close(fig)


    def plot_difference(self, diff: str, pdf: PdfPages):
        arr = self.derived_features[diff]
        fig, ax = plt.subplots()
        ax.hist(arr, bins=BINS[diff])
        ax.set_title(f"N = {len(arr)}")
        ax.set_xlabel(LABEL[diff])
        ax.set_ylabel("Pairs of T3s with matching sim index")
        pdf.savefig(fig)
        plt.close(fig)


class t3:
    pt = 0
    eta = 1
    phi = 2


class t3t3:
    pt0 = 0
    eta0 = 1
    phi0 = 2
    pt1 = 3
    eta1 = 4
    phi1 = 5


#
# Beautifying the plots
#
rcParams.update({
    "font.size": 16,
    "figure.figsize": (8, 8),
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.grid": True,
    "axes.grid.which": "both",
    # "axes.axisbelow": True,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.1,
    "grid.color": "gray",
    "figure.subplot.left": 0.15,
    "figure.subplot.bottom": 0.09,
    "figure.subplot.right": 0.97,
    "figure.subplot.top": 0.95,
})


if __name__ == "__main__":
    main()
