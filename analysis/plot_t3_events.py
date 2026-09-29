"""
Make a few plots of the input T3 data.
One file contains one event.
These files are sometimes called graphs btw.
Each file name looks like "graph_4127.pt"
"""
from glob import glob
import argparse
from typing import Any
import logging
logger = logging.getLogger(__name__)

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams

CMIN = 0.5


def main():
    args = arguments()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    plotter = T3EventPlotter(args.input, args.output, args.num_events)
    plotter.load()
    plotter.quick_check()
    plotter.plot()


def arguments():
    default_input = "/ceph/users/atuna/work/cms_tracking_ml/transformer-oc/data/pu200_t3/train/*.pt"
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=str, default=default_input, help="Input globbable file path")
    parser.add_argument("--output", type=str, default="t3_event_plots.pdf", help="Output PDF file for the plots")
    parser.add_argument("--num-events", type=int, default=10, help="Maximum number of events to plot")
    return parser.parse_args()


class T3EventPlotter:

    def __init__(self, input_glob, output_pdf, num_events) -> None:
        self.input_glob = input_glob
        self.output_pdf = output_pdf
        self.num_events = num_events


    def get_file_number(self, file_path: str) -> int:
        try:
            return int(file_path.split("_")[-1].split(".")[0])
        except ValueError:
            logger.error(f"Could not extract file number from {file_path}")
            raise


    def load(self) -> None:
        logger.info(f"Globbing files from {self.input_glob} ...")
        self.files = glob(self.input_glob)

        # sort files like "file_1.pt", "file_2.pt", ...
        self.files.sort(key=self.get_file_number)

        if len(self.files) == 0:
            raise FileNotFoundError(f"No files found for glob pattern: {self.input_glob}")
        elif len(self.files) > self.num_events:
            logger.info(f"Found {len(self.files)} files, requested {self.num_events}. Skipping the extra")
            self.files = self.files[:self.num_events]

        logger.info(f"Loading {len(self.files)} files ...")
        self.events = [torch.load(f, weights_only=False) for f in self.files]
        self.event_numbers = [self.get_file_number(f) for f in self.files]


    def quick_check(self) -> None:
        for i_event, event in enumerate(self.events):
            if i_event > 0:
                break
            for feature in t3.MD_VARS:
                # MD feature_1 should be:
                #  - different from MD feature_0, and
                #  - identical to feature_2
                if not feature.endswith("_0"):
                    continue
                feature_base = feature.replace("_0", "")
                feature_0 = event.x[:, getattr(t3, feature_base + "_0")]
                feature_1 = event.x[:, getattr(t3, feature_base + "_1")]
                feature_2 = event.x[:, getattr(t3, feature_base + "_2")]
                check_01 = torch.allclose(feature_0, feature_1)
                check_12 = torch.allclose(feature_1, feature_2)
                logger.info(f"{feature_base} check: 0==1 is {check_01}, 1==2 is {check_12}")
                assert not check_01, f"{feature_base} 1 should be different from 0"
                assert check_12, f"{feature_base} 1 should be identical to 2"


    def plot(self) -> None:
        self.bins = {}
        self.bins["pt"] = np.linspace(0, 10, 101)
        self.bins["eta"] = np.linspace(-6, 6, 121)
        self.bins["phi"] = np.linspace(-3.5, 3.5, 141)

        logger.info(f"Writing plots to {self.output_pdf} ...")
        with PdfPages(self.output_pdf) as pdf:

            logger.info(f"Plotting event eta-phi")
            for event, num in zip(self.events, self.event_numbers):
                self.plot_sim_eta_phi(event, num, pdf)

            for feature in ["pt", "eta", "phi"]:
                logger.info(f"Plotting sim {feature}")
                for event, num in zip(self.events, self.event_numbers):
                    self.plot_sim_feature(event, feature, num, pdf)

            logger.info(f"Plotting T3 eta-phi")
            for event, num in zip(self.events, self.event_numbers):
                self.plot_t3_eta_phi(event, num, pdf)

            for feature in ["pt", "eta", "phi"] + t3.LS_VARS + t3.MD_VARS:
                logger.info(f"Plotting T3 {feature}")
                for event, num in zip(self.events, self.event_numbers):
                    self.plot_t3_feature(event, feature, num, pdf)


    def plot_sim_eta_phi(self, event: Any, num: int, pdf: PdfPages) -> None:
        fig, ax = plt.subplots()
        _, _, _, im = ax.hist2d(event.sim_features[:, sim.eta],
                                event.sim_features[:, sim.phi],
                                bins=[self.bins["eta"], self.bins["phi"]],
                                cmin=CMIN,
                                )
        ax.set_xlabel("Sim eta")
        ax.set_ylabel("Sim phi")
        ax.text(0.1, 1.01, f"graph_{num}.pt", transform=ax.transAxes)
        fig.colorbar(im, ax=ax, pad=0.01, label="Sim particles")
        pdf.savefig(fig)
        plt.close(fig)


    def plot_sim_feature(self, event: Any, feature: str, num: int, pdf: PdfPages) -> None:
        fig, ax = plt.subplots()
        ax.hist(event.sim_features[:, getattr(sim, feature)], bins=self.bins[feature])
        ax.text(0.1, 1.01, f"graph_{num}.pt", transform=ax.transAxes)
        ax.set_xlabel(f"Sim {feature}")
        ax.set_ylabel("Sim particles")
        pdf.savefig(fig)
        plt.close(fig)


    def plot_t3_eta_phi(self, event: Any, num: int, pdf: PdfPages) -> None:
        fig, ax = plt.subplots()
        _, _, _, im = ax.hist2d(event.t3_features[:, t3.eta],
                                event.t3_features[:, t3.phi],
                                bins=[self.bins["eta"], self.bins["phi"]],
                                cmin=CMIN,
                                )
        ax.set_xlabel("T3 eta")
        ax.set_ylabel("T3 phi")
        ax.text(0.1, 1.01, f"graph_{num}.pt", transform=ax.transAxes)
        fig.colorbar(im, ax=ax, pad=0.01, label="T3s")
        pdf.savefig(fig)
        plt.close(fig)


    def plot_t3_feature(self, event: Any, feature: str, num: int, pdf: PdfPages) -> None:
        fig, ax = plt.subplots()
        bins = self.bins[feature] if feature in self.bins else 100
        ax.hist(event.x[:, getattr(t3, feature)], bins=bins)
        ax.text(0.1, 1.01, f"graph_{num}.pt", transform=ax.transAxes)
        ax.set_xlabel(f"T3 {feature}")
        ax.set_ylabel("T3s")
        pdf.savefig(fig)
        plt.close(fig)


#
# Keeping track of which columns correspond to which sim variables
#
class sim:
    SIM_VARS = ["sim_pt", "sim_eta", "sim_phi", "sim_q", "sim_vx", "sim_vy", "sim_vz", "sim_vtxperp"]
    pt = SIM_VARS.index("sim_pt")
    eta = SIM_VARS.index("sim_eta")
    phi = SIM_VARS.index("sim_phi")
    q = SIM_VARS.index("sim_q")
    vx = SIM_VARS.index("sim_vx")
    vy = SIM_VARS.index("sim_vy")
    vz = SIM_VARS.index("sim_vz")
    vtxperp = SIM_VARS.index("sim_vtxperp")


#
# Keeping track of which columns correspond to which t3 variables
#
class t3:
    T3_VARS = ["t3_pt", "t3_eta", "t3_phi"]
    LS_VARS = [
        "ls_dPhis_0", "ls_dPhiChanges_0", "ls_dAlphaInners_0", "ls_dAlphaOuters_0", "ls_dAlphaInnerOuters_0",
        "ls_dPhis_1", "ls_dPhiChanges_1", "ls_dAlphaInners_1", "ls_dAlphaOuters_1", "ls_dAlphaInnerOuters_1",
    ]
    MD_VARS = [
        "md_anchor_x_0", "md_anchor_y_0", "md_anchor_z_0", "md_other_x_0", "md_other_y_0", "md_other_z_0", "md_dphi_0", "md_dphichange_0", "md_dz_0",
        "md_anchor_x_1", "md_anchor_y_1", "md_anchor_z_1", "md_other_x_1", "md_other_y_1", "md_other_z_1", "md_dphi_1", "md_dphichange_1", "md_dz_1",
        "md_anchor_x_2", "md_anchor_y_2", "md_anchor_z_2", "md_other_x_2", "md_other_y_2", "md_other_z_2", "md_dphi_2", "md_dphichange_2", "md_dz_2",
        "md_anchor_x_3", "md_anchor_y_3", "md_anchor_z_3", "md_other_x_3", "md_other_y_3", "md_other_z_3", "md_dphi_3", "md_dphichange_3", "md_dz_3",
    ]
    ALL_VARS = T3_VARS + LS_VARS + MD_VARS
    pt = ALL_VARS.index("t3_pt")
    eta = ALL_VARS.index("t3_eta")
    phi = ALL_VARS.index("t3_phi")
    ls_dPhis_0 = ALL_VARS.index("ls_dPhis_0")
    ls_dPhiChanges_0 = ALL_VARS.index("ls_dPhiChanges_0")
    ls_dAlphaInners_0 = ALL_VARS.index("ls_dAlphaInners_0")
    ls_dAlphaOuters_0 = ALL_VARS.index("ls_dAlphaOuters_0")
    ls_dAlphaInnerOuters_0 = ALL_VARS.index("ls_dAlphaInnerOuters_0")
    ls_dPhis_1 = ALL_VARS.index("ls_dPhis_1")
    ls_dPhiChanges_1 = ALL_VARS.index("ls_dPhiChanges_1")
    ls_dAlphaInners_1 = ALL_VARS.index("ls_dAlphaInners_1")
    ls_dAlphaOuters_1 = ALL_VARS.index("ls_dAlphaOuters_1")
    ls_dAlphaInnerOuters_1 = ALL_VARS.index("ls_dAlphaInnerOuters_1")
    md_anchor_x_0 = ALL_VARS.index("md_anchor_x_0")
    md_anchor_y_0 = ALL_VARS.index("md_anchor_y_0")
    md_anchor_z_0 = ALL_VARS.index("md_anchor_z_0")
    md_other_x_0 = ALL_VARS.index("md_other_x_0")
    md_other_y_0 = ALL_VARS.index("md_other_y_0")
    md_other_z_0 = ALL_VARS.index("md_other_z_0")
    md_dphi_0 = ALL_VARS.index("md_dphi_0")
    md_dphichange_0 = ALL_VARS.index("md_dphichange_0")
    md_dz_0 = ALL_VARS.index("md_dz_0")
    md_anchor_x_1 = ALL_VARS.index("md_anchor_x_1")
    md_anchor_y_1 = ALL_VARS.index("md_anchor_y_1")
    md_anchor_z_1 = ALL_VARS.index("md_anchor_z_1")
    md_other_x_1 = ALL_VARS.index("md_other_x_1")
    md_other_y_1 = ALL_VARS.index("md_other_y_1")
    md_other_z_1 = ALL_VARS.index("md_other_z_1")
    md_dphi_1 = ALL_VARS.index("md_dphi_1")
    md_dphichange_1 = ALL_VARS.index("md_dphichange_1")
    md_dz_1 = ALL_VARS.index("md_dz_1")
    md_anchor_x_2 = ALL_VARS.index("md_anchor_x_2")
    md_anchor_y_2 = ALL_VARS.index("md_anchor_y_2")
    md_anchor_z_2 = ALL_VARS.index("md_anchor_z_2")
    md_other_x_2 = ALL_VARS.index("md_other_x_2")
    md_other_y_2 = ALL_VARS.index("md_other_y_2")
    md_other_z_2 = ALL_VARS.index("md_other_z_2")
    md_dphi_2 = ALL_VARS.index("md_dphi_2")
    md_dphichange_2 = ALL_VARS.index("md_dphichange_2")
    md_dz_2 = ALL_VARS.index("md_dz_2")
    md_anchor_x_3 = ALL_VARS.index("md_anchor_x_3")
    md_anchor_y_3 = ALL_VARS.index("md_anchor_y_3")
    md_anchor_z_3 = ALL_VARS.index("md_anchor_z_3")
    md_other_x_3 = ALL_VARS.index("md_other_x_3")
    md_other_y_3 = ALL_VARS.index("md_other_y_3")
    md_other_z_3 = ALL_VARS.index("md_other_z_3")
    md_dphi_3 = ALL_VARS.index("md_dphi_3")
    md_dphichange_3 = ALL_VARS.index("md_dphichange_3")
    md_dz_3 = ALL_VARS.index("md_dz_3")


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
