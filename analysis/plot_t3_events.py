"""
Make a few plots of the input T3 data.
One file contains one event.
These files are sometimes called graphs btw.
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


    def plot(self) -> None:
        logger.info(f"Writing plots to {self.output_pdf} ...")
        with PdfPages(self.output_pdf) as pdf:
            for event, num in zip(self.events, self.event_numbers):
                self.plot_event(event, num, pdf)


    def plot_event(self, event: Any, num: int, pdf: PdfPages) -> None:
        bins = [
            np.linspace(-5, 5, 100),
            np.linspace(-3.2, 3.2, 128),
        ]
        fig, ax = plt.subplots()
        _, _, _, im = ax.hist2d(event.sim_features[:, sim.eta],
                                event.sim_features[:, sim.phi],
                                bins=bins,
                                cmin=CMIN,
                                )
        ax.set_xlabel("Sim eta")
        ax.set_ylabel("Sim phi")
        fig.colorbar(im, ax=ax, pad=0.01, label="Sim particles")
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
