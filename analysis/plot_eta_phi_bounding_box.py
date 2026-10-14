"""
plot_eta_phi_bounding_box.py - draw the eta/phi mask window of each training event

For each training event in a run's train.log, make a 2D histogram of the event's T3
eta and phi (all T3s, before masking) and overlay the mask window used in training.
The run directory must contain train.log and hparams.yaml, which gives the half
widths and training data directory. One page per event.

You can run me like:
> apptainer exec /ceph/users/atuna/ml-tracking.sif python3 analysis/plot_eta_phi_bounding_box.py \
    --lightning-dir output/lightning_logs/version_79 --pdf bounding_boxes.pdf --num-events 10
"""
import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

# I hate this: replace after becoming a package
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(REPO_ROOT, "src"))
from constants import T3_ETA_COLUMN, T3_PHI_COLUMN, ETA_MIN, ETA_MAX, PHI_MIN, PHI_MAX

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

CMAP = "plasma_r"
BOX_COLOR = "black"
TRAIN_LOG = "train.log"
HPARAMS = "hparams.yaml"


def main():
    args = options()

    # Verify files exist in the lightning directory
    log_path = os.path.join(args.lightning_dir, TRAIN_LOG)
    hparams_path = os.path.join(args.lightning_dir, HPARAMS)
    for path in [log_path, hparams_path]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Expected {path} in --lightning-dir")
    with open(hparams_path) as f:
        hparams = yaml.safe_load(f)
    data_dir = hparams["train_data_dir"]
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(REPO_ROOT, data_dir)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Expected {data_dir} to exist")

    # Parse the log file
    events = read_events(log_path=log_path, num_events=args.num_events)
    print(f"Plotting {len(events)} events from {log_path}")

    # Plot each event and save to pdf
    with PdfPages(args.pdf) as pdf:
        for event in events:
            fig = plot_event(
                event=event,
                data_dir=data_dir,
                eta_half_width=hparams["eta_mask_half_width"],
                phi_half_width=hparams["phi_mask_half_width"],
                bins=args.bins,
            )
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Wrote {args.pdf}")


def read_events(log_path, num_events):
    """Return up to num_events training events (dicts) from a train.log, in order."""
    events = []
    with open(log_path) as f:
        for line in f:
            if not line.startswith("stage=train "):
                continue
            fields = dict(item.split("=", maxsplit=1) for item in line.split())
            per_event = zip(
                fields["file"].split(","),
                fields["n_t3s"].split(","),
                fields["eta_center"].split(","),
                fields["phi_center"].split(","),
            )
            for filename, n_t3s, eta_center, phi_center in per_event:
                events.append(
                    {
                        "epoch": int(fields["epoch"]),
                        "batch": int(fields["batch"]),
                        "n_batches": int(fields["n_batches"]),
                        "filename": filename,
                        "n_t3s": int(n_t3s),
                        "eta_center": float(eta_center),
                        "phi_center": float(phi_center),
                    }
                )
                if len(events) == num_events:
                    return events
    return events


def window_rectangles(eta_center, phi_center, eta_half_width, phi_half_width):
    """
    Return the mask window as (eta_lo, phi_lo, eta_width, phi_width) rectangles.
    A window crossing phi = +-pi is split into two rectangles. A half width of None
    means the window spans the full range in that coordinate.
    """
    if eta_half_width is None:
        eta_lo, eta_hi = ETA_MIN, ETA_MAX
    else:
        eta_lo, eta_hi = eta_center - eta_half_width, eta_center + eta_half_width

    if phi_half_width is None or phi_half_width >= math.pi:
        phi_ranges = [(PHI_MIN, PHI_MAX)]
    else:
        phi_lo, phi_hi = phi_center - phi_half_width, phi_center + phi_half_width
        if phi_lo < PHI_MIN:
            phi_ranges = [(phi_lo + 2 * math.pi, PHI_MAX), (PHI_MIN, phi_hi)]
        elif phi_hi > PHI_MAX:
            phi_ranges = [(phi_lo, PHI_MAX), (PHI_MIN, phi_hi - 2 * math.pi)]
        else:
            phi_ranges = [(phi_lo, phi_hi)]

    eta_width = eta_hi - eta_lo
    return [(eta_lo, phi_lo, eta_width, phi_hi - phi_lo) for phi_lo, phi_hi in phi_ranges]


def plot_event(event, data_dir, eta_half_width, phi_half_width, bins):
    graph = torch.load(
        os.path.join(data_dir, event["filename"]),
        map_location="cpu",
        weights_only=False,
    )
    eta = graph.x[:, T3_ETA_COLUMN].numpy()
    phi = graph.x[:, T3_PHI_COLUMN].numpy()

    fig, ax = plt.subplots()
    _, _, _, image = ax.hist2d(
        eta,
        phi,
        bins=bins,
        range=[[ETA_MIN, ETA_MAX],
               [PHI_MIN, PHI_MAX]],
        cmap=CMAP,
        cmin=1,
    )
    fig.colorbar(image, ax=ax, label="T3s per bin", pad=0.01)

    rectangles = window_rectangles(
        eta_center=event["eta_center"],
        phi_center=event["phi_center"],
        eta_half_width=eta_half_width,
        phi_half_width=phi_half_width,
    )
    for eta_lo, phi_lo, eta_width, phi_width in rectangles:
        box = Rectangle(
            (eta_lo, phi_lo),
            eta_width,
            phi_width,
            fill=False,
            edgecolor=BOX_COLOR,
            linewidth=2,
        )
        ax.add_patch(box)

    ax.set_xlim(ETA_MIN, ETA_MAX)
    ax.set_ylim(PHI_MIN, PHI_MAX)
    ax.set_xlabel("T3 $\\eta$")
    ax.set_ylabel("T3 $\\phi$")
    step = f"Epoch {event['epoch']} batch {event['batch']}/{event['n_batches']}"
    eta_center, phi_center = event["eta_center"], event["phi_center"]
    center = f"center ($\\eta$, $\\phi$) = ({eta_center:.3f}, {phi_center:.3f})"
    counts = f"{event['n_t3s']} of {len(eta)} T3s in window"
    ax.set_title(f"{step}: {event['filename']}\n{counts}, {center}", fontsize=10)
    return fig


def options():
    parser = argparse.ArgumentParser(description="Plot T3 eta/phi with mask windows")
    parser.add_argument("--lightning-dir", required=True, help="A run's version_N dir")
    parser.add_argument("--pdf", required=True, help="Output PDF, one page per event")
    parser.add_argument("--num-events", type=int, default=10, help="Events to plot")
    parser.add_argument("--bins", type=int, default=100, help="Bins in eta and phi")
    return parser.parse_args()


if __name__ == "__main__":
    main()
