"""
plot_train_loss.py - training loss as a function of training step

Reads train.log in a run's directory and plots the per-batch training loss
(train_loss_step) against the training step, counted in batches across epochs.
A running mean over --window batches is drawn on top, since per-batch losses
on masked events are noisy.

You can run me like:
> apptainer exec /ceph/users/atuna/ml-tracking.sif python3 analysis/plot_train_loss.py \
    --lightning-dir output/lightning_logs/version_88 --pdf train_loss.pdf
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

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

TRAIN_LOG = "train.log"
RAW_COLOR = "gray"
MEAN_COLOR = "#7e03a8"


def main():
    args = options()
    log_path = os.path.join(args.lightning_dir, TRAIN_LOG)
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"Expected {log_path} in --lightning-dir")

    steps, losses = read_train_loss(log_path=log_path)
    if len(steps) == 0:
        raise ValueError(f"No stage=train lines in {log_path}")
    print(f"Read {len(steps)} training batches from {log_path}")

    fig, ax = plt.subplots()
    ax.plot(steps, losses, color=RAW_COLOR, alpha=0.4, linewidth=0.8, label="Per batch")
    if len(losses) >= args.window:
        kernel = np.ones(args.window) / args.window
        running_mean = np.convolve(losses, kernel, mode="valid")
        ax.plot(
            steps[args.window - 1 :],
            running_mean,
            color=MEAN_COLOR,
            linewidth=2,
            label=f"Running mean ({args.window} batches)",
        )
    ax.set_xlabel("Training step (batches)")
    ax.set_ylabel("Training loss")
    ax.legend()
    fig.savefig(args.pdf)
    print(f"Wrote {args.pdf}")


def read_train_loss(log_path):
    """Return arrays of (global training step, train_loss_step) from a train.log."""
    steps = []
    losses = []
    with open(log_path) as f:
        for line in f:
            if not line.startswith("stage=train "):
                continue
            fields = dict(item.split("=", maxsplit=1) for item in line.split())
            epoch = int(fields["epoch"])
            n_batches = int(fields["n_batches"])
            batch = int(fields["batch"])
            steps.append(epoch * n_batches + batch)
            losses.append(float(fields["train_loss_step"]))
    return np.array(steps), np.array(losses)


def options():
    parser = argparse.ArgumentParser(description="Plot training loss vs training step")
    parser.add_argument("--lightning-dir", required=True, help="A run's version_N dir")
    parser.add_argument("--pdf", required=True, help="Output PDF")
    parser.add_argument("--window", type=int, default=50, help="Running mean window")
    return parser.parse_args()


if __name__ == "__main__":
    main()
