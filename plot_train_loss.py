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
VAL_COLOR = "#eb6834"


def main():
    args = options()
    log_path = os.path.join(args.lightning_dir, TRAIN_LOG)
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"Expected {log_path} in --lightning-dir")

    steps, losses, val_steps, val_losses = read_losses(log_path=log_path)
    if len(steps) == 0:
        raise ValueError(f"No stage=train lines in {log_path}")
    print(f"Read {len(steps)} training batches and {len(val_steps)} validations")

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
    ax.plot(
        val_steps,
        val_losses,
        color=VAL_COLOR,
        marker="o",
        linewidth=1.5,
        label="Validation (end of epoch)",
    )
    ax.set_xlabel("Training step (batches)")
    ax.set_ylabel("Loss")
    # ax.set_ylim([0.0, 2.0])
    ax.set_ylim([1e-3, 2.0])
    ax.semilogy()
    ax.legend()
    fig.savefig(args.pdf)
    print(f"Wrote {args.pdf}")


def read_losses(log_path):
    """
    Return arrays from a train.log: global training step and train_loss_step for each
    batch, and global training step and val_loss for each validation. A validation
    runs after an epoch's training, so it is placed at the last step of that epoch.
    """
    steps = []
    losses = []
    val_steps = []
    val_losses = []
    n_batches = None
    with open(log_path) as f:
        for line in f:
            if not line.startswith("stage="):
                continue
            fields = dict(item.split("=", maxsplit=1) for item in line.split())
            epoch = int(fields["epoch"])
            if fields["stage"] == "train":
                n_batches = int(fields["n_batches"])
                steps.append(epoch * n_batches + int(fields["batch"]))
                losses.append(float(fields["train_loss_step"]))
            elif fields["stage"] == "val" and n_batches is not None:
                val_steps.append((epoch + 1) * n_batches)
                val_losses.append(float(fields["val_loss"]))
    return np.array(steps), np.array(losses), np.array(val_steps), np.array(val_losses)


def options():
    parser = argparse.ArgumentParser(description="Plot training loss vs training step")
    parser.add_argument("--lightning-dir", required=True, help="A run's version_N dir")
    parser.add_argument("--pdf", required=True, help="Output PDF")
    parser.add_argument("--window", type=int, default=50, help="Running mean window")
    return parser.parse_args()


if __name__ == "__main__":
    main()
