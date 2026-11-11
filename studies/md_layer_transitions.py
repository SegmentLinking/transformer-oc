"""
Plot which layer follows which along sim tracks, from MD dataset files.

For each sim track with pT >= --pt_min, its MDs are ordered by the 3D distance sqrt(rt^2 + z^2) of their inner hit
from the origin, and each consecutive pair of MDs counts as one transition (layer of first MD -> layer of second MD).
Two consecutive MDs on the same layer (e.g. overlapping modules) count as a transition on the diagonal.
The last MD of a track counts as a transition to "end".
Truth is MD-level (md_simIdx), so MDs with only one hit from the sim are not included.

Plots:
    layer_transitions_fraction: fraction of transitions from each layer (row) going to each layer (column)
    layer_transitions_count: number of transitions, log color scale

Usage:
    python md_layer_transitions.py --input transformer-datasets/samples/ttbar/train/event_*.pt --output md_plots/
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

N_LAYERS = 11
# Column index N_LAYERS + 1 is "end": no further MD on the track
END = N_LAYERS + 1
LABELS = [str(layer) for layer in range(1, N_LAYERS + 1)] + ["end"]


def event_transitions(
    layer: np.ndarray,
    rt: np.ndarray,
    z: np.ndarray,
    sim_index: np.ndarray,
    sim_pt: np.ndarray,
    pt_min: float,
) -> np.ndarray:
    """
    Transition counts of one event, shape (N_LAYERS + 2, N_LAYERS + 2), indexed by layer number (row 0 is unused)
    and END.
    """
    counts = np.zeros((N_LAYERS + 2, N_LAYERS + 2), dtype=np.int64)
    keep = (sim_index >= 0) & (sim_pt >= pt_min)
    layer = layer[keep]
    sim_index = sim_index[keep]
    r3d = np.hypot(rt[keep], z[keep])

    # Sort by sim, then by distance from the origin within each sim
    order = np.lexsort((r3d, sim_index))
    layer = layer[order]
    sim_index = sim_index[order]

    same_track = sim_index[:-1] == sim_index[1:]
    np.add.at(counts, (layer[:-1][same_track], layer[1:][same_track]), 1)

    # The last MD of each track goes to END
    is_last = np.append(~same_track, True)
    np.add.at(counts, (layer[is_last], END), 1)
    return counts


def plot_matrix(
    matrix: np.ndarray,
    title: str,
    colorbar_label: str,
    output_file: str,
    log: bool,
    fmt: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    if log:
        image = ax.imshow(np.where(matrix > 0, matrix, np.nan), cmap="viridis", norm=LogNorm())
    else:
        image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, label=colorbar_label)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if value == 0:
                continue
            ax.text(
                column,
                row,
                format(value, fmt),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if (log or value < 0.5) else "black",
            )

    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS)
    ax.set_yticks(range(N_LAYERS))
    ax.set_yticklabels(LABELS[:N_LAYERS])
    ax.set_xlabel("Next layer")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    # Barrel/endcap boundaries
    ax.axvline(5.5, color="red", linewidth=1)
    ax.axhline(5.5, color="red", linewidth=1)
    fig.tight_layout()
    fig.savefig(output_file + ".png", dpi=150)
    fig.savefig(output_file + ".pdf")
    plt.close(fig)
    print(f"[done] output={output_file}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--input", type=str, nargs="+", required=True)
    parser.add_argument("--output", type=str, default="md_plots/")
    parser.add_argument("--pt_min", type=float, default=0.8, help="Minimum sim pT [GeV]")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    counts = np.zeros((N_LAYERS + 2, N_LAYERS + 2), dtype=np.int64)
    for path in args.input:
        data = torch.load(path, weights_only=True)
        x = data["x"].numpy()
        counts += event_transitions(
            layer=x[:, 0].astype(np.int64),
            rt=x[:, 1],
            z=x[:, 4],
            sim_index=data["sim_index"].numpy(),
            sim_pt=data["sim_features"][:, 0].numpy(),
            pt_min=args.pt_min,
        )
    print(f"[info] n_events={len(args.input)} n_transitions={counts.sum()}")

    # Drop row/column 0 (no layer 0) and the unused END row
    counts = counts[1 : N_LAYERS + 1, 1:]
    fraction = counts / counts.sum(axis=1, keepdims=True)

    title_suffix = f"sim pT >= {args.pt_min} GeV, {len(args.input)} events"
    plot_matrix(
        matrix=fraction,
        title=f"Layer transitions along sim tracks, fraction per row\n{title_suffix}",
        colorbar_label="Fraction of transitions from this layer",
        output_file=os.path.join(args.output, "layer_transitions_fraction"),
        log=False,
        fmt=".2f",
    )
    plot_matrix(
        matrix=counts,
        title=f"Layer transitions along sim tracks, counts\n{title_suffix}",
        colorbar_label="Transitions",
        output_file=os.path.join(args.output, "layer_transitions_count"),
        log=True,
        fmt="d",
    )


if __name__ == "__main__":
    main()
