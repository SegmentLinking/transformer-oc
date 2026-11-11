"""
Plot eta/phi distances between sibling MDs, to inform dr_threshold and max_neighbors for sparse attention on MDs.

The siblings of an MD with sim index Y are the other MDs with sim index Y, on any layer.
For each MD, the distances |deta|, |dphi| and dR to each sibling use the eta/phi of the inner hits.
The next-layer sibling of an MD is its nearest sibling on the lowest layer number above the MD's own layer
(so barrel layer 6 or a barrel layer skipped by a forward track continues into the endcap layers 7-11).
The k of a sibling is its rank among all MDs of the event (signal and background) sorted by distance to the MD,
i.e. the number of neighbours needed to include it.
MDs are grouped by their own layer and by the pT of their sim particle.

Plots, one figure per variable, one panel per layer, one line per pT bin:
    sibling_fraction_cdf_<var>: mean fraction of an MD's siblings within <var> < Z, vs Z
    farthest_sibling_<var>_{hist,cdf}: <var> of an MD's farthest sibling
    farthest_sibling_k_<var>_{hist,cdf}: k of an MD's farthest sibling
    next_layer_sibling_<var>_{hist,cdf}: <var> of an MD's next-layer sibling
    next_layer_sibling_k_<var>_{hist,cdf}: k of an MD's next-layer sibling

Usage:
    python md_sibling_plots.py --input transformer-datasets/samples/ttbar/train/event_*.pt --output md_plots/
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree

LAYERS = list(range(1, 12))
PT_BINS = [
    (0.8, 1.0),
    (1.0, 2.0),
    (2.0, 3.0),
    (3.0, 4.0),
]
VARS = ["dR", "deta", "dphi"]
LABELS = {
    "dR": r"$\Delta R$",
    "deta": r"$|\Delta\eta|$",
    "dphi": r"$|\Delta\phi|$",
}
# Log bins; values below the first edge (e.g. exact zeros) are clipped into the first bin
BINS = np.logspace(-4, np.log10(4), 101)
K_BINS = np.logspace(0, 5, 101)
# Offset that makes eta positive, for the KD-tree's periodic box
ETA_OFFSET = 50


def tree_coordinates(
    eta: np.ndarray,
    phi: np.ndarray,
    var: str,
) -> tuple[np.ndarray, list[float] | None]:
    """Coordinates and periodic box for a KD-tree whose euclidean distance is <var> (phi wraps around)."""
    phi = np.mod(phi, 2 * np.pi)
    if var == "dR":
        return np.stack([eta + ETA_OFFSET, phi], axis=1), [4 * ETA_OFFSET, 2 * np.pi]
    if var == "deta":
        return (eta + ETA_OFFSET)[:, None], None
    return phi[:, None], [2 * np.pi]


def k_needed(
    eta: np.ndarray,
    phi: np.ndarray,
    var: str,
    query: np.ndarray,
    radius: np.ndarray,
) -> np.ndarray:
    """
    For each MD index in query, the number of other MDs of the event within radius (in <var>) of it.

    With radius = the distance to a sibling, this is the k of that sibling.
    """
    coords, boxsize = tree_coordinates(eta=eta, phi=phi, var=var)
    tree = cKDTree(coords, boxsize=boxsize)
    # Small margin so the sibling itself is not lost to rounding; subtract 1 for the MD itself
    n_within = tree.query_ball_point(coords[query], r=radius * (1 + 1e-9), return_length=True)
    return n_within - 1


def event_siblings(
    path: str,
    hardscatter: bool,
) -> tuple[dict, dict]:
    """
    Sibling distances for one event file.

    With hardscatter, only MDs of sim particles with sim_event == 0 and sim_bunchCrossing == 0 are kept.
    The k ranking always counts every MD of the event.

    Returns (pairs, mds), dicts of arrays:
        pairs: one entry per (MD, sibling) with keys layer, pt, weight (1 / n_siblings), dR, deta, dphi
        mds: one entry per MD with keys layer, pt, and for each var:
            farthest_<var>, farthest_k_<var>, next_<var>, next_k_<var> (NaN when there is no next-layer sibling)
    Only MDs whose sim pT is inside PT_BINS and that have at least one sibling are kept.
    """
    data = torch.load(path, weights_only=True)
    x = data["x"].numpy().astype(np.float64)
    sim_index = data["sim_index"].numpy()
    sim_pt = data["sim_features"][:, 0].numpy()
    sim_event = data["sim_features"][:, 8].numpy()
    sim_bunch_crossing = data["sim_features"][:, 9].numpy()

    layer = x[:, 0].astype(int)
    eta = np.arcsinh(x[:, 4] / x[:, 1])
    phi = np.arctan2(x[:, 3], x[:, 2])

    pairs = {key: [] for key in ["layer", "pt", "weight"] + VARS}
    mds = {key: [] for key in ["index", "layer", "pt"]}
    for var in VARS:
        mds[f"farthest_{var}"] = []
        mds[f"next_{var}"] = []

    # Split the MD indices into groups of equal sim index
    order = np.argsort(sim_index, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(sim_index[order])) + 1)

    for idx in groups:
        n = len(idx)
        pt = sim_pt[idx[0]]
        if sim_index[idx[0]] < 0 or n < 2:
            continue
        if pt < PT_BINS[0][0] or pt >= PT_BINS[-1][1]:
            continue
        if hardscatter and (sim_event[idx[0]] != 0 or sim_bunch_crossing[idx[0]] != 0):
            continue

        # (n, n) matrices: row i holds the distances from MD i to every MD of the group
        deta = np.abs(eta[idx][:, None] - eta[idx][None, :])
        dphi = np.abs((phi[idx][:, None] - phi[idx][None, :] + np.pi) % (2 * np.pi) - np.pi)
        dist = {
            "dR": np.hypot(deta, dphi),
            "deta": deta,
            "dphi": dphi,
        }
        off_diagonal = ~np.eye(n, dtype=bool)

        # Row i: siblings on the lowest layer above MD i's layer (none if MD i is on the group's highest layer)
        group_layer = layer[idx]
        higher = group_layer[None, :] > group_layer[:, None]
        next_layer = np.where(higher, group_layer[None, :], np.iinfo(int).max).min(axis=1)
        on_next_layer = higher & (group_layer[None, :] == next_layer[:, None])

        pairs["layer"].append(np.repeat(group_layer, n - 1))
        pairs["pt"].append(np.full(n * (n - 1), pt))
        pairs["weight"].append(np.full(n * (n - 1), 1 / (n - 1)))
        mds["index"].append(idx)
        mds["layer"].append(group_layer)
        mds["pt"].append(np.full(n, pt))
        for var in VARS:
            pairs[var].append(dist[var][off_diagonal])
            mds[f"farthest_{var}"].append(dist[var].max(axis=1))
            nearest_next = np.where(on_next_layer, dist[var], np.inf).min(axis=1)
            mds[f"next_{var}"].append(np.where(np.isfinite(nearest_next), nearest_next, np.nan))

    pairs = {key: np.concatenate(values) for key, values in pairs.items()}
    mds = {key: np.concatenate(values) for key, values in mds.items()}

    for var in VARS:
        mds[f"farthest_k_{var}"] = k_needed(
            eta=eta,
            phi=phi,
            var=var,
            query=mds["index"],
            radius=mds[f"farthest_{var}"],
        ).astype(np.float64)
        has_next = np.isfinite(mds[f"next_{var}"])
        mds[f"next_k_{var}"] = np.full(len(has_next), np.nan)
        mds[f"next_k_{var}"][has_next] = k_needed(
            eta=eta,
            phi=phi,
            var=var,
            query=mds["index"][has_next],
            radius=mds[f"next_{var}"][has_next],
        )

    del mds["index"]
    pairs = {key: values.astype(np.float32) for key, values in pairs.items()}
    mds = {key: values.astype(np.float32) for key, values in mds.items()}
    return pairs, mds


def layer_grid(title: str) -> tuple[plt.Figure, list[plt.Axes]]:
    """A 3x4 grid of panels: one per layer, and the last one for all layers together."""
    fig, axs = plt.subplots(
        nrows=3,
        ncols=4,
        figsize=(16, 11),
        sharex=True,
    )
    fig.suptitle(title)
    return fig, axs.flatten()


def selection(
    arrays: dict,
    layer: int | None,
    pt_bin: tuple[float, float],
) -> np.ndarray:
    """Mask for one layer (None = all layers) and one pT bin."""
    mask = (arrays["pt"] >= pt_bin[0]) & (arrays["pt"] < pt_bin[1])
    if layer is not None:
        mask &= arrays["layer"] == layer
    return mask


def plot_cdf(
    ax: plt.Axes,
    values: np.ndarray,
    weights: np.ndarray,
    bins: np.ndarray,
    label: str,
) -> None:
    """Cumulative sum of a weighted histogram, drawn as a line at the upper bin edges."""
    counts, _ = np.histogram(values, bins=bins, weights=weights)
    ax.plot(bins[1:], np.cumsum(counts), label=label)


def save(
    fig: plt.Figure,
    path: str,
) -> None:
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"), dpi=100)
    plt.close(fig)
    print(f"[done] output={path}")


def plot_sibling_fraction(
    pairs: dict,
    mds: dict,
    var: str,
    output: str,
) -> None:
    """Mean fraction of an MD's siblings within <var> < Z, vs Z."""
    fig, axs = layer_grid(title=f"Mean fraction of siblings within {LABELS[var]}")
    for ax, layer in zip(axs, LAYERS + [None]):
        for pt_bin in PT_BINS:
            md_mask = selection(arrays=mds, layer=layer, pt_bin=pt_bin)
            pair_mask = selection(arrays=pairs, layer=layer, pt_bin=pt_bin)
            n_md = md_mask.sum()
            if n_md == 0:
                continue
            # Each pair weighs 1 / (n_siblings * n_md), so the cumulative sum is the mean per-MD fraction
            plot_cdf(
                ax=ax,
                values=np.clip(pairs[var][pair_mask], BINS[0], None),
                weights=pairs["weight"][pair_mask] / n_md,
                bins=BINS,
                label=f"{pt_bin[0]} < pT < {pt_bin[1]} GeV",
            )
        ax.set_title("All layers" if layer is None else f"Layer {layer}")
        ax.set_xscale("log")
        ax.set_xlabel(LABELS[var])
        ax.set_ylabel("Fraction")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        if layer is None:
            ax.legend(fontsize="small")
    save(fig=fig, path=os.path.join(output, f"sibling_fraction_cdf_{var}.pdf"))


def plot_per_md(
    mds: dict,
    key: str,
    bins: np.ndarray,
    title: str,
    xlabel: str,
    name: str,
    output: str,
) -> None:
    """Histogram and CDF of the per-MD quantity mds[key]; MDs where it is NaN are skipped."""
    hist_fig, hist_axs = layer_grid(title=title)
    cdf_fig, cdf_axs = layer_grid(title=f"Cumulative: {title}")
    for hist_ax, cdf_ax, layer in zip(hist_axs, cdf_axs, LAYERS + [None]):
        for pt_bin in PT_BINS:
            label = f"{pt_bin[0]} < pT < {pt_bin[1]} GeV"
            mask = selection(arrays=mds, layer=layer, pt_bin=pt_bin) & np.isfinite(mds[key])
            n_md = mask.sum()
            if n_md == 0:
                continue
            values = np.clip(mds[key][mask], bins[0], None)
            weights = np.full(n_md, 1 / n_md)
            hist_ax.hist(
                values,
                bins=bins,
                weights=weights,
                histtype="step",
                label=label,
            )
            plot_cdf(
                ax=cdf_ax,
                values=values,
                weights=weights,
                bins=bins,
                label=label,
            )
            q90, q99 = np.quantile(mds[key][mask], [0.90, 0.99])
            print(
                f"[quantile] key={key} layer={'all' if layer is None else layer} pt={pt_bin[0]}-{pt_bin[1]}"
                f" n_md={n_md} q90={q90:.4g} q99={q99:.4g}"
            )
        for ax in [hist_ax, cdf_ax]:
            ax.set_title("All layers" if layer is None else f"Layer {layer}")
            ax.set_xscale("log")
            ax.set_xlabel(xlabel)
            ax.grid(alpha=0.3)
            if layer is None:
                ax.legend(fontsize="small")
        hist_ax.set_ylabel("Fraction of MDs")
        cdf_ax.set_ylabel("Fraction")
        cdf_ax.set_ylim(0, 1.02)
    save(fig=hist_fig, path=os.path.join(output, f"{name}_hist.pdf"))
    save(fig=cdf_fig, path=os.path.join(output, f"{name}_cdf.pdf"))


def make_plots(
    pairs: dict,
    mds: dict,
    output: str,
) -> None:
    for var in VARS:
        plot_sibling_fraction(
            pairs=pairs,
            mds=mds,
            var=var,
            output=output,
        )
        # (key in mds, bins, title, x label, output name)
        specs = [
            (
                f"farthest_{var}",
                BINS,
                f"Farthest sibling {LABELS[var]}",
                LABELS[var],
                f"farthest_sibling_{var}",
            ),
            (
                f"farthest_k_{var}",
                K_BINS,
                f"k of farthest sibling, ranked by {LABELS[var]}",
                "k",
                f"farthest_sibling_k_{var}",
            ),
            (
                f"next_{var}",
                BINS,
                f"Next-layer sibling {LABELS[var]}",
                LABELS[var],
                f"next_layer_sibling_{var}",
            ),
            (
                f"next_k_{var}",
                K_BINS,
                f"k of next-layer sibling, ranked by {LABELS[var]}",
                "k",
                f"next_layer_sibling_k_{var}",
            ),
        ]
        for key, bins, title, xlabel, name in specs:
            plot_per_md(
                mds=mds,
                key=key,
                bins=bins,
                title=title,
                xlabel=xlabel,
                name=name,
                output=output,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=str, nargs="+", required=True, help="MD event files (event_*.pt)")
    parser.add_argument("--output", type=str, default="md_plots", help="Output directory")
    parser.add_argument("--n_events", type=int, default=50, help="Max events to read (-1=all)")
    parser.add_argument(
        "--hardscatter",
        action="store_true",
        help="Only MDs of sim particles with sim_event==0 and sim_bunchCrossing==0 (k still counts all MDs)",
    )
    args = parser.parse_args()

    paths = args.input if args.n_events == -1 else args.input[: args.n_events]
    os.makedirs(args.output, exist_ok=True)

    all_pairs = []
    all_mds = []
    for path in paths:
        pairs, mds = event_siblings(path=path, hardscatter=args.hardscatter)
        all_pairs.append(pairs)
        all_mds.append(mds)
        print(f"[read] input={path} n_md={len(mds['pt'])} n_pairs={len(pairs['pt'])}")

    pairs = {key: np.concatenate([p[key] for p in all_pairs]) for key in all_pairs[0]}
    mds = {key: np.concatenate([m[key] for m in all_mds]) for key in all_mds[0]}
    make_plots(pairs=pairs, mds=mds, output=args.output)


if __name__ == "__main__":
    main()
