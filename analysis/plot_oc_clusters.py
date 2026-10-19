"""Look at the transformer clusters (oc_*) of an LST standalone ntuple.

Two kinds of output:
  1. distributions.png: number of hits and number of layers per cluster, for sim-matched and fake clusters
     (all events of the ntuple, or the first --n-events).
  2. event<E>_cluster<C>.png: one display per chosen cluster of one event, in the x-y and r-z planes: the hits of
     the cluster with their layers, the hits of the reference sim track that the cluster missed, and the other
     hits of the event nearby (grey), which show where the detector layers are.
     The reference sim track is the matched one (oc_simIdx); for a fake cluster it is the sim track with most
     of the cluster's hits. This needs the tracking ntuple that LST was run on (--tracking), for the hit
     positions and the sim hits.
     If the LST ntuple was written with --md --ls --t3, every T3 of the cluster is drawn as a line through the
     anchor hits of its three mini-doublets.

Layers: B1-B6 barrel, E1-E5 endcap disks (the numbering of oc_nLayers: barrel 1-6, endcap 7-11).

    python3 analysis/plot_oc_clusters.py --ntuple lst.root --output plots                       # distributions only
    python3 analysis/plot_oc_clusters.py --ntuple lst.root --output plots --tracking trk.root \\
        --event 0 --pick random --n 6                                                          # + cluster displays
    ... --event 0 --clusters 12 57 301                                                          # chosen clusters
"""
import argparse
import os
import sys

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot
from matplotlib.lines import Line2D

SURFACE, INK, INK_SECONDARY, GRID, CONTEXT = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#c9c8c2"
BLUE, ORANGE = "#2a78d6", "#eb6834"
T3_LINE = "#52514e"


def layer_name(layer):
    return f"B{layer}" if layer <= 6 else f"E{layer - 6}"


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=10)


# ----------------------------------------------------------------------------------------------- distributions
def plot_distributions(tree, n_events, path):
    arrays = tree.arrays(["oc_hitIdxs", "oc_nLayers", "oc_isFake"], entry_stop=n_events)
    n_hits = ak.to_numpy(ak.flatten(ak.num(arrays["oc_hitIdxs"], axis=2)))
    n_layers = ak.to_numpy(ak.flatten(arrays["oc_nLayers"]))
    fake = ak.to_numpy(ak.flatten(arrays["oc_isFake"])) == 1
    n_events = len(arrays)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.6, 1]})
    series = [("Matched to a sim track", ~fake, BLUE), ("Fake", fake, ORANGE)]
    for ax, values, xlabel in ((axes[0], n_hits, "hits per cluster"), (axes[1], n_layers, "layers per cluster")):
        style_axes(ax)
        ax.grid(False, axis="x")
        lo, hi = values.min(), max(values.max(), values.min() + 1)
        if xlabel.startswith("hits"):
            hi = min(hi, int(np.percentile(values, 99.5)))  # the last bin collects the tail
        edges = np.arange(lo, hi + 2) - 0.5
        width = 0.4 if hi - lo >= 14 else 0.28
        for i, (label, mask, colour) in enumerate(series):
            counts = np.histogram(np.minimum(values[mask], hi), edges)[0] / n_events
            ax.bar(np.arange(lo, hi + 1) + (i - 0.5) * width, counts, width=width * 0.9, color=colour, label=label)
        step = 1 if hi - lo < 14 else 2
        ticks = np.arange(lo, hi + 1, step)
        ax.set_xticks(ticks, labels=[f"{t}+" if (t == hi and values.max() > hi) else str(t) for t in ticks])
        ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=11)
    axes[0].set_ylabel("clusters per event", color=INK_SECONDARY, fontsize=11)
    legend = fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=10, ncol=2, loc="upper right",
                        bbox_to_anchor=(0.985, 0.995))
    for text in legend.get_texts():
        text.set_color(INK)
    fig.suptitle("Hits and layers per transformer cluster", x=0.06, y=0.985, ha="left", color=INK, fontsize=14,
                 fontweight="bold")
    fig.text(0.06, 0.905, f"{n_events} events, {len(n_hits)} clusters ({100 * fake.mean():.1f}% fake)", ha="left",
             color=INK_SECONDARY, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)

    print(f"{n_events} events, {len(n_hits)} clusters ({len(n_hits) / n_events:.0f} per event), {100 * fake.mean():.1f}% fake")
    print("\n| | all | matched | fake |\n|---|---|---|---|")
    for name, values in (("hits", n_hits), ("layers", n_layers)):
        cells = [f"{np.mean(v):.1f} / {np.median(v):.0f}" for v in (values, values[~fake], values[fake])]
        print(f"| {name} per cluster, mean / median | " + " | ".join(cells) + " |")
    print("\n| layers | clusters | share | fake rate |\n|---|---|---|---|")
    for layers in np.unique(n_layers):
        mask = n_layers == layers
        print(f"| {layers} | {mask.sum()} | {100 * mask.mean():.1f}% | {100 * fake[mask].mean():.1f}% |")


# ----------------------------------------------------------------------------------------------- cluster display
def load_event(tree, tracking, event, tracking_entry):
    out = tree.arrays(
        ["oc_hitIdxs", "oc_nT3s", "oc_nLayers", "oc_pt", "oc_eta", "oc_phi", "oc_isFake", "oc_simIdx", "oc_pMatched",
         "sim_pt"],
        entry_start=event, entry_stop=event + 1,
    )[0]
    inp = tracking.arrays(
        ["ph2_x", "ph2_y", "ph2_z", "ph2_subdet", "ph2_layer", "ph2_simHitIdx", "simhit_simTrkIdx", "sim_pt", "sim_eta"],
        entry_start=tracking_entry, entry_stop=tracking_entry + 1,
    )[0]
    # T3 segments need T3 -> LS -> MD -> anchor hit position, i.e. the --md --ls --t3 branches
    t3_branches = ["oc_t3Idx", "t3_lsIdx0", "t3_lsIdx1", "ls_mdIdx0", "ls_mdIdx1", "md_anchor_x", "md_anchor_y", "md_anchor_z"]
    t3_anchors = None
    if all(name in tree.keys() for name in t3_branches):
        t3 = tree.arrays(t3_branches, entry_start=event, entry_stop=event + 1)[0]
        ls0, ls1 = ak.to_numpy(t3["t3_lsIdx0"]), ak.to_numpy(t3["t3_lsIdx1"])
        md0, md1 = ak.to_numpy(t3["ls_mdIdx0"]), ak.to_numpy(t3["ls_mdIdx1"])
        mds = np.stack([md0[ls0], md1[ls0], md1[ls1]], axis=1)  # [nT3, 3]
        x, y, z = (ak.to_numpy(t3[f"md_anchor_{c}"])[mds] for c in "xyz")
        t3_anchors = {"x": x, "y": y, "z": z, "r": np.hypot(x, y), "oc_t3Idx": t3["oc_t3Idx"]}
    else:
        print("no T3 segments in the displays: the LST ntuple was not written with --md --ls --t3")
    n_sim = len(out["sim_pt"])
    if not np.array_equal(ak.to_numpy(out["sim_pt"]), ak.to_numpy(inp["sim_pt"])[:n_sim]):
        sys.exit(f"tracking ntuple entry {tracking_entry} is not the event of LST ntuple entry {event}")
    hits = {
        "x": ak.to_numpy(inp["ph2_x"]), "y": ak.to_numpy(inp["ph2_y"]), "z": ak.to_numpy(inp["ph2_z"]),
        "layer": ak.to_numpy(inp["ph2_layer"]) + 6 * (ak.to_numpy(inp["ph2_subdet"]) == 4),
    }
    hits["r"] = np.hypot(hits["x"], hits["y"])
    hits["phi"] = np.arctan2(hits["y"], hits["x"])
    # sim tracks of every hit, and hits of every sim track
    simhit_trk = ak.to_numpy(inp["simhit_simTrkIdx"])
    hit_sims = [set(simhit_trk[list(simhits)].tolist()) - {-1} for simhits in ak.to_list(inp["ph2_simHitIdx"])]
    sim_hits = {}
    for hit, sims in enumerate(hit_sims):
        for sim in sims:
            sim_hits.setdefault(sim, []).append(hit)
    return out, inp, hits, hit_sims, sim_hits, t3_anchors


def plot_cluster(event, cluster, out, inp, hits, hit_sims, sim_hits, t3_anchors, path):
    cluster_hits = np.asarray(out["oc_hitIdxs"][cluster])
    fake = out["oc_isFake"][cluster] == 1
    # reference sim track: the matched one, else the one with most of the cluster's hits
    counts = {}
    for hit in cluster_hits:
        for sim in hit_sims[hit]:
            counts[sim] = counts.get(sim, 0) + 1
    reference = int(out["oc_simIdx"][cluster])
    if reference < 0:
        reference = max(counts, key=counts.get) if counts else None
    on_reference = np.array([reference in hit_sims[hit] for hit in cluster_hits]) if reference is not None else \
        np.zeros(len(cluster_hits), dtype=bool)
    missed = np.setdiff1d(sim_hits.get(reference, []), cluster_hits).astype(int)

    shown = np.concatenate([cluster_hits, missed])
    phi0 = np.arctan2(np.mean(hits["y"][cluster_hits]), np.mean(hits["x"][cluster_hits]))
    dphi = np.abs(np.arctan2(np.sin(hits["phi"] - phi0), np.cos(hits["phi"] - phi0)))
    # square x-y window around the hits, so that the aspect ratio is 1
    half = 0.5 * max(np.ptp(hits["x"][shown]), np.ptp(hits["y"][shown])) + 12
    x_mid = 0.5 * (hits["x"][shown].min() + hits["x"][shown].max())
    y_mid = 0.5 * (hits["y"][shown].min() + hits["y"][shown].max())
    x_lim, y_lim = (x_mid - half, x_mid + half), (y_mid - half, y_mid + half)
    z_lim = (hits["z"][shown].min() - 25, hits["z"][shown].max() + 25)
    r_lim = (max(0, hits["r"][shown].min() - 15), min(118, hits["r"][shown].max() + 15))

    fig, (ax_xy, ax_rz) = plt.subplots(1, 2, figsize=(13, 6), facecolor=SURFACE)
    for ax, (h, v), context in (
        (ax_xy, ("x", "y"), (hits["z"] > z_lim[0]) & (hits["z"] < z_lim[1])),
        (ax_rz, ("z", "r"), dphi < 0.15),
    ):
        style_axes(ax)
        ax.scatter(hits[h][context], hits[v][context], s=3, color=CONTEXT, linewidths=0, zorder=1, rasterized=True)
        if t3_anchors is not None:
            for t3 in t3_anchors["oc_t3Idx"][cluster]:
                ax.plot(t3_anchors[h][t3], t3_anchors[v][t3], color=T3_LINE, linewidth=1.3, alpha=0.55, zorder=2,
                        solid_capstyle="round")
        ax.scatter(hits[h][missed], hits[v][missed], s=62, facecolors="none", edgecolors=BLUE, linewidths=1.6, zorder=3)
        other = cluster_hits[~on_reference]
        ax.scatter(hits[h][other], hits[v][other], s=62, marker="s", color=ORANGE, edgecolors=SURFACE, linewidths=1.2, zorder=4)
        same = cluster_hits[on_reference]
        ax.scatter(hits[h][same], hits[v][same], s=62, marker="o", color=BLUE, edgecolors=SURFACE, linewidths=1.2, zorder=5)
        ax.set_xlabel(f"{h} [cm]", color=INK_SECONDARY, fontsize=11)
        ax.set_ylabel(f"{v} [cm]", color=INK_SECONDARY, fontsize=11)
    ax_xy.set_xlim(*x_lim)
    ax_xy.set_ylim(*y_lim)
    ax_xy.set_aspect("equal", adjustable="box")
    ax_rz.set_xlim(*z_lim)
    ax_rz.set_ylim(*r_lim)

    # one layer label per layer of the cluster, next to its hits in the r-z view
    layers = sorted(set(hits["layer"][cluster_hits].tolist()))
    for layer in layers:
        in_layer = cluster_hits[hits["layer"][cluster_hits] == layer]
        ax_rz.annotate(layer_name(layer), (hits["z"][in_layer].max(), hits["r"][in_layer].max()), xytext=(7, 5),
                       textcoords="offset points", color=INK, fontsize=10, fontweight="bold", zorder=6)

    handles = [] if t3_anchors is None else [
        Line2D([], [], color=T3_LINE, linewidth=1.3, alpha=0.55, label="T3 (through its three anchor hits)")
    ]
    handles += [
        Line2D([], [], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor=SURFACE, markersize=9,
               label="cluster hit on the reference sim track"),
        Line2D([], [], marker="s", color="none", markerfacecolor=ORANGE, markeredgecolor=SURFACE, markersize=9,
               label="cluster hit not on it"),
        Line2D([], [], marker="o", color="none", markerfacecolor="none", markeredgecolor=BLUE, markeredgewidth=1.6,
               markersize=8, label="sim track hit missed by the cluster"),
        Line2D([], [], marker="o", color="none", markerfacecolor=CONTEXT, markeredgecolor="none", markersize=4,
               label="other hits (x-y: same z range; r-z: within 0.15 rad in phi)"),
    ]
    legend = fig.legend(handles=handles, frameon=False, fontsize=9.5, ncol=2, loc="upper right", bbox_to_anchor=(0.99, 1.0))
    for text in legend.get_texts():
        text.set_color(INK)

    n_hits = len(cluster_hits)
    if reference is None:
        truth = "fake, no hit with a sim track"
    else:
        share = f"{int(on_reference.sum())}/{n_hits} hits"
        sim_info = f"sim track {reference} (pT {inp['sim_pt'][reference]:.2f} GeV, eta {inp['sim_eta'][reference]:.2f})"
        truth = f"{'fake; most hits from' if fake else 'matched to'} {sim_info}: {share}"
    fig.suptitle(f"Event {event}, cluster {cluster}", x=0.05, y=0.985, ha="left", color=INK, fontsize=14, fontweight="bold")
    fig.text(
        0.05, 0.90,
        f"{out['oc_nT3s'][cluster]} T3s, {n_hits} hits, {len(layers)} layers ({' '.join(layer_name(l) for l in layers)}); "
        f"pT {out['oc_pt'][cluster]:.2f} GeV, eta {out['oc_eta'][cluster]:.2f}, phi {out['oc_phi'][cluster]:.2f}\n{truth}",
        ha="left", va="top", color=INK_SECONDARY, fontsize=10, linespacing=1.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return truth, layers


def pick_clusters(out, how, n, seed):
    n_clusters = len(out["oc_nT3s"])
    fake = ak.to_numpy(out["oc_isFake"]) == 1
    candidates = {"random": np.arange(n_clusters), "first": np.arange(n_clusters), "fake": np.flatnonzero(fake),
                  "matched": np.flatnonzero(~fake)}.get(how)
    if how == "largest":
        return np.argsort(-ak.to_numpy(ak.num(out["oc_hitIdxs"])), kind="stable")[:n]
    if how == "first":
        return candidates[:n]
    return np.sort(np.random.default_rng(seed).choice(candidates, size=min(n, len(candidates)), replace=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ntuple", required=True, help="LST standalone ntuple with oc_* branches")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--tree", default="tree")
    parser.add_argument("--n-events", type=int, default=None, help="events used for the distributions (default all)")
    parser.add_argument("--tracking", default=None, help="tracking ntuple that LST was run on (for the cluster displays)")
    parser.add_argument("--tracking-tree", default="trackingNtuple/tree")
    parser.add_argument("--event", type=int, default=0, help="LST ntuple entry of the cluster displays")
    parser.add_argument("--tracking-entry", type=int, default=None, help="its tracking ntuple entry (default: the same)")
    parser.add_argument("--clusters", type=int, nargs="+", default=None, help="cluster indices to display")
    parser.add_argument("--pick", choices=["random", "first", "fake", "matched", "largest"], default="random",
                        help="how to choose clusters when --clusters is not given")
    parser.add_argument("--n", type=int, default=6, help="number of clusters to choose")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    tree = uproot.open(args.ntuple)[args.tree]
    plot_distributions(tree, args.n_events, os.path.join(args.output, "distributions.png"))

    if args.tracking is not None:
        tracking = uproot.open(args.tracking)[args.tracking_tree]
        entry = args.event if args.tracking_entry is None else args.tracking_entry
        out, inp, hits, hit_sims, sim_hits, t3_anchors = load_event(tree, tracking, args.event, entry)
        clusters = args.clusters if args.clusters is not None else pick_clusters(out, args.pick, args.n, args.seed)
        print(f"\ncluster displays, event {args.event}:")
        for cluster in clusters:
            path = os.path.join(args.output, f"event{args.event}_cluster{cluster}.png")
            truth, layers = plot_cluster(args.event, int(cluster), out, inp, hits, hit_sims, sim_hits, t3_anchors, path)
            print(f"  cluster {cluster}: {len(out['oc_hitIdxs'][cluster])} hits, {len(layers)} layers; {truth}")
    print(f"\nplots in {args.output}/")


if __name__ == "__main__":
    main()
