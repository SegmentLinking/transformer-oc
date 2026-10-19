"""Transformer clusters (oc) vs LST T5 / T4 / T5+T4, from one LST standalone ntuple.

Input: ntuple(s) written by `lst --t5 --t4 --transformer_model <pt>` (oc_*, t5_*, t4_*, sim_* branches).
All object types use the LST hit matching of the ntuple (a sim track matches an object if it has a sim hit in
more than 75% of the object's hits), and one sim track selection for the efficiency denominator.

  efficiency      selected sim tracks with at least one matched object / selected sim tracks
  fake rate       objects without a matched sim track / objects
  duplicate rate  objects with a matched (hard-scatter) sim track that is matched by more than one object / objects

Sim tracks in the ntuple are the hard-scatter, in-time ones. Selection (as in LST's efficiency code):
q != 0, |vz| < 30 cm, vxy < 2.5 cm; |eta| < --eta-cut for the pt plots; pt > --pt-cut for the eta plots.
Objects: |eta| < --eta-cut for the pt plots; pt > --pt-cut for the eta plots.

    python3 analysis/compare_oc_t5_t4.py --input 'lst_oc*.root' --output output/comparison/plots
"""
import argparse
import glob
import json
import os
import sys

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot
from scipy.stats import beta

MATCH_FRAC = 0.75
PT_BINS = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0])
ETA_BINS = np.linspace(-2.6, 2.6, 27)

# name, label, object types, colour (fixed per entity), marker
SERIES = [
    ("oc", "Transformer clusters", ("oc",), "#2a78d6", "o"),
    ("t5", "T5", ("t5",), "#eb6834", "s"),
    ("t4", "T4", ("t4",), "#1baf7a", "^"),
    ("t5t4", "T5 + T4", ("t5", "t4"), "#eda100", "D"),
]
METRICS = [("efficiency", "Efficiency"), ("fake", "Fake rate"), ("duplicate", "Duplicate rate")]
SURFACE, INK, INK_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def take(per_event, index):
    """per_event[event][index[event][i][j]] for a doubly jagged index."""
    n = ak.to_numpy(ak.num(per_event, axis=1))
    starts = np.concatenate([[0], np.cumsum(n)[:-1]])
    flat = ak.flatten(per_event)[ak.flatten(index + starts, axis=None)]
    inner = ak.unflatten(flat, ak.flatten(ak.num(index, axis=2)))
    return ak.unflatten(inner, ak.num(index, axis=1))


def clopper_pearson(k, n, level=0.683):
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    alpha = 1 - level
    lo = np.where(k > 0, beta.ppf(alpha / 2, k, n - k + 1), 0.0)
    hi = np.where(k < n, beta.ppf(1 - alpha / 2, k + 1, n - k), 1.0)
    return lo, hi


class Counts:
    """Numerator and denominator histograms vs pt and eta, plus the integrated counts."""

    def __init__(self):
        self.num = {"pt": np.zeros(len(PT_BINS) - 1), "eta": np.zeros(len(ETA_BINS) - 1)}
        self.den = {"pt": np.zeros(len(PT_BINS) - 1), "eta": np.zeros(len(ETA_BINS) - 1)}
        self.total = np.zeros(2)  # numerator, denominator with both the pt and the eta cut

    def fill(self, pt, eta, passed, pt_cut, eta_cut):
        pt, eta, passed = (ak.to_numpy(ak.flatten(x, axis=None)) for x in (pt, eta, passed))
        in_eta, in_pt = np.abs(eta) < eta_cut, pt > pt_cut
        self.den["pt"] += np.histogram(pt[in_eta], PT_BINS)[0]
        self.num["pt"] += np.histogram(pt[in_eta & passed], PT_BINS)[0]
        self.den["eta"] += np.histogram(eta[in_pt], ETA_BINS)[0]
        self.num["eta"] += np.histogram(eta[in_pt & passed], ETA_BINS)[0]
        self.total += [np.sum(in_eta & in_pt & passed), np.sum(in_eta & in_pt)]


def process(arrays, results, n_objects, args):
    n_sim = ak.num(arrays["sim_pt"])
    sim_selected = (
        (arrays["sim_q"] != 0)
        & (abs(arrays["sim_vz"]) < 30)
        & (np.hypot(arrays["sim_vx"], arrays["sim_vy"]) < 2.5)
    )
    sim_pt, sim_eta = arrays["sim_pt"][sim_selected], arrays["sim_eta"][sim_selected]

    # per object type: which objects are used, and per sim track the number of matched used objects
    used, n_matched = {}, {}
    for obj in ("oc", "t5", "t4"):
        used[obj] = ak.ones_like(arrays[f"{obj}_isFake"]) == 1
        if obj == "oc" and args.oc_min_layers > 0:
            used[obj] = arrays["oc_nLayers"] >= args.oc_min_layers
        matched = (arrays[f"sim_{obj}IdxAllFrac"] > MATCH_FRAC) & take(used[obj], arrays[f"sim_{obj}IdxAll"])
        n_matched[obj] = ak.sum(matched, axis=2)

    for name, _, objs, _, _ in SERIES:
        n_matched_series = sum(n_matched[obj] for obj in objs)
        results[name]["efficiency"].fill(sim_pt, sim_eta, (n_matched_series > 0)[sim_selected], args.pt_cut, args.eta_cut)
        for obj in objs:
            pt, eta = arrays[f"{obj}_pt"][used[obj]], arrays[f"{obj}_eta"][used[obj]]
            results[name]["fake"].fill(pt, eta, (arrays[f"{obj}_isFake"] == 1)[used[obj]], args.pt_cut, args.eta_cut)
            # duplicate: a matched hard-scatter sim track with more than one matched object of this series
            sim_idx = arrays[f"{obj}_simIdxAll"]
            sim_idx = sim_idx[sim_idx < n_sim]
            duplicate = ak.any(take(n_matched_series, sim_idx) > 1, axis=2)
            results[name]["duplicate"].fill(pt, eta, duplicate[used[obj]], args.pt_cut, args.eta_cut)
            n_objects[name] += ak.sum(used[obj])
            if len(objs) == 1 and not (obj == "oc" and args.oc_min_layers > 0):
                # cross-check with the flag of the ntuple
                if not ak.all(duplicate == (arrays[f"{obj}_isDuplicate"] == 1)):
                    raise RuntimeError(f"{obj}_isDuplicate differs from the recomputed duplicate flag")


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=10)


def plot_metric(results, metric, title, subtitle, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    for ax, (var, bins, xlabel) in zip(axes, [("pt", PT_BINS, "pT [GeV]"), ("eta", ETA_BINS, "eta")]):
        style_axes(ax)
        centres = np.sqrt(bins[:-1] * bins[1:]) if var == "pt" else 0.5 * (bins[:-1] + bins[1:])
        top = 0.0
        for name, label, objs, colour, marker in SERIES:
            counts = results[name][metric]
            num, den = counts.num[var], counts.den[var]
            ok = den > 0
            rate = num[ok] / den[ok]
            lo, hi = clopper_pearson(num[ok], den[ok])
            ax.errorbar(
                centres[ok], rate, yerr=[rate - lo, hi - rate], color=colour, marker=marker, markersize=6.5,
                markeredgecolor=SURFACE, markeredgewidth=1.2, linewidth=2, elinewidth=1.2, label=label,
                zorder=2 if len(objs) > 1 else 3,  # the sum goes under its parts, which it often overlaps
            )
            top = max(top, np.max(hi, initial=0.0))
        if var == "pt":
            ax.set_xscale("log")
            ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50], labels=["0.5", "1", "2", "5", "10", "20", "50"])
            ax.minorticks_off()
        who = "sim track" if metric == "efficiency" else "object"
        ax.set_xlabel(f"{who} {xlabel}", color=INK_SECONDARY, fontsize=11)
        ax.set_ylim(0, 1.05 if metric == "efficiency" else min(1.05, 1.15 * top + 0.01))
    axes[0].set_ylabel(title, color=INK_SECONDARY, fontsize=11)
    handles, labels = axes[1].get_legend_handles_labels()
    legend = fig.legend(handles, labels, frameon=False, fontsize=10, ncol=len(SERIES), loc="upper right",
                        bbox_to_anchor=(0.985, 0.995), handletextpad=0.5, columnspacing=1.6)
    for text in legend.get_texts():
        text.set_color(INK)
    fig.suptitle(title, x=0.06, y=0.985, ha="left", color=INK, fontsize=14, fontweight="bold")
    fig.text(0.06, 0.905, subtitle, ha="left", color=INK_SECONDARY, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, nargs="+", help="LST ntuple(s) or wildcard(s)")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--tree", default="tree")
    parser.add_argument("--pt-cut", type=float, default=0.9)
    parser.add_argument("--eta-cut", type=float, default=2.4)
    parser.add_argument("--oc-min-layers", type=int, default=0, help="use only clusters with at least this many layers")
    parser.add_argument("--n-events", type=int, default=-1)
    args = parser.parse_args()

    files = sorted(f for pattern in args.input for f in glob.glob(pattern))
    if not files:
        sys.exit(f"no input files match {args.input}")
    os.makedirs(args.output, exist_ok=True)

    branches = ["sim_pt", "sim_eta", "sim_q", "sim_vx", "sim_vy", "sim_vz", "oc_nLayers"]
    for obj in ("oc", "t5", "t4"):
        branches += [f"{obj}_pt", f"{obj}_eta", f"{obj}_isFake", f"{obj}_isDuplicate", f"{obj}_simIdxAll"]
        branches += [f"sim_{obj}IdxAll", f"sim_{obj}IdxAllFrac"]

    results = {name: {metric: Counts() for metric, _ in METRICS} for name, *_ in SERIES}
    n_objects = {name: 0 for name, *_ in SERIES}
    n_events = 0
    for arrays in uproot.iterate([f"{f}:{args.tree}" for f in files], branches, step_size=20):
        if 0 <= args.n_events < n_events + len(arrays):
            arrays = arrays[: args.n_events - n_events]
        if len(arrays) == 0:
            break
        process(arrays, results, n_objects, args)
        n_events += len(arrays)

    # the efficiency denominator must be the same for every series
    reference = results[SERIES[0][0]]["efficiency"]
    for name, *_ in SERIES:
        eff = results[name]["efficiency"]
        assert all(np.array_equal(eff.den[v], reference.den[v]) for v in ("pt", "eta")) and eff.total[1] == reference.total[1]

    oc_cut = f", clusters with >= {args.oc_min_layers} layers" if args.oc_min_layers > 0 else ""
    subtitle = (
        f"PU200 ttbar, {n_events} events; pT plots: |eta| < {args.eta_cut:g}; eta plots: pT > {args.pt_cut:g} GeV"
        f"{oc_cut}; bars: 68% Clopper-Pearson"
    )
    for metric, title in METRICS:
        plot_metric(results, metric, title, subtitle, os.path.join(args.output, f"{metric}.png"))

    summary = {"events": n_events, "pt_cut": args.pt_cut, "eta_cut": args.eta_cut, "oc_min_layers": args.oc_min_layers,
               "sim_tracks_selected": int(reference.total[1]), "series": {}}
    lines = [
        f"{n_events} events; selected sim tracks (pt > {args.pt_cut:g}, |eta| < {args.eta_cut:g}): {int(reference.total[1])}"
        f" ({reference.total[1] / n_events:.1f} per event), the same for every row{oc_cut}",
        "",
        "| objects | per event (all) | per event (pt, eta cuts) | efficiency | fake rate | duplicate rate |",
        "|---|---|---|---|---|---|",
    ]
    for name, label, *_ in SERIES:
        row = {"objects_per_event": n_objects[name] / n_events}
        cells = []
        for metric, _ in METRICS:
            k, n = results[name][metric].total
            lo, hi = clopper_pearson(k, n)
            row[metric] = {"num": int(k), "den": int(n), "rate": k / n if n else None, "lo": float(lo), "hi": float(hi)}
            cells.append(f"{100 * k / n:.1f}% ({int(k)}/{int(n)})" if n else "n/a")
            for var, bins in (("pt", PT_BINS), ("eta", ETA_BINS)):
                row[metric][var] = {"bins": bins.tolist(), "num": results[name][metric].num[var].tolist(),
                                    "den": results[name][metric].den[var].tolist()}
        summary["series"][name] = row
        selected_per_event = results[name]["fake"].total[1] / n_events
        lines.append(f"| {label} | {row['objects_per_event']:.0f} | {selected_per_event:.0f} | " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.output, "summary.md"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nplots and tables in {args.output}/")


if __name__ == "__main__":
    main()
