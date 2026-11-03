"""Transformer clusters (oc) vs LST T5 / T4 / T5+T4, from one LST standalone ntuple.

Input: ntuple(s) written by `lst --t5 --t4 --transformer_model <pt>` (oc_*, t5_*, t4_*, sim_* branches).
All object types use the LST hit matching of the ntuple (a sim track matches an object if it has a sim hit in
more than 75% of the object's hits), and one sim track selection for the efficiency denominator.

  efficiency      selected sim tracks with at least one matched object / selected sim tracks
  fake rate       objects without a matched sim track / objects
  duplicate rate  objects with a matched (hard-scatter) sim track that is matched by more than one object / objects

Sim tracks in the ntuple are the hard-scatter, in-time ones. Selection (as in analysis/main.py), for all plots:
q != 0, pt > --pt-cut, |eta| < --eta-cut, |vz| < --vz-cut (30 cm), vxy < --vxy-cut (2.5 cm).
Objects (as in analysis/main.py), for all plots: --pt-cut < pt < 100 GeV, no eta cut (|eta| < --obj-eta-cut if given).

T4 and T5 + T4 are only included with --t4.

Plots: <metric>.png (pT and eta side by side), and cms_<metric>_<pt|eta>.png in the style of compare_oc_t5_t4_md.py, with its pT and eta bins
(read from --config, as in compare_oc_t5_t4_md.py).

Efficiency vs vertex displacement (efficiency_vxy.png, cms_efficiency_vxy.png), as in LST's efficiency code
(performance.cc, "N minus dxy cut"): the sim selection above without the vxy cut; vxy = sqrt(vx^2 + vy^2) of the sim
vertex, bins from --vxy-max and --vxy-bins (default 0-30 cm in 2 cm bins, the LST range with its coarse binning),
overflow in the last bin.

    python3 analysis/compare_oc_t5_t4.py --input 'lst_oc*.root' --output output/comparison/plots [--t4]
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
import mplhep as hep
import numpy as np
import uproot
import yaml
from scipy.stats import beta

MATCH_FRAC = 0.75
PT_BINS = np.array([0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0])
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


def clopper_pearson(k, n, level=0.95):  # as main.py (proportion_confint default)
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    alpha = 1 - level
    lo = np.where(k > 0, beta.ppf(alpha / 2, k, n - k + 1), 0.0)
    hi = np.where(k < n, beta.ppf(1 - alpha / 2, k + 1, n - k), 1.0)
    return lo, hi


class Counts:
    """Numerator and denominator histograms vs pt and eta, plus the integrated counts."""

    def __init__(self, pt_bins=PT_BINS, eta_bins=ETA_BINS):
        self.pt_bins, self.eta_bins = pt_bins, eta_bins
        self.num = {"pt": np.zeros(len(pt_bins) - 1), "eta": np.zeros(len(eta_bins) - 1)}
        self.den = {"pt": np.zeros(len(pt_bins) - 1), "eta": np.zeros(len(eta_bins) - 1)}
        self.total = np.zeros(2)  # numerator, denominator with both the pt and the eta cut

    def fill(self, pt, eta, passed):
        """pt, eta, passed of the selected sim tracks or objects."""
        pt, eta, passed = (ak.to_numpy(ak.flatten(x, axis=None)) for x in (pt, eta, passed))
        self.den["pt"] += np.histogram(pt, self.pt_bins)[0]
        self.num["pt"] += np.histogram(pt[passed], self.pt_bins)[0]
        self.den["eta"] += np.histogram(eta, self.eta_bins)[0]
        self.num["eta"] += np.histogram(eta[passed], self.eta_bins)[0]
        self.total += [np.sum(passed), len(passed)]


class VxyCounts:
    """Efficiency numerator and denominator vs sim vertex vxy; overflow goes into the last bin (as in LST)."""

    def __init__(self, bins):
        self.bins = bins
        self.num, self.den = np.zeros(len(bins) - 1), np.zeros(len(bins) - 1)

    def fill(self, vxy, passed):
        vxy, passed = (ak.to_numpy(ak.flatten(x, axis=None)) for x in (vxy, passed))
        vxy = np.minimum(vxy, 0.5 * (self.bins[-2] + self.bins[-1]))
        self.den += np.histogram(vxy, self.bins)[0]
        self.num += np.histogram(vxy[passed], self.bins)[0]


def process(arrays, all_results, n_objects, args, vxy_results=None):
    """Fills every results dict of all_results (the same counts with different bins)."""
    n_sim = ak.num(arrays["sim_pt"])
    sim_selected = (
        (arrays["sim_q"] != 0)
        & (arrays["sim_pt"] > args.pt_cut)
        & (abs(arrays["sim_eta"]) < args.eta_cut)
        & (abs(arrays["sim_vz"]) < args.vz_cut)
        & (np.hypot(arrays["sim_vx"], arrays["sim_vy"]) < args.vxy_cut)
    )
    sim_pt, sim_eta = arrays["sim_pt"][sim_selected], arrays["sim_eta"][sim_selected]
    # efficiency vs vxy: every cut except the one on vxy
    sim_vxy = np.hypot(arrays["sim_vx"], arrays["sim_vy"])
    sim_selected_vxy = (
        (arrays["sim_q"] != 0)
        & (arrays["sim_pt"] > args.pt_cut)
        & (abs(arrays["sim_eta"]) < args.eta_cut)
        & (abs(arrays["sim_vz"]) < args.vz_cut)
    )

    # per object type: which objects are used, and per sim track the number of matched used objects
    used, n_matched = {}, {}
    for obj in args.objects:
        used[obj] = ak.ones_like(arrays[f"{obj}_isFake"]) == 1
        if obj == "oc" and args.oc_min_layers > 0:
            used[obj] = arrays["oc_nLayers"] >= args.oc_min_layers
        matched = (arrays[f"sim_{obj}IdxAllFrac"] > MATCH_FRAC) & take(used[obj], arrays[f"sim_{obj}IdxAll"])
        n_matched[obj] = ak.sum(matched, axis=2)

    for name, _, objs, _, _ in args.series:
        n_matched_series = sum(n_matched[obj] for obj in objs)
        for results in all_results:
            results[name]["efficiency"].fill(sim_pt, sim_eta, (n_matched_series > 0)[sim_selected])
        if vxy_results is not None:
            vxy_results[name].fill(sim_vxy[sim_selected_vxy], (n_matched_series > 0)[sim_selected_vxy])
        for obj in objs:
            # the pt window selects the objects of the fake and duplicate rates only, not the matched ones
            plotted = used[obj] & (arrays[f"{obj}_pt"] > args.pt_cut) & (arrays[f"{obj}_pt"] < 100)
            if args.obj_eta_cut is not None:
                plotted = plotted & (abs(arrays[f"{obj}_eta"]) < args.obj_eta_cut)
            pt, eta = arrays[f"{obj}_pt"][plotted], arrays[f"{obj}_eta"][plotted]
            for results in all_results:
                results[name]["fake"].fill(pt, eta, (arrays[f"{obj}_isFake"] == 1)[plotted])
            # duplicate: a matched hard-scatter sim track with more than one matched object of this series
            sim_idx = arrays[f"{obj}_simIdxAll"]
            sim_idx = sim_idx[sim_idx < n_sim]
            duplicate = ak.any(take(n_matched_series, sim_idx) > 1, axis=2)
            for results in all_results:
                results[name]["duplicate"].fill(pt, eta, duplicate[plotted])
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


def plot_metric(results, series, metric, title, subtitle, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    for ax, (var, bins, xlabel) in zip(axes, [("pt", PT_BINS, "pT [GeV]"), ("eta", ETA_BINS, "eta")]):
        style_axes(ax)
        centres = np.sqrt(bins[:-1] * bins[1:]) if var == "pt" else 0.5 * (bins[:-1] + bins[1:])
        top = 0.0
        for name, label, objs, colour, marker in series:
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
            ax.set_xticks([0.8, 1, 2, 5, 10, 20, 50], labels=["0.8", "1", "2", "5", "10", "20", "50"])
            ax.minorticks_off()
        who = "sim track" if metric == "efficiency" else "object"
        ax.set_xlabel(f"{who} {xlabel}", color=INK_SECONDARY, fontsize=11)
        ax.set_ylim(0, 1.05 if metric == "efficiency" else min(1.05, 1.15 * top + 0.01))
    axes[0].set_ylabel(title, color=INK_SECONDARY, fontsize=11)
    handles, labels = axes[1].get_legend_handles_labels()
    legend = fig.legend(handles, labels, frameon=False, fontsize=10, ncol=len(series), loc="upper right",
                        bbox_to_anchor=(0.985, 0.995), handletextpad=0.5, columnspacing=1.6)
    for text in legend.get_texts():
        text.set_color(INK)
    fig.suptitle(title, x=0.06, y=0.985, ha="left", color=INK, fontsize=14, fontweight="bold")
    fig.text(0.06, 0.905, subtitle, ha="left", color=INK_SECONDARY, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# labels and colours of compare_oc_t5_t4_md.py for the cms_* plots
CMS_SERIES = {"oc": ("Transformer OC", "blue"), "t5": ("LST T5s", "red"), "t4": ("LST T4s", "green"),
              "t5t4": ("LST T5s + T4s", "orange")}


def cms_sample_text(sample):
    return r"$t\bar{t}$, $\langle\mu\rangle=200$" if sample == "PU200 ttbar" else sample


def plot_efficiency_vxy(vxy_results, series, subtitle, path):
    """Efficiency vs sim vertex vxy, in the style of plot_metric."""
    fig, ax = plt.subplots(figsize=(7, 4.6), facecolor=SURFACE)
    style_axes(ax)
    for name, label, objs, colour, marker in series:
        counts = vxy_results[name]
        centres = 0.5 * (counts.bins[:-1] + counts.bins[1:])
        ok = counts.den > 0
        rate = counts.num[ok] / counts.den[ok]
        lo, hi = clopper_pearson(counts.num[ok], counts.den[ok])
        ax.errorbar(centres[ok], rate, yerr=[rate - lo, hi - rate], color=colour, marker=marker, markersize=6.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, linewidth=2, elinewidth=1.2, label=label,
                    zorder=2 if len(objs) > 1 else 3)
    ax.set_xlabel("sim track vertex vxy [cm] (last bin: overflow)", color=INK_SECONDARY, fontsize=11)
    ax.set_ylabel("Efficiency", color=INK_SECONDARY, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(vxy_results[series[0][0]].bins[0], vxy_results[series[0][0]].bins[-1])
    legend = ax.legend(frameon=False, fontsize=10, loc="upper right")
    for text in legend.get_texts():
        text.set_color(INK)
    fig.suptitle("Efficiency vs vertex displacement", x=0.08, y=0.985, ha="left", color=INK, fontsize=14,
                 fontweight="bold")
    fig.text(0.08, 0.895, subtitle, ha="left", color=INK_SECONDARY, fontsize=8, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_efficiency_vxy_cms(vxy_results, series, sample, output):
    with plt.style.context(hep.style.CMS):
        fig, ax = plt.subplots(figsize=(12, 8))
        for name, _, _, _, marker in series:
            label, colour = CMS_SERIES[name]
            counts = vxy_results[name]
            centres = 0.5 * (counts.bins[:-1] + counts.bins[1:])
            ok = counts.den > 0
            rate = counts.num[ok] / counts.den[ok]
            lo, hi = clopper_pearson(counts.num[ok], counts.den[ok])
            ax.errorbar(centres[ok], rate, yerr=[rate - lo, hi - rate], fmt=marker, color=colour, label=label,
                        capsize=3, markersize=5)
        ax.set_xlabel(r"Simulated track $r_{vertex}$ [cm]")
        ax.set_ylabel("Efficiency")
        ax.set_ylim(0, 1.19)
        hep.cms.label("Preliminary", data=False, rlabel=r"Phase-2 (14 TeV)", ax=ax)
        ax.text(0.95, 0.95, cms_sample_text(sample), transform=ax.transAxes, fontsize=20,
                verticalalignment="top", horizontalalignment="right")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output, "cms_efficiency_vxy.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_metric_cms(results, series, metric, title, output, sample="PU200 ttbar"):
    """One plot per variable, in the style of compare_oc_t5_t4_md.py; empty bins are not drawn."""
    counts = results[series[0][0]][metric]
    centers = {"pt": (counts.pt_bins[:-1] + counts.pt_bins[1:]) / 2, "eta": (counts.eta_bins[:-1] + counts.eta_bins[1:]) / 2}
    xlabels = {
        ("efficiency", "pt"): "Simulated track $p_T$ [GeV]", ("efficiency", "eta"): r"Simulated track $\eta$",
        ("fake", "pt"): "Reconstructed track $p_T$ [GeV]", ("fake", "eta"): r"Reconstructed track $\eta$",
        ("duplicate", "pt"): "Reconstructed track $p_T$ [GeV]", ("duplicate", "eta"): r"Reconstructed track $\eta$",
    }
    ylims = {"efficiency": (0, 1.19), "fake": (0, 0.3), "duplicate": (0, 0.2)}
    with plt.style.context(hep.style.CMS):  # not hep.style.use, which would also restyle the other plots
        for var in ("pt", "eta"):
            fig, ax = plt.subplots(figsize=(12, 8))
            top = 0.0
            for name, _, _, _, marker in series:
                label, colour = CMS_SERIES[name]
                counts = results[name][metric]
                num, den = counts.num[var], counts.den[var]
                ok = den > 0
                rate = num[ok] / den[ok]
                lo, hi = clopper_pearson(num[ok], den[ok])
                ax.errorbar(centers[var][ok], rate, yerr=[rate - lo, hi - rate], fmt=marker, color=colour, label=label,
                            capsize=3, markersize=5)
                top = max(top, np.max(rate, initial=0.0))
            ax.set_xlabel(xlabels[(metric, var)])
            ax.set_ylabel(title)
            low, high = ylims[metric]
            ax.set_ylim(low, high if metric == "efficiency" else max(high, min(1.19, 1.4 * top)))
            hep.cms.label("Preliminary", data=False, rlabel=r"Phase-2 (14 TeV)", ax=ax)
            ax.text(0.95, 0.95, cms_sample_text(sample), transform=ax.transAxes, fontsize=20,
                    verticalalignment="top", horizontalalignment="right")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(output, f"cms_{metric}_{var}.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, nargs="+", help="LST ntuple(s) or wildcard(s)")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--config", default="./config_pu200.yaml", help="config with the bins of the cms_* plots")
    parser.add_argument("--tree", default="tree")
    parser.add_argument("--pt-cut", type=float, default=0.8)
    parser.add_argument("--eta-cut", type=float, default=2.4)
    parser.add_argument("--obj-eta-cut", type=float, default=None,
                        help="objects of the fake and duplicate rates: |eta| below this (default: no cut)")
    parser.add_argument("--vz-cut", type=float, default=30, help="sim tracks: |vz| below this [cm]")
    parser.add_argument("--vxy-cut", type=float, default=2.5, help="sim tracks: vxy below this [cm]")
    parser.add_argument("--sample", default="PU200 ttbar", help="sample name in the plot titles")
    parser.add_argument("--vxy-max", type=float, default=30, help="upper edge of the efficiency-vs-vxy plot [cm]")
    parser.add_argument("--vxy-bins", type=int, default=15, help="number of bins of the efficiency-vs-vxy plot")
    parser.add_argument("--oc-min-layers", type=int, default=0, help="use only clusters with at least this many layers")
    parser.add_argument("--n-events", type=int, default=-1)
    parser.add_argument("--t4", action="store_true", help="also plot T4 and T5 + T4")
    args = parser.parse_args()
    args.objects = ["oc", "t5"] + (["t4"] if args.t4 else [])
    args.series = [s for s in SERIES if all(obj in args.objects for obj in s[2])]

    with open(args.config) as f:
        config = yaml.safe_load(f)
    # bins of the cms_* plots, as in compare_oc_t5_t4_md.py
    cms_pt_bins = np.linspace(config.get("pt_bins_start", 0.5), config.get("pt_bins_end", 2.0), config.get("pt_bins_count", 50))
    cms_eta_bins = np.linspace(config.get("eta_bins_start", -4), config.get("eta_bins_end", 4), config.get("eta_bins_count", 50))

    files = sorted(f for pattern in args.input for f in glob.glob(pattern))
    if not files:
        sys.exit(f"no input files match {args.input}")
    os.makedirs(args.output, exist_ok=True)

    branches = ["sim_pt", "sim_eta", "sim_q", "sim_vx", "sim_vy", "sim_vz", "oc_nLayers"]
    for obj in args.objects:
        branches += [f"{obj}_pt", f"{obj}_eta", f"{obj}_isFake", f"{obj}_isDuplicate", f"{obj}_simIdxAll"]
        branches += [f"sim_{obj}IdxAll", f"sim_{obj}IdxAllFrac"]

    results = {name: {metric: Counts() for metric, _ in METRICS} for name, *_ in args.series}
    cms_results = {name: {metric: Counts(cms_pt_bins, cms_eta_bins) for metric, _ in METRICS} for name, *_ in args.series}
    vxy_bins = np.linspace(0, args.vxy_max, args.vxy_bins + 1)
    vxy_results = {name: VxyCounts(vxy_bins) for name, *_ in args.series}
    n_objects = {name: 0 for name, *_ in args.series}
    n_events = 0
    for arrays in uproot.iterate([f"{f}:{args.tree}" for f in files], branches, step_size=20):
        if 0 <= args.n_events < n_events + len(arrays):
            arrays = arrays[: args.n_events - n_events]
        if len(arrays) == 0:
            break
        process(arrays, (results, cms_results), n_objects, args, vxy_results)
        n_events += len(arrays)

    # the efficiency denominator must be the same for every series
    reference = results[args.series[0][0]]["efficiency"]
    for name, *_ in args.series:
        eff = results[name]["efficiency"]
        assert all(np.array_equal(eff.den[v], reference.den[v]) for v in ("pt", "eta")) and eff.total[1] == reference.total[1]
        assert np.array_equal(vxy_results[name].den, vxy_results[args.series[0][0]].den)

    oc_cut = f", clusters with >= {args.oc_min_layers} layers" if args.oc_min_layers > 0 else ""
    vz_text = f"|vz| < {args.vz_cut:g} cm" if args.vz_cut < 1e6 else "no vz cut"
    obj_eta_text = f", |eta| < {args.obj_eta_cut:g}" if args.obj_eta_cut is not None else ""
    vxy_text = f"vxy < {args.vxy_cut:g} cm" if args.vxy_cut < 1e6 else "no vxy cut"
    subtitle = (
        f"{args.sample}, {n_events} events; sim tracks: pT > {args.pt_cut:g} GeV, |eta| < {args.eta_cut:g}, {vz_text},"
        f" {vxy_text}; objects: {args.pt_cut:g} < pT < 100 GeV{obj_eta_text}{oc_cut}; bars: 95% Clopper-Pearson"
    )
    for metric, title in METRICS:
        plot_metric(results, args.series, metric, title, subtitle, os.path.join(args.output, f"{metric}.png"))
        plot_metric_cms(cms_results, args.series, metric, title, args.output, args.sample)
    vxy_subtitle = (
        f"{args.sample}, {n_events} events; sim tracks: pT > {args.pt_cut:g} GeV, |eta| < {args.eta_cut:g},"
        f" {vz_text}, no vxy cut{oc_cut}; bars: 95% Clopper-Pearson"
    )
    plot_efficiency_vxy(vxy_results, args.series, vxy_subtitle, os.path.join(args.output, "efficiency_vxy.png"))
    plot_efficiency_vxy_cms(vxy_results, args.series, args.sample, args.output)

    summary = {"events": n_events, "pt_cut": args.pt_cut, "eta_cut": args.eta_cut, "obj_eta_cut": args.obj_eta_cut, "oc_min_layers": args.oc_min_layers,
               "sim_tracks_selected": int(reference.total[1]), "series": {}}
    lines = [
        f"{n_events} events; selected sim tracks (pt > {args.pt_cut:g}, |eta| < {args.eta_cut:g}): {int(reference.total[1])}"
        f" ({reference.total[1] / n_events:.1f} per event), the same for every row{oc_cut}"
        + (f"; objects with |eta| < {args.obj_eta_cut:g}" if args.obj_eta_cut is not None else ""),
        "",
        "| objects | per event (all) | per event (pt cut) | efficiency | fake rate | duplicate rate |",
        "|---|---|---|---|---|---|",
    ]
    for name, label, *_ in args.series:
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
        row["efficiency"]["vxy"] = {"bins": vxy_bins.tolist(), "num": vxy_results[name].num.tolist(),
                                    "den": vxy_results[name].den.tolist(), "overflow_in_last_bin": True}
        summary["series"][name] = row
        selected_per_event = results[name]["fake"].total[1] / n_events
        per_event = [f"{x:.0f}" if x >= 10 else f"{x:.2f}" for x in (row["objects_per_event"], selected_per_event)]
        lines.append(f"| {label} | {per_event[0]} | {per_event[1]} | " + " | ".join(cells) + " |")
    # efficiency vs vxy (sim selection without the vxy cut)
    lines += ["", f"Efficiency vs sim vertex vxy (no vxy cut, {vz_text}; last bin includes overflow):", "",
              "| vxy [cm] | sim tracks | " + " | ".join(label for _, label, *_ in args.series) + " |",
              "|---|---|" + "---|" * len(args.series)]
    for i in range(len(vxy_bins) - 1):
        den = vxy_results[args.series[0][0]].den[i]
        edge = f"{vxy_bins[i]:g}-{vxy_bins[i + 1]:g}" + ("+" if i == len(vxy_bins) - 2 else "")
        cells = [f"{100 * vxy_results[name].num[i] / den:.1f}%" if den else "n/a" for name, *_ in args.series]
        lines.append(f"| {edge} | {int(den)} | " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.output, "summary.md"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nplots and tables in {args.output}/")


if __name__ == "__main__":
    main()
