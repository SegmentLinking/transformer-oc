"""Transformer cluster efficiency: from the MD matching to the LST hit matching, one change at a time.

Input: an LST ntuple with the oc_*, sim_ocIdxAll(Frac) and MD/LS/T3 branches
(`lst --md --ls --t3 --t5 --transformer_model <pt>`).

The same clusters (>= --oc-min-layers distinct MD layers, as in main.py) and sim track selection (q != 0,
pT > 0.8 GeV, |eta| < 2.4, |vz| < 30 cm, vxy < 2.5 cm) in every variant:

  matching    MD    majority sim track of the cluster's MD slots (4 per T3), fraction >= purity_threshold
                    (compare_oc_t5_t4_md.py, as in main.py)
              hit   LST hit matching of the ntuple: sim_ocIdxAllFrac > 0.75 (compare_oc_t5_t4.py)
  denominator T3    selected sim tracks that are the truth label (t3_simIdx) of at least one T3 (main.py)
              all   all selected sim tracks (as for the LST T5s and T4s)

    python3 analysis/compare_oc_matching_steps.py --input lst.root --output plots/matching_steps
"""
import argparse
import glob
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_oc_t5_t4_md import MAX_T3_PT, flat, majority_match, n_distinct, offsets, to_global

hep.style.use(hep.style.CMS)

PT_BINS = np.array([0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0])
ETA_BINS = np.linspace(-2.4, 2.4, 25)

# name, label, matching, denominator, colour, marker
VARIANTS = [
    ("md_t3", "MD matching, T3 denominator (old)", "md", "t3", "black", "o"),
    ("hit_t3", "Hit matching, T3 denominator", "hit", "t3", "tab:blue", "s"),
    ("md_all", "MD matching, all sim tracks", "md", "all", "tab:orange", "^"),
    ("hit_all", "Hit matching, all sim tracks (new)", "hit", "all", "tab:red", "D"),
]


class Counts:
    def __init__(self):
        self.num = {"pt": np.zeros(len(PT_BINS) - 1), "eta": np.zeros(len(ETA_BINS) - 1)}
        self.den = {"pt": np.zeros(len(PT_BINS) - 1), "eta": np.zeros(len(ETA_BINS) - 1)}
        self.total = np.zeros(2)

    def fill(self, pt, eta, passed):
        pt = np.clip(pt, None, PT_BINS[-1] - 1e-6)  # overflow into the last bin
        for var, x, bins in (("pt", pt, PT_BINS), ("eta", eta, ETA_BINS)):
            self.den[var] += np.histogram(x, bins)[0]
            self.num[var] += np.histogram(x[passed], bins)[0]
        self.total += [np.sum(passed), len(passed)]


def process(arrays, results, totals, args):
    n_sim = ak.to_numpy(ak.num(arrays["sim_pt"]))
    sim_offsets = offsets(arrays["sim_pt"])
    n_sim_total = int(np.sum(n_sim))

    # MD matching of the clusters, as in compare_oc_t5_t4_md.py
    ls_md = [to_global(arrays[f"ls_mdIdx{m}"], arrays["md_simIdx"]) for m in (0, 1)]
    t3_ls = [to_global(arrays[f"t3_lsIdx{l}"], arrays["ls_mdIdx0"]) for l in (0, 1)]
    t3_mds = np.stack([ls_md[m][t3_ls[l]] for l in (0, 1) for m in (0, 1)], axis=1)
    md_sim, md_layer = flat(arrays["md_simIdx"]).astype(np.int64), flat(arrays["md_layer"]).astype(np.int64)

    n_oc = ak.to_numpy(ak.num(arrays["oc_t3Idx"], axis=1))
    t3s_per_oc = flat(ak.num(arrays["oc_t3Idx"], axis=2))
    oc_of_t3 = np.repeat(np.arange(len(t3s_per_oc)), t3s_per_oc)
    t3 = flat(arrays["oc_t3Idx"]).astype(np.int64) + np.repeat(np.repeat(offsets(arrays["t3_simIdx"]), n_oc), t3s_per_oc)
    oc_of_md = np.repeat(oc_of_t3, 4)
    mds = t3_mds[t3].ravel()
    sim, _ = majority_match(oc_of_md, md_sim[mds], 4 * t3s_per_oc, args.purity_threshold)
    keep = n_distinct(oc_of_md, md_layer[mds], len(sim)) >= args.oc_min_layers
    evt = np.repeat(np.arange(len(n_oc)), n_oc)
    sim = np.where((sim >= 0) & (sim < n_sim[evt]) & keep, sim + sim_offsets[evt], -1)
    n_matched_md = np.bincount(sim[sim >= 0], minlength=n_sim_total)
    totals["clusters"] += int(np.sum(keep))

    # LST hit matching of the ntuple (only fractions > 0.75 are stored), kept clusters only
    idx, frac = arrays["sim_ocIdxAll"], arrays["sim_ocIdxAllFrac"]
    per_sim = flat(ak.num(idx, axis=2))
    per_evt = ak.to_numpy(ak.sum(ak.num(idx, axis=2), axis=1))
    oc_global = flat(idx).astype(np.int64) + np.repeat(offsets(arrays["oc_t3Idx"]), per_evt)
    sim_of_entry = np.repeat(np.arange(n_sim_total), per_sim)
    good = (flat(frac) > 0.75) & keep[oc_global]
    n_matched_hit = np.bincount(sim_of_entry[good], minlength=n_sim_total)

    sim_pt, sim_eta, sim_q, sim_vx, sim_vy, sim_vz = (flat(arrays[f"sim_{v}"]) for v in ("pt", "eta", "q", "vx", "vy", "vz"))
    selected = ((sim_q != 0) & (sim_pt > 0.8) & (np.abs(sim_eta) < 2.4) & (np.abs(sim_vz) < 30)
                & (np.hypot(sim_vx, sim_vy) < 2.5))
    t3_sim = arrays["t3_simIdx"][(arrays["t3_simIdx"] >= 0) & (arrays["t3_pt"] < MAX_T3_PT)]
    t3_sim = t3_sim[t3_sim < n_sim]
    has_t3 = np.zeros(n_sim_total, dtype=bool)
    has_t3[to_global(t3_sim, arrays["sim_pt"])] = True

    matched = {"md": n_matched_md > 0, "hit": n_matched_hit > 0}
    denominators = {"t3": selected & has_t3, "all": selected}
    for name, _, matching, den, _, _ in VARIANTS:
        s = denominators[den]
        results[name].fill(sim_pt[s], sim_eta[s], matched[matching][s])

    s = selected & has_t3
    totals["sims"] += int(np.sum(selected))
    totals["sims_t3"] += int(np.sum(s))
    totals["md_only"] += int(np.sum(s & matched["md"] & ~matched["hit"]))
    totals["hit_only"] += int(np.sum(s & ~matched["md"] & matched["hit"]))
    totals["both"] += int(np.sum(s & matched["md"] & matched["hit"]))
    totals["no_t3_matched"] += int(np.sum(selected & ~has_t3 & matched["hit"]))
    totals["no_t3_matched_md"] += int(np.sum(selected & ~has_t3 & matched["md"]))


def errors(k, n):
    valid = n > 0
    k, n1 = np.where(valid, k, 0), np.where(valid, n, 1)
    eff = k / n1
    # 95% Clopper-Pearson, as main.py and compare_oc_t5_t4.py
    lo = np.where(k > 0, beta.ppf(0.025, k, n1 - k + 1), 0.0)
    hi = np.where(k < n1, beta.ppf(0.975, k + 1, n1 - k), 1.0)
    return np.where(valid, eff, np.nan), [np.where(valid, eff - lo, 0), np.where(valid, hi - eff, 0)]


def plot(results, output, n_events):
    for var, bins, xlabel in (("pt", PT_BINS, "Simulated track $p_T$ [GeV]"), ("eta", ETA_BINS, r"Simulated track $\eta$")):
        fig, ax = plt.subplots(figsize=(12, 8))
        centers = (bins[:-1] + bins[1:]) / 2
        for name, label, _, _, colour, marker in VARIANTS:
            c = results[name]
            eff, err = errors(c.num[var], c.den[var])
            ax.errorbar(centers, eff, yerr=err, fmt=marker, color=colour, label=label, capsize=3,
                        markersize=6)
        if var == "pt":
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Efficiency")
        ax.set_ylim(0, 1.25)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=16)
        hep.cms.label("Preliminary", data=False, rlabel=rf"$t\bar{{t}}$ PU200, {n_events} events", ax=ax)
        plt.tight_layout()
        plt.savefig(f"{output}/matching_steps_efficiency_{var}.png", dpi=200, bbox_inches="tight")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="./config_pu200.yaml", help="config with purity_threshold")
    parser.add_argument("--input", required=True, nargs="+", help="LST ntuple(s) or wildcard(s)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tree", default="tree")
    parser.add_argument("--oc-min-layers", type=int, default=5, help="drop clusters with less MD layers (main.py: 5)")
    parser.add_argument("--n-events", type=int, default=-1)
    args = parser.parse_args()

    with open(args.config) as f:
        args.purity_threshold = yaml.safe_load(f).get("purity_threshold", 0.75)
    files = sorted(f for pattern in args.input for f in glob.glob(pattern))
    if not files:
        sys.exit(f"no input files match {args.input}")
    os.makedirs(args.output, exist_ok=True)

    branches = ["sim_pt", "sim_eta", "sim_q", "sim_vx", "sim_vy", "sim_vz", "t3_pt", "t3_simIdx", "t3_lsIdx0",
                "t3_lsIdx1", "ls_mdIdx0", "ls_mdIdx1", "md_simIdx", "md_layer", "oc_t3Idx", "sim_ocIdxAll",
                "sim_ocIdxAllFrac"]
    results = {name: Counts() for name, *_ in VARIANTS}
    totals = dict.fromkeys(["clusters", "sims", "sims_t3", "md_only", "hit_only", "both", "no_t3_matched",
                            "no_t3_matched_md"], 0)
    n_events = 0
    for arrays in uproot.iterate([f"{f}:{args.tree}" for f in files], branches, step_size=20):
        if 0 <= args.n_events < n_events + len(arrays):
            arrays = arrays[: args.n_events - n_events]
        if len(arrays) == 0:
            break
        process(arrays, results, totals, args)
        n_events += len(arrays)
        print(f"processed {n_events} events", flush=True)

    plot(results, args.output, n_events)

    t = totals
    lines = [
        f"Events: {n_events}, clusters with >= {args.oc_min_layers} MD layers: {t['clusters'] / n_events:.0f} per event, "
        f"MD purity threshold: {args.purity_threshold}",
        f"Selected sim tracks: {t['sims']}, with a truth-matched T3: {t['sims_t3']} ({t['sims_t3'] / t['sims']:.3f})",
        "",
        f"{'':40s} {'matched':>8s} {'tracks':>8s} {'efficiency':>10s}",
    ]
    for name, label, *_ in VARIANTS:
        k, n = results[name].total
        lines.append(f"{label:40s} {k:8.0f} {n:8.0f} {k / n:10.4f}")
    lines += [
        "",
        "Sim tracks with a truth-matched T3:",
        f"  matched by both          {t['both']}",
        f"  MD matching only         {t['md_only']}",
        f"  hit matching only        {t['hit_only']}",
        "Sim tracks without a truth-matched T3, matched anyway:",
        f"  MD matching              {t['no_t3_matched_md']}",
        f"  hit matching             {t['no_t3_matched']}",
    ]
    text = "\n".join(lines)
    print("\n" + text)
    with open(f"{args.output}/matching_steps.txt", "w") as f:
        f.write(text + "\n")
    print(f"\nPlots saved to: {args.output}/")


if __name__ == "__main__":
    main()
