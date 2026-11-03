"""Transformer clusters (oc) vs LST T5 / T4 / T5+T4 with the definitions of analysis/main.py.

Input: LST ntuple(s) written by `lst --md --ls --t3 --t4 --t5 --transformer_model <pt>`, which hold the
oc_t3Idx, t5_t3Idx0/1, t4_t3Idx0/1, t3_lsIdx0/1, ls_mdIdx0/1, md_simIdx and md_layer branches.

Every definition is the one main.py uses, which differs between the transformer and the LST objects.

Minidoublet (MD) matching: the MDs of a T3 are (ls0.md0, ls0.md1, ls1.md0, ls1.md1), so the shared MD is
counted twice; the MDs of an object are those of all of its T3s (a T5 or a T4 has 2 T3s, a cluster any
number). The object is matched to the majority sim track of its MDs if that sim track has a large enough
fraction of the MDs (MDs without a sim track count in the denominator). An object matched to a pileup sim
track, which is not in the sim_* branches, is not a fake.

Sim tracks: q != 0, pT > 0.8 GeV, |eta| < 2.4, |vz| < 30 cm, vxy < 2.5 cm; the pT plots of the efficiency
also use eta_cut of the config (and the pT plots of the transformer duplicate rate).

Transformer clusters (main.py calculate_tracking_metrics); clusters with less than --oc-min-layers distinct
MD layers are dropped:
  efficiency      selected sim tracks with at least one T3 matched to them (t3_simIdx, T3 pT < 2000 GeV as
                  in the training graphs) that are matched by a cluster (MD fraction >= purity_threshold)
  duplicate rate  of the same sim tracks, those matched by more than one cluster
  fake rate       clusters with 0.8 < pT < 100 GeV with MD fraction < purity_threshold
  pT above the last bin goes into the last bin.

T5s, T4s, T5 + T4 (main.py load_t5_data / calculate_t5_metrics):
  efficiency      selected sim tracks (no T3 requirement) with sim_<obj>IdxAllFrac > purity_threshold
                  (LST hit matching) for at least one object
  fake rate       objects with 0.8 < pT < 100 GeV with MD fraction <= purity_threshold
  duplicate rate  of the same objects, those flagged as duplicate by LST (<obj>_isDuplicate: the object shares
                  a hit-matched sim track with another object; recomputed for T5 + T4)
  pT outside the bins is dropped. The duplicate rate is binned in the object pT, eta (as in main.py).

Bins, eta_cut and purity_threshold are read from the config, as in main.py.

T4 and T5 + T4 are only included with --t4.

    python3 analysis/compare_oc_t5_t4_md.py --config config_pu200.yaml --input 'lst_oc*.root' --output plots [--t4]
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

hep.style.use(hep.style.CMS)

MAX_T3_PT = 2000.0  # T3s kept in the training graphs (data/constants.py)

# name, label, object types, colour, marker
SERIES = [
    ("oc", "Transformer OC", ("oc",), "blue", "o"),
    ("t5", "LST T5s", ("t5",), "red", "s"),
    ("t4", "LST T4s", ("t4",), "green", "^"),
    ("t5t4", "LST T5s + T4s", ("t5", "t4"), "orange", "D"),
]
METRICS = [("efficiency", "Efficiency"), ("fake_rate", "Fake rate"), ("duplicate_rate", "Duplicate rate")]


def flat(x):
    return ak.to_numpy(ak.flatten(x, axis=None))


def offsets(per_event):
    """Index of the first entry of every event in the flattened per_event[event][i]."""
    n = ak.to_numpy(ak.num(per_event, axis=1))
    return np.concatenate([[0], np.cumsum(n)[:-1]]).astype(np.int64)


def to_global(index, per_event):
    """Flattened index[event][i] (an index into per_event[event]) as an index into the flattened per_event."""
    return flat(index).astype(np.int64) + np.repeat(offsets(per_event), ak.to_numpy(ak.num(index, axis=1)))


def majority_match(obj, sim, n_slots, threshold, strict=False):
    """Sim track (-1 if none) and purity of every object, from the sim tracks sim of its MDs; obj is the object
    of every entry of sim and n_slots the number of MDs of every object (MDs without a sim track included).
    Matched if purity >= threshold, or > threshold if strict."""
    n_obj = len(n_slots)
    obj, sim = obj[sim >= 0], sim[sim >= 0]
    best_sim = np.full(n_obj, -1, dtype=np.int64)
    best_count = np.zeros(n_obj)
    if len(sim):
        m = sim.max() + 1
        uniq, counts = np.unique(obj * m + sim, return_counts=True)
        u_obj, u_sim = uniq // m, uniq % m
        order = np.lexsort((-counts, u_obj))  # per object, the most frequent sim track first
        first = order[np.r_[True, u_obj[order][1:] != u_obj[order][:-1]]]
        best_sim[u_obj[first]] = u_sim[first]
        best_count[u_obj[first]] = counts[first]
    purity = np.divide(best_count, n_slots, out=np.zeros(n_obj), where=n_slots > 0)
    matched = purity > threshold if strict else purity >= threshold
    return np.where(matched, best_sim, -1), purity


def n_distinct(obj, values, n_obj):
    """Number of distinct values per object; obj is the object of every entry of values."""
    if len(values) == 0:
        return np.zeros(n_obj, dtype=int)
    m = values.max() + 1
    return np.bincount(np.unique(obj * m + values) // m, minlength=n_obj)


class Counts:
    """Numerator and denominator histograms vs pT and eta."""

    def __init__(self, pt_bins, eta_bins, overflow):
        self.pt_bins, self.eta_bins, self.overflow = pt_bins, eta_bins, overflow
        self.num = {"pt": np.zeros(len(pt_bins) - 1), "eta": np.zeros(len(eta_bins) - 1)}
        self.den = {"pt": np.zeros(len(pt_bins) - 1), "eta": np.zeros(len(eta_bins) - 1)}

    def fill(self, pt, eta, passed, in_pt_plot):
        if self.overflow:
            pt = np.clip(pt, None, self.pt_bins[-1])  # into the last bin
        self.den["pt"] += np.histogram(pt[in_pt_plot], self.pt_bins)[0]
        self.num["pt"] += np.histogram(pt[in_pt_plot & passed], self.pt_bins)[0]
        self.den["eta"] += np.histogram(eta, self.eta_bins)[0]
        self.num["eta"] += np.histogram(eta[passed], self.eta_bins)[0]


def process(arrays, objects, results, totals, args):
    n_sim = ak.to_numpy(ak.num(arrays["sim_pt"]))
    sim_offsets = offsets(arrays["sim_pt"])
    n_sim_total = int(np.sum(n_sim))

    # MDs of every T3 (global indices): (ls0.md0, ls0.md1, ls1.md0, ls1.md1), as in the training graphs
    ls_md = [to_global(arrays[f"ls_mdIdx{m}"], arrays["md_simIdx"]) for m in (0, 1)]
    t3_ls = [to_global(arrays[f"t3_lsIdx{l}"], arrays["ls_mdIdx0"]) for l in (0, 1)]
    t3_mds = np.stack([ls_md[m][t3_ls[l]] for l in (0, 1) for m in (0, 1)], axis=1)
    md_sim, md_layer = flat(arrays["md_simIdx"]).astype(np.int64), flat(arrays["md_layer"]).astype(np.int64)

    # per object type: kept objects, MD-matched sim track (global index, -1 if none or pileup), fake flag, pt, eta
    matched_sim, is_fake, kinematics = {}, {}, {}
    for obj in objects:
        # the T3s of every object: object index (in the chunk) and global T3 index of every (object, T3) pair
        if obj == "oc":
            n_obj = ak.to_numpy(ak.num(arrays["oc_t3Idx"], axis=1))
            t3s_per_obj = flat(ak.num(arrays["oc_t3Idx"], axis=2))
            obj_of_t3 = np.repeat(np.arange(len(t3s_per_obj)), t3s_per_obj)
            t3_off = np.repeat(np.repeat(offsets(arrays["t3_simIdx"]), n_obj), t3s_per_obj)
            t3 = flat(arrays["oc_t3Idx"]).astype(np.int64) + t3_off
        else:
            n_obj = ak.to_numpy(ak.num(arrays[f"{obj}_t3Idx0"], axis=1))
            t3_pair = [to_global(arrays[f"{obj}_t3Idx{i}"], arrays["t3_simIdx"]) for i in (0, 1)]
            t3s_per_obj = np.full(len(t3_pair[0]), 2)
            obj_of_t3 = np.repeat(np.arange(len(t3_pair[0])), 2)
            t3 = np.stack(t3_pair, axis=1).ravel()
        obj_of_md = np.repeat(obj_of_t3, 4)
        mds = t3_mds[t3].ravel()
        # main.py: a cluster is real if its MD fraction is >= purity_threshold, a T5 if it is > purity_threshold
        sim, _ = majority_match(obj_of_md, md_sim[mds], 4 * t3s_per_obj, args.purity_threshold, strict=obj != "oc")
        keep = np.ones(len(sim), dtype=bool)
        if obj == "oc" and args.oc_min_layers > 0:
            keep = n_distinct(obj_of_md, md_layer[mds], len(sim)) >= args.oc_min_layers
        evt = np.repeat(np.arange(len(n_obj)), n_obj)
        is_fake[obj] = (sim < 0)[keep]
        sim = np.where((sim >= 0) & (sim < n_sim[evt]), sim + sim_offsets[evt], -1)
        matched_sim[obj] = sim[keep]
        kinematics[obj] = (flat(arrays[f"{obj}_pt"])[keep], flat(arrays[f"{obj}_eta"])[keep])
        totals["objects"][obj] += int(np.sum(keep))

    sim_pt, sim_eta, sim_q, sim_vx, sim_vy, sim_vz = (flat(arrays[f"sim_{v}"]) for v in ("pt", "eta", "q", "vx", "vy", "vz"))
    selected = ((sim_q != 0) & (sim_pt > 0.8) & (np.abs(sim_eta) < 2.4) & (np.abs(sim_vz) < 30)
                & (np.hypot(sim_vx, sim_vy) < 2.5))
    in_pt_plot = np.ones(n_sim_total, dtype=bool)
    if args.eta_cut is not None:
        in_pt_plot = (sim_eta >= args.eta_cut[0]) & (sim_eta <= args.eta_cut[1])
    # transformer: only sim tracks with at least one T3 matched to them
    t3_sim = arrays["t3_simIdx"][(arrays["t3_simIdx"] >= 0) & (arrays["t3_pt"] < MAX_T3_PT)]
    t3_sim = t3_sim[t3_sim < n_sim]
    has_t3 = np.zeros(n_sim_total, dtype=bool)
    has_t3[to_global(t3_sim, arrays["sim_pt"])] = True
    totals["sims"] += int(np.sum(selected))
    totals["sims_with_t3"] += int(np.sum(selected & has_t3))

    for name, _, objs, _, _ in SERIES:
        if not all(obj in objects for obj in objs):
            continue
        r = results[name]
        if objs == ("oc",):
            s = selected & has_t3
            n_matched = np.bincount(matched_sim["oc"][matched_sim["oc"] >= 0], minlength=n_sim_total)
            r["efficiency"].fill(sim_pt[s], sim_eta[s], (n_matched > 0)[s], in_pt_plot[s])
            r["duplicate_rate"].fill(sim_pt[s], sim_eta[s], (n_matched > 1)[s], in_pt_plot[s])
            pt, eta = kinematics["oc"]
            used = (pt > 0.8) & (pt < 100)
            r["fake_rate"].fill(pt[used], eta[used], is_fake["oc"][used], np.ones(np.sum(used), dtype=bool))
            continue
        # LST objects: efficiency and duplicates from the LST hit matching (objects with > purity_threshold of
        # their hits from one sim track), fakes from the MDs
        n_matched = sum(flat(ak.sum(arrays[f"sim_{obj}IdxAllFrac"] > args.purity_threshold, axis=-1)) for obj in objs)
        s = selected
        r["efficiency"].fill(sim_pt[s], sim_eta[s], (n_matched > 0)[s], in_pt_plot[s])
        for obj in objs:
            # duplicate: a hit-matched sim track (in the sim_* branches) of the object is matched by > 1 object
            idx = arrays[f"{obj}_simIdxAll"]
            idx = idx[idx < ak.num(arrays["sim_pt"])]
            per_obj = flat(ak.num(idx, axis=2))
            per_evt = ak.to_numpy(ak.sum(ak.num(idx, axis=2), axis=1))
            g = flat(idx).astype(np.int64) + np.repeat(sim_offsets, per_evt)
            obj_of_entry = np.repeat(np.arange(len(per_obj)), per_obj)
            duplicate = np.zeros(len(per_obj), dtype=bool)
            duplicate[obj_of_entry[n_matched[g] > 1]] = True
            if len(objs) == 1 and not np.array_equal(duplicate, flat(arrays[f"{obj}_isDuplicate"]) == 1):
                raise RuntimeError(f"{obj}_isDuplicate differs from the recomputed duplicate flag")
            pt, eta = kinematics[obj]
            used = (pt > 0.8) & (pt < 100)
            all_used = np.ones(np.sum(used), dtype=bool)
            r["fake_rate"].fill(pt[used], eta[used], is_fake[obj][used], all_used)
            r["duplicate_rate"].fill(pt[used], eta[used], duplicate[used], all_used)


def clopper_pearson_error(k, N):
    from statsmodels.stats.proportion import proportion_confint  # here so the module imports without statsmodels

    valid = N > 0
    _k = np.where(valid, k, 0)
    _N = np.where(valid, N, 1)
    eff = np.where(valid, _k / _N, 0)
    low, high = proportion_confint(_k, _N, method="beta")
    return [np.where(valid, eff - low, 0), np.where(valid, high - eff, 0)]


def plot_comparison(results, series, pt_bins, eta_bins, output, tag):
    centers = {"pt": (pt_bins[:-1] + pt_bins[1:]) / 2, "eta": (eta_bins[:-1] + eta_bins[1:]) / 2}
    xlabels = {
        ("efficiency", "pt"): "Simulated track $p_T$ [GeV]", ("efficiency", "eta"): r"Simulated track $\eta$",
        # main.py: per sim track for the transformer, per object for the LST objects
        ("duplicate_rate", "pt"): "Track $p_T$ [GeV]", ("duplicate_rate", "eta"): r"Track $\eta$",
        ("fake_rate", "pt"): "Reconstructed track $p_T$ [GeV]", ("fake_rate", "eta"): r"Reconstructed track $\eta$",
    }
    ylims = {"efficiency": (0, 1.19), "fake_rate": (0, 0.3), "duplicate_rate": (0, 0.2)}
    for metric, ylabel in METRICS:
        for var in ("pt", "eta"):
            fig, ax = plt.subplots(figsize=(12, 8))
            top = 0.0
            for name, label, _, colour, marker in series:
                counts = results[name][metric]
                num, den = counts.num[var], counts.den[var]
                rate = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
                ax.errorbar(centers[var], rate, yerr=clopper_pearson_error(num, den), fmt=marker, color=colour,
                            label=label, capsize=3, markersize=5)
                top = max(top, np.max(rate, initial=0.0))
            ax.set_xlabel(xlabels[(metric, var)])
            ax.set_ylabel(ylabel)
            low, high = ylims[metric]
            ax.set_ylim(low, high)
            # ax.set_ylim(low, high if metric == "efficiency" else max(high, min(1.19, 1.4 * top)))
            hep.cms.label("Preliminary", data=False, rlabel=r"Phase-2 (14 TeV)", ax=ax)
            ax.text(0.95, 0.95, r"$t\bar{t}$, $\langle\mu\rangle=200$", transform=ax.transAxes, fontsize=20,
                    verticalalignment="top", horizontalalignment="right")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output}/comparison_md_{metric}_{var}{tag}.png", dpi=300, bbox_inches="tight")
            plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="./config_pu200.yaml", help="config with the bins, eta_cut and purity_threshold")
    parser.add_argument("--input", required=True, nargs="+", help="LST ntuple(s) or wildcard(s)")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--tree", default="tree")
    parser.add_argument("--oc-min-layers", type=int, default=5, help="drop clusters with less MD layers (main.py: 5)")
    parser.add_argument("--n-events", type=int, default=-1)
    parser.add_argument("--t4", action="store_true", help="also plot T4 and T5 + T4")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    args.purity_threshold = config.get("purity_threshold", 0.75)
    args.eta_cut = config.get("eta_cut", (-2.5, 2.5))
    pt_bins = np.linspace(config.get("pt_bins_start", 0.5), config.get("pt_bins_end", 2.0), config.get("pt_bins_count", 50))
    eta_bins = np.linspace(config.get("eta_bins_start", -4), config.get("eta_bins_end", 4), config.get("eta_bins_count", 50))

    files = sorted(f for pattern in args.input for f in glob.glob(pattern))
    if not files:
        sys.exit(f"no input files match {args.input}")
    os.makedirs(args.output, exist_ok=True)

    available = set(uproot.open(f"{files[0]}:{args.tree}").keys())
    required = {"oc": ["oc_t3Idx"], "t5": ["t5_t3Idx0", "t5_t3Idx1"]}
    if args.t4:
        required["t4"] = ["t4_t3Idx0", "t4_t3Idx1"]
    objects = []
    for obj, needed in required.items():
        if all(b in available for b in needed):
            objects.append(obj)
        else:
            print(f"WARNING: {obj} skipped, the ntuple has no {', '.join(b for b in needed if b not in available)}")
    branches = ["sim_pt", "sim_eta", "sim_q", "sim_vx", "sim_vy", "sim_vz", "t3_pt", "t3_simIdx",
                "t3_lsIdx0", "t3_lsIdx1", "ls_mdIdx0", "ls_mdIdx1", "md_simIdx", "md_layer"]
    for obj in objects:
        branches += required[obj] + [f"{obj}_pt", f"{obj}_eta"]
        if obj != "oc":
            branches += [f"sim_{obj}IdxAllFrac", f"{obj}_simIdxAll", f"{obj}_isDuplicate"]
    missing = [b for b in branches if b not in available]
    if missing:
        sys.exit(f"the ntuple has no {', '.join(missing)} (run lst with --md --ls --t3 --t4 --t5)")
    series = [s for s in SERIES if all(obj in objects for obj in s[2])]

    # main.py puts pT above the last bin into the last bin for the transformer only
    results = {name: {metric: Counts(pt_bins, eta_bins, overflow=name == "oc") for metric, _ in METRICS}
               for name, *_ in series}
    totals = {"sims": 0, "sims_with_t3": 0, "objects": {obj: 0 for obj in objects}}
    n_events = 0
    for arrays in uproot.iterate([f"{f}:{args.tree}" for f in files], branches, step_size=20):
        if 0 <= args.n_events < n_events + len(arrays):
            arrays = arrays[: args.n_events - n_events]
        if len(arrays) == 0:
            break
        process(arrays, objects, results, totals, args)
        n_events += len(arrays)
        print(f"processed {n_events} events")

    tag = f"_oc{args.oc_min_layers}layers" if args.oc_min_layers != 5 else ""
    plot_comparison(results, series, pt_bins, eta_bins, args.output, tag)

    print("\n" + "=" * 60)
    print(f"Events: {n_events}, purity threshold: {args.purity_threshold}, eta cut (pT plots): {args.eta_cut}, "
          f"clusters with >= {args.oc_min_layers} MD layers")
    print(f"Selected sim tracks: {totals['sims']} (LST denominator), with a matched T3: {totals['sims_with_t3']}"
          f" (transformer denominator)")
    print(f"{'':16s} {'objects/evt':>11s} {'efficiency':>11s} {'fake rate':>10s} {'dup. rate':>10s}")
    for name, label, objs, _, _ in series:
        r = results[name]
        rate = lambda m: r[m].num["eta"].sum() / max(r[m].den["eta"].sum(), 1)
        n_obj = sum(totals["objects"][o] for o in objs) / max(n_events, 1)
        print(f"{label:16s} {n_obj:11.0f} {rate('efficiency'):11.3f} {rate('fake_rate'):10.3f} {rate('duplicate_rate'):10.3f}")
    print("(integrated over the eta plots; duplicate rate: per sim track for the transformer, per object for LST)")
    print(f"\nPlots saved to: {args.output}/")


if __name__ == "__main__":
    main()
