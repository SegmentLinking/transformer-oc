"""
check_t3_pmatched.py - cross check t3_pMatched against the per-hit sim indices

For every T3 in an LST Ntuple, recompute the matched fraction "by hand" from the
six t3_hit_{0..5}_simIdx branches and compare it to the t3_pMatched branch
computed by LST. The by-hand calculation mirrors matchedSimTrkIdxsAndFracs in
trkCore.cc: a hit can have several sim indices, so every combination of one sim
index per hit is tried, and the best fraction for the most common sim index is kept.

It reports how many T3s agree within --tolerance, how many are above or below
t3_pMatched, and how often hits have more than one sim index. No mismatches are
expected now that all sim indices per hit are saved in the ntuple.

You can run me like:
> apptainer exec ml-tracking.sif python3 analysis/check_t3_pmatched.py --input LSTNtuple.root --max-events 1
"""
import argparse
import itertools
import uproot
import awkward as ak
import logging
logger = logging.getLogger(__name__)

N_HITS = 6
UNMATCHED = -1
HIT_BRANCHES = [f"t3_hit_{it}_simIdx" for it in range(N_HITS)]

def main():
    args = options()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info(f"Analyzing {args.input}")
    analyze_rfile(args.input, args.treename, args.max_events, args.max_print, args.tolerance)

def analyze_rfile(input_file: str, treename: str, max_events: int, max_print: int, tolerance: float):
    with uproot.open(input_file) as fi:
        logger.info(f"Opening {treename}")
        tr = fi[treename]
        analyze_ttree(tr, max_events, max_print, tolerance)

def analyze_ttree(tr, max_events: int, max_print: int, tolerance: float):
    logger.info(f"Fetching data from {tr.name}")
    data = tr.arrays(["t3_pMatched"] + HIT_BRANCHES, entry_stop=max_events)
    n_total, n_agree, n_below, n_above, n_printed = 0, 0, 0, 0, 0
    n_hits, n_hits_multi, n_t3_multi = 0, 0, 0

    for i_event in range(len(data)):
        logger.info(f"Processing event {i_event}")
        ref = ak.to_list(data["t3_pMatched"][i_event])
        # sim_idxs[it][i_t3] is the list of sim indices for hit `it` of T3 `i_t3`
        sim_idxs = [ak.to_list(data[branch][i_event]) for branch in HIT_BRANCHES]

        for i_t3 in range(len(ref)):
            hits = [sim_idxs[it][i_t3] for it in range(N_HITS)]
            # a hit has multiple parents if it has more than one distinct sim index
            n_multi = sum(len(set(sim_idxs)) > 1 for sim_idxs in hits)
            n_hits += len(hits)
            n_hits_multi += n_multi
            n_t3_multi += n_multi > 0
            by_hand = p_matched(hits)
            diff = by_hand - ref[i_t3]
            n_total += 1
            if abs(diff) <= tolerance:
                n_agree += 1
                continue
            if diff < 0:
                n_below += 1
            else:
                n_above += 1
            if n_printed < max_print:
                n_printed += 1
                logger.info(f"Mismatch: event {i_event} t3 {i_t3} hits {hits} by hand {by_hand:.3f} t3_pMatched {ref[i_t3]:.3f}")

    logger.info(f"Checked {n_total} T3s in {len(data)} events")
    logger.info(f"Agree:                 {n_agree} ({fraction(n_agree, n_total)})")
    logger.info(f"By hand < t3_pMatched: {n_below} ({fraction(n_below, n_total)})")
    logger.info(f"By hand > t3_pMatched: {n_above} ({fraction(n_above, n_total)})")
    logger.info(f"T3 hits with >1 sim index:      {n_hits_multi} of {n_hits} ({fraction(n_hits_multi, n_hits)})")
    logger.info(f"T3s with at least one such hit: {n_t3_multi} of {n_total} ({fraction(n_t3_multi, n_total)})")
    if n_below or n_above:
        logger.warning("Mismatches found: none are expected now that all sim indices are saved")

def p_matched(hits: list) -> float:
    # Mirrors matchedSimTrkIdxsAndFracs in trkCore.cc.
    # hits: one list of sim indices per hit; an empty list means the hit is unmatched.
    #  - try every combination of one sim index per hit, and keep the best fraction
    #  - within a combination, take the most common sim index
    #    (ties: whichever appears first in the hit list)
    #  - if the most common "sim index" is unmatched, that combination contributes 0
    candidates = [sim_idxs if sim_idxs else [UNMATCHED] for sim_idxs in hits]
    order = list(dict.fromkeys(itertools.chain.from_iterable(candidates)))  # first-appearance order, for ties
    best_fraction = 0.0
    for combination in itertools.product(*candidates):
        best = max(order, key=combination.count)  # max returns the first maximum
        if best >= 0:
            best_fraction = max(best_fraction, combination.count(best) / len(combination))
    return best_fraction

def fraction(num: int, den: int) -> str:
    return f"{100 * num / den:.2f}%" if den else "n/a"

def options():
    parser = argparse.ArgumentParser(description="Cross check t3_pMatched against t3_hit_*_simIdx in a LST Ntuple file")
    parser.add_argument("--input", type=str, required=True, help="Input LST Ntuple file")
    parser.add_argument("--treename", type=str, default="tree", help="Input LST Ntuple tree name")
    parser.add_argument("--max-events", type=int, default=None, help="Number of events to check (default: all)")
    parser.add_argument("--max-print", type=int, default=20, help="Number of mismatches to print")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Allowed difference between the two values")
    return parser.parse_args()

if __name__ == "__main__":
    main()
