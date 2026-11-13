"""
Check the harness matcher against LST's own T5 matching in an LSTNtuple.

LST matches a T5 on its 5 base MDs only (getHitsFromT5, 10 hits), while t5_hitIndices also holds the hits of
extension layers. So the comparison with t5_pMatched and t5_simIdx uses 5-layer T5s only. For extended T5s, the
script counts how often matching on all stored hits changes the matched sim.

The first 10 entries of t5_hitIndices are the base hits (extension layers are appended after them), so matching
every T5 on its first 10 hits should reproduce t5_simIdx for all T5s, and the set of sims found by any T5 should
reproduce the sims with a non-empty sim_t5IdxAll (LST's sim-side view, used for efficiency).

Logs key=value counts per event and in total. Expected: matched sims agree for all 5-layer T5s. t5_pMatched can
differ below 0.5, where LST skips hit-sim combinations whose most common entry is "no sim".

Usage (from the repo root):
    apptainer exec /ceph/users/atuna/ml-tracking.sif python3 -m harness.validate_t5_matching \
        --input LSTNtuple.root --n_events 3
"""

from argparse import ArgumentParser, Namespace
from collections import Counter

import awkward as ak
import uproot

from harness.matching import (
    MATCH_FRACTION,
    UNMATCHED,
    best_sim,
    hit_sims_from_pairs,
    matched_sim,
)
from harness.ntuple import MD_HIT_COLUMNS, hit_sim_pairs

T5_COLUMNS = ["t5_hitIndices", "t5_nLayers", "t5_pMatched", "t5_simIdx", "sim_t5IdxAll"]
N_BASE_HITS = 10


def main() -> None:
    args = parse_args()
    events = read_events(path=args.input, n_events=args.n_events)
    total = Counter()
    for entry, event in enumerate(events):
        counts = compare_event(event=event)
        total.update(counts)
        print(f"[event] entry={entry} {format_counts(counts=counts)}")
    print(f"[total] {format_counts(counts=total)}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--n_events", type=int, default=1)
    return parser.parse_args()


def read_events(
    path: str,
    n_events: int,
) -> ak.Array:
    """The MD hit truth and T5 columns of the first n_events events."""
    with uproot.open(path) as f:
        return f["tree"].arrays(MD_HIT_COLUMNS + T5_COLUMNS, entry_stop=n_events)


def compare_event(event: ak.Record) -> Counter:
    """Counts of agreement between the harness matcher and LST's T5 matching in one event."""
    hit_sims = hit_sims_from_pairs(pairs=hit_sim_pairs(event=event))
    counts = Counter()
    sims_found = set()
    for hits, n_layers, p_matched, lst_sim in zip(
        event["t5_hitIndices"].tolist(),
        event["t5_nLayers"].tolist(),
        event["t5_pMatched"].tolist(),
        event["t5_simIdx"].tolist(),
    ):
        sim, fraction = best_sim(hits=hits, hit_sims=hit_sims)
        harness_sim = sim if fraction > MATCH_FRACTION else UNMATCHED
        lst_sim = lst_sim if lst_sim >= 0 else UNMATCHED
        base_sim = matched_sim(hits=hits[:N_BASE_HITS], hit_sims=hit_sims)
        counts["t5_base_hits_sim_agree"] += base_sim == lst_sim
        if base_sim != UNMATCHED:
            sims_found.add(base_sim)
        if n_layers == 5:
            counts["t5_5layer"] += 1
            counts["t5_5layer_sim_agree"] += harness_sim == lst_sim
            fraction_agrees = abs(fraction - p_matched) < 1e-4
            counts["t5_5layer_fraction_agree"] += fraction_agrees
            counts["t5_5layer_fraction_differ_above_0.5"] += (
                not fraction_agrees and max(fraction, p_matched) > 0.5
            )
        else:
            counts["t5_extended"] += 1
            counts["t5_extended_sim_changes"] += harness_sim != lst_sim
    counts["t5"] = len(event["t5_nLayers"])
    lst_sims_found = {
        sim for sim, t5s in enumerate(event["sim_t5IdxAll"].tolist()) if t5s
    }
    counts["sims_found_harness"] = len(sims_found)
    counts["sims_found_lst"] = len(lst_sims_found)
    counts["sims_found_both"] = len(sims_found & lst_sims_found)
    return counts


def format_counts(counts: Counter) -> str:
    """key=value pairs, sorted by key."""
    return " ".join(f"{key}={value}" for key, value in sorted(counts.items()))


if __name__ == "__main__":
    main()
