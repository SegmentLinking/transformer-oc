"""
Check the harness matcher against LST's own T5 matching in an LSTNtuple.

LST matches a T5 on its 5 base MDs only (getHitsFromT5, 10 hits), while t5_hitIndices also holds the hits of
extension layers. So the comparison with t5_pMatched and t5_simIdx uses 5-layer T5s only. For extended T5s, the
script counts how often matching on all stored hits changes the matched sim.

Logs key=value counts per event and in total. Expected: matched sims agree for all 5-layer T5s. t5_pMatched can
differ below 0.5, where LST skips hit-sim combinations whose most common entry is "no sim".

Usage (from the repo root):
    apptainer exec /ceph/users/atuna/ml-tracking.sif python3 -m harness.validate_t5_matching \
        --input LSTNtuple.root --n_events 3
"""

from argparse import ArgumentParser
from collections import Counter

import uproot

from harness.matching import MATCH_FRACTION, UNMATCHED, best_sim, hit_sims_from_pairs
from harness.ntuple import MD_HIT_COLUMNS, hit_sim_pairs

T5_COLUMNS = ["t5_hitIndices", "t5_nLayers", "t5_pMatched", "t5_simIdx"]


def compare_event(event) -> Counter:
    hit_sims = hit_sims_from_pairs(pairs=hit_sim_pairs(event=event))
    counts = Counter()
    for hits, n_layers, p_matched, lst_sim in zip(
        event["t5_hitIndices"].tolist(),
        event["t5_nLayers"].tolist(),
        event["t5_pMatched"].tolist(),
        event["t5_simIdx"].tolist(),
    ):
        sim, fraction = best_sim(hits=hits, hit_sims=hit_sims)
        harness_sim = sim if fraction > MATCH_FRACTION else UNMATCHED
        lst_sim = lst_sim if lst_sim >= 0 else UNMATCHED
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
    return counts


def main() -> None:
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--n_events", type=int, default=1)
    args = parser.parse_args()

    total = Counter()
    with uproot.open(args.input) as f:
        events = f["tree"].arrays(MD_HIT_COLUMNS + T5_COLUMNS, entry_stop=args.n_events)
    for entry, event in enumerate(events):
        counts = compare_event(event=event)
        total += counts
        print(
            f"[event] entry={entry} "
            + " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        )
    print("[total] " + " ".join(f"{k}={v}" for k, v in sorted(total.items())))


if __name__ == "__main__":
    main()
