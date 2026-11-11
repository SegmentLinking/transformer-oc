"""
Tests for harness/matching.py and harness/ntuple.py (plain asserts; runs without pytest).

You can run me like (from the repo root):
> apptainer exec /ceph/users/atuna/ml-tracking.sif python3 -m harness.test_matching
"""

import awkward as ak
import numpy as np

from harness.matching import UNMATCHED, best_sim, hit_sims_from_pairs, matched_sim
from harness.ntuple import hit_sim_pairs

# hit 1, 2, 3: sim 5; hit 4: sims 5 and 7; hit 6: sim 7; hit 9: no sim
HIT_SIMS = hit_sims_from_pairs(
    pairs=np.array([[1, 5], [2, 5], [3, 5], [4, 5], [4, 7], [6, 7]])
)


def test_hit_sims_from_pairs():
    assert HIT_SIMS == {1: [5], 2: [5], 3: [5], 4: [5, 7], 6: [7]}


def test_all_hits_one_sim():
    assert best_sim(hits=[1, 2, 3, 4], hit_sims=HIT_SIMS) == (5, 1.0)
    assert matched_sim(hits=[1, 2, 3, 4], hit_sims=HIT_SIMS) == 5


def test_exactly_75_percent_is_not_matched():
    # 3 of 4 hits from sim 5: LST requires strictly more than 75%
    assert best_sim(hits=[1, 2, 3, 9], hit_sims=HIT_SIMS) == (5, 0.75)
    assert matched_sim(hits=[1, 2, 3, 9], hit_sims=HIT_SIMS) == UNMATCHED


def test_repeated_hit_counts_once():
    # hit 1 appears twice (e.g. shared by two MDs): 4 unique hits, 3 from sim 5
    assert best_sim(hits=[1, 1, 2, 3, 9], hit_sims=HIT_SIMS) == (5, 0.75)


def test_multi_sim_hit_counts_for_each_sim():
    # hit 4 carries sims 5 and 7: sim 7 has hits 4 and 6, sim 5 only hit 4
    assert best_sim(hits=[4, 6], hit_sims=HIT_SIMS) == (7, 1.0)


def test_tie_goes_to_smaller_sim():
    assert best_sim(hits=[1, 6], hit_sims=HIT_SIMS) == (5, 0.5)


def test_no_sim_hits():
    assert best_sim(hits=[9], hit_sims=HIT_SIMS) == (UNMATCHED, 0.0)
    assert matched_sim(hits=[9], hit_sims=HIT_SIMS) == UNMATCHED


def test_hit_sim_pairs_from_ntuple_event():
    # MD 0: hits 10 (sim 5) and 11 (no sim); MD 1: hits 10 (shared) and 12 (sims 7 and 5)
    event = ak.Record(
        {
            "md_anchor_hitIdx": [10, 10],
            "md_other_hitIdx": [11, 12],
            "md_anchor_simIdxs": [[5], [5]],
            "md_other_simIdxs": [[], [7, 5]],
        }
    )
    pairs = hit_sim_pairs(event=event)
    assert np.array_equal(pairs, [[10, 5], [12, 5], [12, 7]])


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_"):
            func()
            print(f"{name}: passed")
