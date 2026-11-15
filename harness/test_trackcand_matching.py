"""
Tests for harness/trackcand_matching.py (plain asserts; runs without pytest).

From the repo root:
apptainer exec /ceph/users/atuna/ml-tracking.sif python3 -m harness.test_trackcand_matching
"""

from harness.trackcand_matching import UNMATCHED, best_sim, matched_sim

# Hits 1-4: sim 5; hit 5: sims 5 and 7; hits 6-7: sim 7; hits 8-9: sim 8; hit 10: no sim (not in the map)
HIT_SIMS = {
    1: [5],
    2: [5],
    3: [5],
    4: [5],
    5: [5, 7],
    6: [7],
    7: [7],
    8: [8],
    9: [8],
}


def test_best_sim_counts_unique_hits():
    # Hit 1 three times still counts once: sim 5 has 4 of 5 unique hits
    assert best_sim(hits=[1, 1, 1, 2, 3, 4, 6], hit_sims=HIT_SIMS) == (5, 0.8)


def test_best_sim_hit_with_no_sim_counts_in_the_fraction():
    # Hit 10 has no sim but is one of the 4 unique hits
    assert best_sim(hits=[1, 2, 3, 10], hit_sims=HIT_SIMS) == (5, 0.75)


def test_best_sim_hit_with_two_sims_counts_for_both():
    # Hit 5 counts for sim 5 and sim 7: sim 7 has 3 of 3
    assert best_sim(hits=[5, 6, 7], hit_sims=HIT_SIMS) == (7, 1.0)


def test_best_sim_tie_goes_to_smaller_sim():
    # Sim 7 and sim 8 have 2 hits each
    assert best_sim(hits=[6, 7, 8, 9], hit_sims=HIT_SIMS) == (7, 0.5)
    # Hit 5 carries sims 5 and 7, both at 1 of 1
    assert best_sim(hits=[5], hit_sims=HIT_SIMS) == (5, 1.0)


def test_best_sim_no_sim_hits_outnumber_best_sim():
    # Hits 10, 11, 12 have no sim: 3 no-sim hits beat sim 5's 2 hits
    assert best_sim(hits=[1, 2, 10, 11, 12], hit_sims=HIT_SIMS) == (UNMATCHED, 0.0)


def test_best_sim_no_sim_hits_tie_best_sim_first_appearance_wins():
    # 2 no-sim hits and 2 hits of sim 5: whichever appears first in the hit list wins
    assert best_sim(hits=[10, 11, 1, 2], hit_sims=HIT_SIMS) == (UNMATCHED, 0.0)
    assert best_sim(hits=[1, 2, 10, 11], hit_sims=HIT_SIMS) == (5, 0.5)
    # Sims 7 and 8 tie at 2 hits; sim 8's first hit comes before the first no-sim hit, so a sim wins
    assert best_sim(hits=[8, 10, 6, 7, 9, 11], hit_sims=HIT_SIMS) == (7, 2 / 6)


def test_best_sim_no_hit_with_a_sim():
    assert best_sim(hits=[10, 11], hit_sims=HIT_SIMS) == (UNMATCHED, 0.0)


def test_matched_sim_needs_strictly_more_than_the_fraction():
    # 4 of 5 = 0.8: matched
    assert matched_sim(hits=[1, 2, 3, 4, 6], hit_sims=HIT_SIMS) == 5
    # 3 of 4 = 0.75 exactly: not matched (strict, as LST)
    assert matched_sim(hits=[1, 2, 3, 10], hit_sims=HIT_SIMS) == UNMATCHED
    assert matched_sim(hits=[10], hit_sims=HIT_SIMS) == UNMATCHED


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_"):
            func()
            print(f"[test] name={name} result=passed")
