"""
Tests for the harness, one section per module, bottom-up.

From the repo root:
apptainer exec /ceph/users/atuna/ml-tracking.sif python3 -m pytest harness/test_harness.py
"""

import os

import awkward as ak
import numpy as np
import pytest

from harness.events import read_file_list
from harness.metrics import (
    is_denominator,
    is_duplicate,
    n_trackcands_per_sim,
)
from harness.pmatched import UNMATCHED, best_sim, matched_sim
from harness.trackcands import (
    NO_HIT,
    trackcands_from_mds_matched_to_sim,
    trackcands_from_t5s,
)

# pmatched.py


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


# metrics.py


# Sim 0 passes every cut; each of sims 1-7 fails exactly one
SIMS = {
    "sim_q": np.array([1, 0, -1, 1, 1, 1, 1, 1]),
    "sim_pt": np.array([2.0, 2.0, 0.9, 2.0, 2.0, 2.0, 2.0, 2.0]),
    "sim_eta": np.array([0.5, 0.5, 0.5, 2.5, 0.5, 0.5, 0.5, 0.5]),
    "sim_vtxperp": np.array([0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0]),
    "sim_vz": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 31.0, 0.0, 0.0]),
    "sim_bunchCrossing": np.array([0, 0, 0, 0, 0, 0, 1, 0]),
    "sim_event": np.array([0, 0, 0, 0, 0, 0, 0, 2]),
}


def test_is_denominator():
    expected = [True, False, False, False, False, False, False, False]
    assert is_denominator(sims=SIMS).tolist() == expected
    # Without the pT cut, sim 2 (pT 0.9) joins
    expected = [True, False, True, False, False, False, False, False]
    assert is_denominator(sims=SIMS, pt_min=0.0).tolist() == expected


def test_n_trackcands_per_sim():
    trackcand_simIdx = np.array([0, 0, 2, UNMATCHED, 2, 2])
    n_trackcands_of_sim = n_trackcands_per_sim(
        trackcand_simIdx=trackcand_simIdx,
        n_sims=4,
    )
    assert n_trackcands_of_sim.tolist() == [2, 0, 3, 0]
    # Sim 4 is not stored (index >= n_sims): not counted
    trackcand_simIdx = np.array([0, 4, 4])
    n_trackcands_of_sim = n_trackcands_per_sim(
        trackcand_simIdx=trackcand_simIdx,
        n_sims=4,
    )
    assert n_trackcands_of_sim.tolist() == [1, 0, 0, 0]


def test_is_duplicate():
    # 4 stored sims. Sim 0 has two track candidates, sim 2 one; sim 9 (not stored, so pileup) has two, but
    # they aren't duplicates
    trackcand_simIdx = np.array([0, 0, 2, UNMATCHED, 9, 9])
    n_trackcands_of_sim = n_trackcands_per_sim(
        trackcand_simIdx=trackcand_simIdx,
        n_sims=4,
    )
    duplicate = is_duplicate(
        trackcand_simIdx=trackcand_simIdx,
        n_trackcands_of_sim=n_trackcands_of_sim,
    )
    assert duplicate.tolist() == [True, True, False, False, False, False]


# trackcands.py


# T5 0: 10 hits, never removed
# T5 1: 12 hits (2 from an extension layer), cross-cleaned against a pT5/pT3 (bit 4)
# T5 2: 10 hits, removed by the first pass of T5-T5 duplicate cleaning (bit 1)
EVENT = ak.Record(
    {
        "t5_hitIndices": [
            list(range(0, 10)),
            list(range(10, 22)),
            list(range(30, 40)),
        ],
        "t5_simIdx": [3, -999, 5],
        "t5_dnnScore": [0.9, 0.8, 0.7],
        "t5_pt": [1.5, 2.5, 3.5],
        "t5_eta": [0.1, 0.2, 0.3],
        "t5_phi": [-1.0, 0.0, 1.0],
        "t5_isDupBitmask": [0, 4, 1],
    }
)


def test_trackcands_from_t5s_keeps_every_t5():
    trackcands = trackcands_from_t5s(event=EVENT)
    assert trackcands.trackcand_hitIndices.tolist() == [
        list(range(0, 10)),
        list(range(10, 20)),
        list(range(30, 40)),
    ]
    assert trackcands.trackcand_simIdx.tolist() == [3, -999, 5]
    assert trackcands.trackcand_score.tolist() == [0.9, 0.8, 0.7]
    assert trackcands.trackcand_pt.tolist() == [1.5, 2.5, 3.5]
    assert trackcands.trackcand_eta.tolist() == [0.1, 0.2, 0.3]
    assert trackcands.trackcand_phi.tolist() == [-1.0, 0.0, 1.0]


def test_trackcands_from_t5s_drop_duplicate_cleaned():
    # T5 2 (bit 1) is dropped; T5 1 (bit 4 only) is kept
    trackcands = trackcands_from_t5s(
        event=EVENT,
        drop_duplicate_cleaned=True,
    )
    assert trackcands.trackcand_hitIndices.tolist() == [
        list(range(0, 10)),
        list(range(10, 20)),
    ]
    assert trackcands.trackcand_simIdx.tolist() == [3, -999]
    assert trackcands.trackcand_score.tolist() == [0.9, 0.8]
    assert trackcands.trackcand_pt.tolist() == [1.5, 2.5]
    assert trackcands.trackcand_eta.tolist() == [0.1, 0.2]
    assert trackcands.trackcand_phi.tolist() == [-1.0, 0.0]


def test_trackcands_from_t5s_rejects_several_events():
    # Two events: an ak.Array, not an ak.Record
    events = ak.Array([EVENT.tolist(), EVENT.tolist()])
    with pytest.raises(TypeError):
        trackcands_from_t5s(event=events)


def test_trackcands_from_t5s_rejects_t5_with_too_few_hits():
    # T5 1 has 9 hits
    event = ak.Record(
        {
            "t5_hitIndices": [list(range(0, 10)), list(range(10, 19))],
            "t5_simIdx": [3, 4],
            "t5_dnnScore": [0.9, 0.8],
            "t5_pt": [1.5, 2.5],
            "t5_eta": [0.1, 0.2],
            "t5_phi": [-1.0, 0.0],
            "t5_isDupBitmask": [0, 0],
        }
    )
    with pytest.raises(ValueError):
        trackcands_from_t5s(event=event)


# Three stored sims (0, 1, 2); sim 7 is pileup (index >= 3)
# MD 0: layer 1, hits 10 and 11, matched to sim 0
# MD 1: layer 2, hits 12 and 13, matched to sim 0
# MD 2: layer 2, hits 14 and 15, matched to sim 1
# MD 3: layer 3, hits 16 and 17, matched to no sim (e.g. one hit from sim 1, one from sim 2)
# MD 4: layer 4, hits 18 and 19, matched to pileup sim 7
# MD 5: pLS, hits 20 and 21, matched to sim 0; skipped
MD_EVENT = ak.Record(
    {
        "md_isPLS": [False, False, False, False, False, True],
        "md_anchor_hitIdx": [10, 12, 14, 16, 18, 20],
        "md_other_hitIdx": [11, 13, 15, 17, 19, 21],
        "md_layer": [1, 2, 2, 3, 4, 0],
        "md_simIdx": [0, 0, 1, -999, 7, 0],
        "sim_pt": [1.5, 2.5, 3.5],
        "sim_eta": [0.1, 0.2, 0.3],
        "sim_phi": [-1.0, 0.0, 1.0],
    }
)


def test_trackcands_from_mds_matched_to_sim():
    # Sim 2 has no MD matched to it; pileup sim 7 and the unmatched MD give no track candidate
    trackcands = trackcands_from_mds_matched_to_sim(event=MD_EVENT)
    assert trackcands.trackcand_hitIndices.tolist() == [
        [10, 11, 12, 13],
        [14, 15, NO_HIT, NO_HIT],
    ]
    assert trackcands.trackcand_simIdx.tolist() == [0, 1]
    assert trackcands.trackcand_score.tolist() == [1.0, 1.0]
    assert trackcands.trackcand_pt.tolist() == [1.5, 2.5]
    assert trackcands.trackcand_eta.tolist() == [0.1, 0.2]
    assert trackcands.trackcand_phi.tolist() == [-1.0, 0.0]


def test_trackcands_from_mds_matched_to_sim_min_layers():
    # Sim 0's MDs span layers 1 and 2; sim 1's only layer 2
    trackcands = trackcands_from_mds_matched_to_sim(
        event=MD_EVENT,
        min_layers=2,
    )
    assert trackcands.trackcand_simIdx.tolist() == [0]
    assert trackcands.trackcand_pt.tolist() == [1.5]
    trackcands = trackcands_from_mds_matched_to_sim(
        event=MD_EVENT,
        min_layers=3,
    )
    assert trackcands.trackcand_hitIndices.shape == (0, 0)
    assert trackcands.trackcand_simIdx.tolist() == []
    assert trackcands.trackcand_pt.tolist() == []


# events.py


def test_read_file_list(tmp_path):
    # Files keep list order
    path = os.path.join(tmp_path, "val.txt")
    with open(path, "w") as list_file:
        list_file.write("b.root\na.root\n")
    assert read_file_list(path=path) == ["b.root", "a.root"]


def test_read_file_list_empty(tmp_path):
    # A dataset with no files has an empty list
    path = os.path.join(tmp_path, "test.txt")
    open(path, "w").close()
    assert read_file_list(path=path) == []
