"""
Tests for harness/trackcands.py (plain asserts; runs without pytest).

From the repo root:
apptainer exec /ceph/users/atuna/ml-tracking.sif python3 -m harness.test_trackcands
"""

import awkward as ak

from harness.trackcands import (
    NO_HIT,
    trackcands_from_mds_matched_to_sim,
    trackcands_from_t5s,
)

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
    assert trackcands.trackcand_score.tolist() == [0.9, 0.8]
    assert trackcands.trackcand_pt.tolist() == [1.5, 2.5]
    assert trackcands.trackcand_eta.tolist() == [0.1, 0.2]
    assert trackcands.trackcand_phi.tolist() == [-1.0, 0.0]


def test_trackcands_from_t5s_rejects_several_events():
    # Two events: an ak.Array, not an ak.Record
    events = ak.Array([EVENT.tolist(), EVENT.tolist()])
    try:
        trackcands_from_t5s(event=events)
    except TypeError:
        return
    raise AssertionError("expected a TypeError for an ak.Array of events")


def test_trackcands_from_t5s_rejects_t5_with_too_few_hits():
    # T5 1 has 9 hits
    event = ak.Record(
        {
            "t5_hitIndices": [list(range(0, 10)), list(range(10, 19))],
            "t5_dnnScore": [0.9, 0.8],
            "t5_pt": [1.5, 2.5],
            "t5_eta": [0.1, 0.2],
            "t5_phi": [-1.0, 0.0],
            "t5_isDupBitmask": [0, 0],
        }
    )
    try:
        trackcands_from_t5s(event=event)
    except ValueError:
        return
    raise AssertionError("expected a ValueError for a T5 with 9 hits")


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
    assert trackcands.trackcand_pt.tolist() == [1.5]
    trackcands = trackcands_from_mds_matched_to_sim(
        event=MD_EVENT,
        min_layers=3,
    )
    assert trackcands.trackcand_hitIndices.shape == (0, 0)
    assert trackcands.trackcand_pt.tolist() == []


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_"):
            func()
            print(f"[test] name={name} result=passed")
