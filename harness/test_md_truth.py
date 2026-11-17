"""
Tests for harness/md_truth.py.

From the repo root:
apptainer exec /ceph/users/atuna/ml-tracking.sif python3 -m pytest harness/test_md_truth.py
"""

import awkward as ak
import pytest

from harness.md_truth import hit_sims_from_mds


def test_hit_sims_from_mds():
    # MD 0: hits 10 (sim 5) and 11 (no sim)
    # MD 1: hits 11 (no sim) and 12 (sims 7 and 5); hit 11 is shared with MD 0
    # MD 2: hits 12 (sim 9 here, so hit 12 collects 5, 7 and 9) and 13 (sim 5)
    # MD 3: pLS, hits 20 and 21 (sim 5), skipped
    event = ak.Record(
        {
            "md_anchor_hitIdx": [10, 11, 12, 20],
            "md_other_hitIdx": [11, 12, 13, 21],
            "md_anchor_simIdxs": [[5], [], [9], [5]],
            "md_other_simIdxs": [[], [7, 5], [5], [5]],
            "md_isPLS": [False, False, False, True],
        }
    )
    hit_sims = hit_sims_from_mds(event=event)
    assert hit_sims == {10: [5], 12: [5, 7, 9], 13: [5]}


def test_hit_sims_from_mds_rejects_several_events():
    event = {
        "md_anchor_hitIdx": [10],
        "md_other_hitIdx": [11],
        "md_anchor_simIdxs": [[5]],
        "md_other_simIdxs": [[5]],
        "md_isPLS": [False],
    }
    # Two events: an ak.Array, not an ak.Record
    events = ak.Array([event, event])
    with pytest.raises(TypeError):
        hit_sims_from_mds(event=events)
