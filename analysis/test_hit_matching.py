"""
test_hit_matching.py - tests for the LST-style cluster matching in hit_matching.py

You can run me like (from the repo root):
> apptainer exec /ceph/users/atuna/ml-tracking.sif python3 analysis/test_hit_matching.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from constants import HIT_UNMATCHED as U
from hit_matching import match_clusters, select_sims


def _cand(rows, width=2):
    """(N, H, width) candidates from per-T3 lists of per-hit sim lists."""
    return np.array([[h + [U] * (width - len(h)) for h in t3] for t3 in rows], dtype=np.int64)


def test_shared_hits_counted_once():
    # two T3s sharing hits 3 and 4: 7 distinct hits, all on sim 5 except hit 6 (sim 8)
    hit_idx = np.array([[0, 1, 2, 3, 4, 5], [3, 4, 5, 6, 6, 5]])
    cand = _cand([[[5]] * 6, [[5], [5], [5], [8], [8], [5]]])
    layer = np.array([[1, 1, 2, 2, 3, 3], [2, 3, 3, 4, 4, 3]])
    m = match_clusters(np.array([0, 0]), hit_idx, cand, layer)
    assert m["label"].tolist() == [0] and m["n_t3"].tolist() == [2]
    assert m["n_hits"].tolist() == [7] and m["n_layers"].tolist() == [4]
    assert m["sim"].tolist() == [5] and m["n_matched_hits"].tolist() == [6]
    assert np.isclose(m["frac"][0], 6 / 7)


def test_multi_sim_hits_credit_every_sim():
    # every hit has sim 2; hits 0-2 also sim 9: sim 2 has all hits
    hit_idx = np.arange(6).reshape(1, 6)
    cand = _cand([[[9, 2], [9, 2], [9, 2], [2], [2], [2]]])
    m = match_clusters(np.array([3]), hit_idx, cand, np.zeros((1, 6)))
    assert m["sim"].tolist() == [2] and m["frac"].tolist() == [1.0]


def test_tie_to_larger_sim_and_noise_and_unclustered():
    hit_idx = np.array([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17]])
    cand = _cand([[[4], [4], [4], [6], [6], [6]], [[]] * 6, [[1]] * 6])
    labels = np.array([0, 1, -1])
    m = match_clusters(labels, hit_idx, cand, np.zeros((3, 6)))
    assert m["label"].tolist() == [0, 1]
    assert m["sim"].tolist() == [6, -1]
    assert np.allclose(m["frac"], [0.5, 0.0])


def test_no_clusters():
    m = match_clusters(np.array([-1]), np.zeros((1, 6), int), _cand([[[1]] * 6]), np.zeros((1, 6)))
    assert len(m["label"]) == 0 and len(m["frac"]) == 0


def test_select_sims():
    # pt, eta, phi, q, vx, vy, vz, vtxperp
    s = np.array([
        [1.0, 0.0, 0, 1, 0, 0, 0, 0],    # selected
        [0.5, 0.0, 0, 1, 0, 0, 0, 0],    # low pT
        [1.0, 2.5, 0, 1, 0, 0, 0, 0],    # forward
        [1.0, 0.0, 0, 0, 0, 0, 0, 0],    # neutral
        [1.0, 0.0, 0, -1, 3, 0, 0, 3],   # displaced
    ])
    assert select_sims(s).tolist() == [True, False, False, False, False]


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_"):
            func()
            print(f"{name}: passed")
