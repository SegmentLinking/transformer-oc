"""
Tests for harness/metrics.py.

From the repo root:
apptainer exec /ceph/users/atuna/ml-tracking.sif python3 -m pytest harness/test_metrics.py
"""

import numpy as np

from harness.trackcand_matching import UNMATCHED
from harness.metrics import (
    is_denominator,
    is_duplicate,
    n_trackcands_per_sim,
)

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
