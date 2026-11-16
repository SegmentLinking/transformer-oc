"""
Track-level metrics for one event's track candidates, following LST's conventions.

- Denominator: sims with LST's getDenomSimTrkType == 7 (trkCore.cc).
- Efficiency: a denominator sim is found if at least one track candidate is matched to it.
- Fake: a track candidate matched to no sim (trackcand_simIdx < 0).
- Duplicate: a track candidate matched to a stored sim that more than one track candidate is matched
  to (write_lst_ntuple.cc, t5_isDuplicate).

Sim indices refer to the full sim list, but a nominal-mode ntuple stores only the first n_sims sims (the hard
scatter of the current bunch crossing). A track candidate matched to a sim with index >= n_sims is matched to
pileup: not a fake, but never a duplicate.
"""

import numpy as np

SIM_COLUMNS = [
    "sim_q",
    "sim_pt",
    "sim_eta",
    "sim_phi",
    "sim_vtxperp",
    "sim_vz",
    "sim_bunchCrossing",
    "sim_event",
]
DENOMINATOR_PT_MIN = 1.0
DENOMINATOR_ABS_ETA_MAX = 2.4
DENOMINATOR_VTXPERP_MAX = 2.5
DENOMINATOR_VERTEX_ABS_Z_MAX = 30.0
# In time (other bunch crossings are nonzero, negative or positive) and hard scatter (pileup events are 1, 2, ...)
DENOMINATOR_BUNCH_CROSSING = 0
DENOMINATOR_SIM_EVENT = 0


def is_denominator(
    sims: dict[str, np.ndarray],
    pt_min: float = DENOMINATOR_PT_MIN,
) -> np.ndarray:
    """
    True for sims in LST's efficiency denominator, shape (n_sims,).

    Charged, pT >= pt_min, |eta| <= 2.4, production vertex rt <= 2.5 cm and |z| <= 30 cm, and hard scatter of the
    current bunch crossing (always true in a nominal-mode ntuple). Efficiency vs pT passes pt_min=0.
    """
    return (
        (sims["sim_q"] != 0)
        & (sims["sim_pt"] >= pt_min)
        & (np.abs(sims["sim_eta"]) <= DENOMINATOR_ABS_ETA_MAX)
        & (sims["sim_vtxperp"] <= DENOMINATOR_VTXPERP_MAX)
        & (np.abs(sims["sim_vz"]) <= DENOMINATOR_VERTEX_ABS_Z_MAX)
        & (sims["sim_bunchCrossing"] == DENOMINATOR_BUNCH_CROSSING)
        & (sims["sim_event"] == DENOMINATOR_SIM_EVENT)
    )


def n_trackcands_per_sim(
    trackcand_simIdx: np.ndarray,
    n_sims: int,
) -> np.ndarray:
    """
    Number of track candidates matched to each stored sim, shape (n_sims,).

    trackcand_simIdx: each track candidate's matched sim index, shape (n_trackcands,).

    Track candidates matched to no sim (< 0) or to a sim that isn't stored (>= n_sims) are not counted.
    """
    is_stored = (trackcand_simIdx >= 0) & (trackcand_simIdx < n_sims)
    # bincount: how often each value 0, 1, ..., n_sims - 1 appears
    return np.bincount(trackcand_simIdx[is_stored], minlength=n_sims)


def is_duplicate(
    trackcand_simIdx: np.ndarray,
    n_trackcands_of_sim: np.ndarray,
) -> np.ndarray:
    """
    True for track candidates matched to a stored sim with more than one matched track candidate,
    shape (n_trackcands,).

    n_trackcands_of_sim is from n_trackcands_per_sim; its length is the number of stored sims.
    """
    duplicate = np.zeros(len(trackcand_simIdx), dtype=bool)
    is_stored = (trackcand_simIdx >= 0) & (trackcand_simIdx < len(n_trackcands_of_sim))
    duplicate[is_stored] = n_trackcands_of_sim[trackcand_simIdx[is_stored]] > 1
    return duplicate
