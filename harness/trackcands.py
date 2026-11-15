"""
One event's track candidates as per-event arrays, built from LST T5s (and later from a model's output).
"""

from dataclasses import dataclass

import awkward as ak
import numpy as np

T5_COLUMNS = [
    "t5_hitIndices",
    "t5_dnnScore",
    "t5_pt",
    "t5_eta",
    "t5_phi",
    "t5_isDupBitmask",
]
# The hits of the 5 base MDs; t5_hitIndices appends the hits of extension layers after them
N_T5_BASE_HITS = 10
# t5_isDupBitmask bits set when LST's T5-T5 duplicate cleaning removes a T5, in its first or second pass
# (Kernels.h, rmQuintupletFromMemory). Bit 4 (cross-cleaning against pT5/pT3) is not included.
T5_DUPLICATE_CLEANING_BITS = 1 | 2
# Pads trackcand_hitIndices where a track candidate has fewer hits than the longest; test with hit < 0
NO_HIT = -999


@dataclass
class EventTrackCandidates:
    """
    One event's track candidates: entry i of every field belongs to track candidate i.

    trackcand_hitIndices has shape (n_trackcands, n_hits_max), padded with NO_HIT; the others (n_trackcands,).
    """

    trackcand_hitIndices: np.ndarray
    trackcand_score: np.ndarray
    trackcand_pt: np.ndarray
    trackcand_eta: np.ndarray
    trackcand_phi: np.ndarray


def trackcands_from_t5s(
    event: ak.Record,
    drop_duplicate_cleaned: bool = False,
) -> EventTrackCandidates:
    """
    The event's T5s as track candidates: their base hits, score t5_dnnScore, and t5_pt, t5_eta, t5_phi.

    By default every T5 is kept. With drop_duplicate_cleaned, T5s removed by LST's T5-T5 duplicate cleaning
    (any of T5_DUPLICATE_CLEANING_BITS set in t5_isDupBitmask) are dropped.

    Raises TypeError if event is not one event, and ValueError if a kept T5 has fewer than N_T5_BASE_HITS hits.
    """
    if not isinstance(event, ak.Record):
        raise TypeError(
            f"trackcands_from_t5s takes one event (an ak.Record, e.g. events[i]), got {type(event).__name__}"
        )
    is_kept = np.ones(len(event["t5_pt"]), dtype=bool)
    if drop_duplicate_cleaned:
        dup_bitmask = ak.to_numpy(event["t5_isDupBitmask"])
        is_kept = (dup_bitmask & T5_DUPLICATE_CLEANING_BITS) == 0
    hit_indices = event["t5_hitIndices"][is_kept]
    # Data-quality check: every T5 has its base hits
    n_hits = ak.num(hit_indices)
    if ak.any(n_hits < N_T5_BASE_HITS):
        raise ValueError(
            f"a T5 has fewer than N_T5_BASE_HITS={N_T5_BASE_HITS} hits: n_hits_min={ak.min(n_hits)}"
        )
    # pad_none with clip=True cuts every row to exactly N_T5_BASE_HITS (the extension hits are dropped), giving
    # a regular array; fill_none would replace padding with NO_HIT, but after the check above there is none
    base_hits = ak.pad_none(
        hit_indices,
        target=N_T5_BASE_HITS,
        clip=True,
    )
    return EventTrackCandidates(
        trackcand_hitIndices=ak.to_numpy(ak.fill_none(base_hits, NO_HIT)),
        trackcand_score=ak.to_numpy(event["t5_dnnScore"][is_kept]),
        trackcand_pt=ak.to_numpy(event["t5_pt"][is_kept]),
        trackcand_eta=ak.to_numpy(event["t5_eta"][is_kept]),
        trackcand_phi=ak.to_numpy(event["t5_phi"][is_kept]),
    )
