"""
One event's track candidates. Fields are stored as per-event arrays.

This file includes two methods for building track candidates from LSTNtuple:
- LST T5s
- MDs matched to the same sim track, as a ground-truth check

MD-based ML models are responsible for building their own track candidates.
"""

from dataclasses import dataclass

import awkward as ak
import numpy as np

# The hits of the 5 base MDs; t5_hitIndices appends the hits of extension layers after them
N_T5_BASE_HITS = 10
# t5_isDupBitmask bits set when LST's T5-T5 duplicate cleaning removes a T5, in its first or second pass
# (Kernels.h, rmQuintupletFromMemory). Bit 4 (cross-cleaning against pT5/pT3) is not included.
T5_DUPLICATE_CLEANING_BITS = 1 | 2
# Pads trackcand_hitIndices where a track candidate has fewer hits than the longest; test with hit < 0
NO_HIT = -999
# Score of every ground-truth track candidate
TRUTH_SCORE = 1.0


@dataclass
class EventTrackCandidates:
    """
    One event's track candidates: entry i of every field belongs to track candidate i.

    trackcand_hitIndices has shape (n_trackcands, n_hits_max), padded with NO_HIT; the others (n_trackcands,).
    trackcand_simIdx is the matched sim, < 0 for none, from the track candidate's own source: LST's t5_simIdx for
    T5s, and the sim it was built from for the ground truth.
    """

    trackcand_hitIndices: np.ndarray
    trackcand_simIdx: np.ndarray
    trackcand_score: np.ndarray
    trackcand_pt: np.ndarray
    trackcand_eta: np.ndarray
    trackcand_phi: np.ndarray


def trackcands_from_t5s(
    event: ak.Record,
    drop_duplicate_cleaned: bool = False,
) -> EventTrackCandidates:
    """
    The event's T5s as track candidates: their base hits, sim t5_simIdx, score t5_dnnScore, and t5_pt, t5_eta,
    t5_phi.

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
        trackcand_simIdx=ak.to_numpy(event["t5_simIdx"][is_kept]),
        trackcand_score=ak.to_numpy(event["t5_dnnScore"][is_kept]),
        trackcand_pt=ak.to_numpy(event["t5_pt"][is_kept]),
        trackcand_eta=ak.to_numpy(event["t5_eta"][is_kept]),
        trackcand_phi=ak.to_numpy(event["t5_phi"][is_kept]),
    )


def trackcands_from_mds_matched_to_sim(
    event: ak.Record,
    min_layers: int = 0,
) -> EventTrackCandidates:
    """
    Ground truth: one track candidate per stored sim, made of the hits of the OT MDs matched to the sim
    (md_simIdx: both hits from that sim). MDs with hits from different sims are left out, though a model could
    recover them at a matching fraction below 1. pT, eta and phi are the sim's; score TRUTH_SCORE.

    Sims with no such MD, or whose MDs span fewer than min_layers unique layers (md_layer), get no track
    candidate. Track candidates are in increasing sim index; hits are sorted and padded with NO_HIT.
    """
    n_sims = len(event["sim_pt"])
    is_ot = ~ak.to_numpy(event["md_isPLS"])
    md_simIdx = ak.to_numpy(event["md_simIdx"])[is_ot].tolist()
    md_layer = ak.to_numpy(event["md_layer"])[is_ot].tolist()
    md_anchor_hitIdx = ak.to_numpy(event["md_anchor_hitIdx"])[is_ot].tolist()
    md_other_hitIdx = ak.to_numpy(event["md_other_hitIdx"])[is_ot].tolist()
    hits_of_sim = {}
    layers_of_sim = {}
    for sim, layer, anchor_hit, other_hit in zip(
        md_simIdx,
        md_layer,
        md_anchor_hitIdx,
        md_other_hitIdx,
    ):
        # Matched to no sim (< 0), or to a pileup sim (no sim_* entries)
        if sim < 0 or sim >= n_sims:
            continue
        if sim not in hits_of_sim:
            hits_of_sim[sim] = set()
            layers_of_sim[sim] = set()
        hits_of_sim[sim].update([anchor_hit, other_hit])
        layers_of_sim[sim].add(layer)
    kept_sims = [
        sim for sim in sorted(hits_of_sim) if len(layers_of_sim[sim]) >= min_layers
    ]
    kept_hits = [sorted(hits_of_sim[sim]) for sim in kept_sims]
    # default=0: an event with no track candidates gives shape (0, 0)
    n_hits_max = max([len(hits) for hits in kept_hits], default=0)
    hit_indices = np.full((len(kept_hits), n_hits_max), NO_HIT, dtype=int)
    for i_trackcand, hits in enumerate(kept_hits):
        hit_indices[i_trackcand, : len(hits)] = hits
    kept_sims = np.array(kept_sims, dtype=int)
    return EventTrackCandidates(
        trackcand_hitIndices=hit_indices,
        trackcand_simIdx=kept_sims,
        trackcand_score=np.full(len(kept_sims), TRUTH_SCORE),
        trackcand_pt=ak.to_numpy(event["sim_pt"])[kept_sims],
        trackcand_eta=ak.to_numpy(event["sim_eta"])[kept_sims],
        trackcand_phi=ak.to_numpy(event["sim_phi"])[kept_sims],
    )
