"""
Track candidates built by mdml models: one event's track candidates, and how to build them from MD clusters.
"""

from dataclasses import dataclass

import numpy as np

from harness.pmatched import matched_sim
from harness.trackcands import NO_HIT
from mdml.data import MDsFromOneEvent

# Columns of md_predKinematics: q/pT, eta, cos phi, sin phi
N_PRED_KINEMATICS = 4


@dataclass
class TrackCandidatesFromOneEvent:
    """
    One event's track candidates: entry i of every field belongs to track candidate i. A copy of the harness's
    EventTrackCandidates, with the same fields, so the harness reads it as its own.

    trackcand_hitIndices has shape (n_trackcands, n_hits_max), padded with NO_HIT; the others (n_trackcands,).
    trackcand_simIdx is the matched sim, < 0 for none.
    """

    trackcand_hitIndices: np.ndarray
    trackcand_simIdx: np.ndarray
    trackcand_score: np.ndarray
    trackcand_pt: np.ndarray
    trackcand_eta: np.ndarray
    trackcand_phi: np.ndarray


def trackcands_from_mdclusters(
    mds: MDsFromOneEvent,
    md_mdclusterIdx: np.ndarray,
    md_predProbFake: np.ndarray,
    md_predKinematics: np.ndarray,
) -> TrackCandidatesFromOneEvent:
    """
    One track candidate per MD cluster index >= 0, in increasing index:
    - trackcand_hitIndices: the sorted unique hits of its MDs (md_hitIdx), padded with NO_HIT
    - trackcand_simIdx: matched_sim (more than 75% of the unique hits) with a hit -> sims map built from its
      MDs' md_inner_simIdxs and md_outer_simIdxs (padding < 0 dropped), < 0 for none. md_simIdx is not
      enough: a hit can have several sims, and an MD with md_simIdx < 0 can have one hit of the matched sim.
    - trackcand_score: mean of 1 - p(fake) over its MDs
    - trackcand_pt: 1 / |mean predicted q/pT|; trackcand_eta: mean predicted eta;
      trackcand_phi: atan2(mean predicted sin phi, mean predicted cos phi)
    """
    md_hitIdx = mds.md_hitIdx.numpy()
    md_inner_simIdxs = mds.md_inner_simIdxs.numpy()
    md_outer_simIdxs = mds.md_outer_simIdxs.numpy()
    hit_lists, sims, scores, kinematics = [], [], [], []
    for mdcluster in np.unique(md_mdclusterIdx[md_mdclusterIdx >= 0]):
        md_indices = np.flatnonzero(md_mdclusterIdx == mdcluster)
        # Only this MD cluster's hits; a hit shared by two of its MDs is one key, so it is counted once
        hit_sims = {}
        for md in md_indices:
            inner_hit, outer_hit = md_hitIdx[md]
            hit_sims[int(inner_hit)] = [
                int(sim) for sim in md_inner_simIdxs[md] if sim >= 0
            ]
            hit_sims[int(outer_hit)] = [
                int(sim) for sim in md_outer_simIdxs[md] if sim >= 0
            ]
        hits = sorted(hit_sims)
        hit_lists.append(hits)
        sims.append(matched_sim(hits=hits, hit_sims=hit_sims))
        scores.append(np.mean(1 - md_predProbFake[md_indices]))
        kinematics.append(md_predKinematics[md_indices].mean(axis=0))
    n_hits_max = max((len(hits) for hits in hit_lists), default=0)
    trackcand_hitIndices = np.full((len(hit_lists), n_hits_max), NO_HIT)
    for trackcand, hits in enumerate(hit_lists):
        trackcand_hitIndices[trackcand, : len(hits)] = hits
    kinematics = np.array(kinematics).reshape(-1, N_PRED_KINEMATICS)
    return TrackCandidatesFromOneEvent(
        trackcand_hitIndices=trackcand_hitIndices,
        trackcand_simIdx=np.array(sims, dtype=np.int64),
        trackcand_score=np.array(scores),
        trackcand_pt=1 / np.abs(kinematics[:, 0]),
        trackcand_eta=kinematics[:, 1],
        trackcand_phi=np.arctan2(kinematics[:, 3], kinematics[:, 2]),
    )
