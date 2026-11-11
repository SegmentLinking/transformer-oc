"""
Read hit-level truth from an LSTNtuple with the per-hit MD branches (md_{anchor,other}_{hitIdx,simIdxs}).
"""

import awkward as ak
import numpy as np

MD_HIT_COLUMNS = [
    "md_anchor_hitIdx",
    "md_other_hitIdx",
    "md_anchor_simIdxs",
    "md_other_simIdxs",
]


def hit_sim_pairs(event: ak.Record) -> np.ndarray:
    """
    Unique [hit index, sim index] pairs over the hits of all MDs of one event, shape (n_pairs, 2), sorted.

    Hit indices are LST hit indices, the same as in t5_hitIndices. A hit with no sim gives no pair.
    """
    hits = []
    sims = []
    for prefix in ["md_anchor", "md_other"]:
        hit_sims = event[f"{prefix}_simIdxs"]
        hit_index = ak.to_numpy(event[f"{prefix}_hitIdx"])
        n_sims = ak.to_numpy(ak.num(hit_sims))
        hits.append(np.repeat(hit_index, n_sims))
        sims.append(ak.to_numpy(ak.flatten(hit_sims)))
    pairs = np.stack([np.concatenate(hits), np.concatenate(sims)], axis=1)
    return np.unique(pairs.astype(np.int64), axis=0)
