"""
Given a container of MDs, map the hits of those MDs onto their respective sim track(s).

NB: one hit can be mapped onto multiple sim tracks.

Namely: {
  hit_i: [simtrack_a],
  hit_j: [simtrack_b],
  hit_k: [simtrack_b, simtrack_c],
  # etc
}
"""

import awkward as ak

MD_HIT_IDX = ["md_anchor_hitIdx", "md_other_hitIdx"]
MD_SIM_IDXS = ["md_anchor_simIdxs", "md_other_simIdxs"]


def hit_sims_from_mds(event: ak.Record) -> dict[int, list[int]]:
    """
    Each LST hit index in the event's OT MDs, mapped to the sims it belongs to (sorted, without repeats).

    A hit appears once even if it is in several MDs. Hits with no sim are left out. Pixel (pLS) MDs are skipped.
    """
    if not isinstance(event, ak.Record):
        raise TypeError(
            f"hit_sims_from_mds takes one event (an ak.Record, e.g. events[i]), got {type(event).__name__}"
        )
    is_ot = ~event["md_isPLS"]
    sim_sets = {}
    for hit_column, sims_column in zip(MD_HIT_IDX, MD_SIM_IDXS):
        hit_idx = ak.to_list(event[hit_column][is_ot])
        sim_idxs = ak.to_list(event[sims_column][is_ot])
        for hit, sims in zip(hit_idx, sim_idxs):
            if len(sims) == 0:
                continue
            if hit not in sim_sets:
                sim_sets[hit] = set()
            sim_sets[hit].update(sims)
    return {hit: sorted(sims) for hit, sims in sim_sets.items()}
