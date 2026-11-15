"""
Match a track candidate to a sim track, following LST's matchedSimTrkIdxsAndFracs (trkCore.cc).

Each of the candidate's unique hits counts once for every sim it belongs to (a hit with two sims counts for both).
The candidate matches its best sim if that sim has more than MATCH_FRACTION of the unique hits.

Like LST, hits with no sim compete as one more "sim": if they outnumber the best sim, the fraction is 0.
"""

MATCH_FRACTION = 0.75
UNMATCHED = -999


def best_sim(
    hits: list[int],
    hit_sims: dict[int, list[int]],
) -> tuple[int, float]:
    """
    The sim with the most of the candidate's unique hits, and its fraction of them.

    hit_sims is the map from md_truth.hit_sims_from_mds. On a tie between sims, the smaller sim index wins.

    Returns (UNMATCHED, 0.0) when hits with no sim outnumber the best sim, or tie with it and come first (LST
    breaks ties by first appearance in the hit list), which includes the case where no hit has a sim.
    """
    # Unique hits in their original order, which LST's tie rule needs
    unique_hits = list(dict.fromkeys(hits))
    n_hits_of_sim = {}
    i_first_hit_of_sim = {}
    n_no_sim = 0
    i_first_no_sim = None
    for i_hit, hit in enumerate(unique_hits):
        sims = hit_sims.get(hit, [])
        if len(sims) == 0:
            if i_first_no_sim is None:
                i_first_no_sim = i_hit
            n_no_sim += 1
        for sim in sims:
            if sim not in n_hits_of_sim:
                n_hits_of_sim[sim] = 0
                i_first_hit_of_sim[sim] = i_hit
            n_hits_of_sim[sim] += 1

    best = UNMATCHED
    n_best = 0
    # In increasing sim order with a strict >, so a tie keeps the smaller sim
    for sim in sorted(n_hits_of_sim):
        if n_hits_of_sim[sim] > n_best:
            best = sim
            n_best = n_hits_of_sim[sim]

    # Hits with no sim as one more "sim", as LST
    if n_no_sim > n_best:
        return UNMATCHED, 0.0
    if n_no_sim == n_best:
        i_first_best = min(
            i_first_hit_of_sim[sim]
            for sim in n_hits_of_sim
            if n_hits_of_sim[sim] == n_best
        )
        if i_first_no_sim < i_first_best:
            return UNMATCHED, 0.0
    return best, n_best / len(unique_hits)


def matched_sim(
    hits: list[int],
    hit_sims: dict[int, list[int]],
) -> int:
    """The candidate's best sim if its fraction is above MATCH_FRACTION, else UNMATCHED."""
    sim, fraction = best_sim(hits=hits, hit_sims=hit_sims)
    if fraction > MATCH_FRACTION:
        return sim
    return UNMATCHED
