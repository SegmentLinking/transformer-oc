"""
Hit-level sim matching of track candidates, following LST's matchedSimTrkIdxsAndFracs (trkCore.cc).

A candidate is a collection of LST hit indices. Each of its unique hits counts once for every sim it carries.
A hit with several sims counts for each of them, which equals LST's maximum over the hit-sim permutations.
The candidate matches its best sim if that sim's share of the unique hits is above MATCH_FRACTION
(strict, as in LST).
"""

from collections import Counter

import numpy as np

MATCH_FRACTION = 0.75
UNMATCHED = -999


def hit_sims_from_pairs(pairs: np.ndarray) -> dict[int, list[int]]:
    """
    Map each hit index to the list of its sims.

    Input: pairs of shape (n_pairs, 2) with columns [hit index, sim index], one row per (hit, sim).
    Hits with no sim have no row, and so no entry.
    """
    hit_sims = {}
    for hit, sim in pairs.tolist():
        hit_sims.setdefault(hit, []).append(sim)
    return hit_sims


def best_sim(
    hits: list[int],
    hit_sims: dict[int, list[int]],
) -> tuple[int, float]:
    """
    The sim with the most hits of a candidate, and its share of the candidate's unique hits.

    Returns (UNMATCHED, 0.0) if no hit has a sim. Ties go to the smaller sim index.
    """
    unique_hits = set(hits)
    counts = Counter()
    for hit in unique_hits:
        counts.update(hit_sims.get(hit, []))
    if not counts:
        return UNMATCHED, 0.0
    sim, n_hits = max(sorted(counts.items()), key=lambda item: item[1])
    return sim, n_hits / len(unique_hits)


def matched_sim(
    hits: list[int],
    hit_sims: dict[int, list[int]],
    match_fraction: float = MATCH_FRACTION,
) -> int:
    """The sim a candidate is matched to, or UNMATCHED if its best sim has match_fraction of the hits or less."""
    sim, fraction = best_sim(hits=hits, hit_sims=hit_sims)
    return sim if fraction > match_fraction else UNMATCHED
