"""Training targets from the per-hit truth of the graphs (hit_id, hit_sim_candidates from data/t3_processing.py)."""
import torch

from constants import HIT_UNMATCHED


def hit_targets(hit_id, hit_sim_candidates, min_matched_hits):
    """
    Sim track and hit fraction of every T3, with LST's count rule.

    Inputs: hit_id (N, H) int64 ids of the T3's hits, rows of hit_sim_candidates (N_hits, K) int64
    (sim tracks of every distinct hit, padded with HIT_UNMATCHED).
    Outputs: target (N,) int64 and frac (N,) float32.

    For every sim track, count the T3's hits with that sim track among their candidates
    (as matchedSimTrkIdxsAndFracsFromCounts in LST). The best sim track has the largest count,
    ties going to the larger sim index. frac = best count / H. target is the best sim track
    if its count >= min_matched_hits, otherwise -1 (noise). With H = 6 and min_matched_hits = 5,
    target equals LST's t3_simIdx (match fraction > 0.75).
    """
    n_t3, n_hits = hit_id.shape
    cand = hit_sim_candidates[hit_id]                      # N, H, K
    sims = cand.reshape(n_t3, -1)                          # N, H*K: every sim track seen by the T3
    in_hit = (sims[:, :, None, None] == cand[:, None, :, :]).any(dim=-1)  # N, H*K, H
    counts = in_hit.sum(dim=-1)
    counts = torch.where(sims != HIT_UNMATCHED, counts, torch.zeros_like(counts))

    # one integer key per (count, sim), so that the max is deterministic on every device
    key = (counts.to(torch.int64) << 32) | (sims - HIT_UNMATCHED)
    best = key.max(dim=1).values
    best_count = best >> 32
    best_sim = (best & 0xFFFFFFFF) + HIT_UNMATCHED

    target = torch.where(best_count >= min_matched_hits, best_sim, torch.full_like(best_sim, -1))
    frac = best_count.to(torch.float32) / n_hits
    return target, frac
