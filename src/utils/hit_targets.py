"""Training targets from the per-hit truth of the graphs (hit_{k}_simIdxs from transformer-datasets)."""
import re

import torch
import torch.nn.functional as F

from constants import HIT_UNMATCHED

HIT_SLOT_FIELD = re.compile(r"^hit_(\d+)_simIdxs$")


def hit_candidates(graph):
    """
    Sim track candidates of every hit of every T3, as one (N, H, K) int64 tensor.

    The graph has one field hit_{k}_simIdxs (N, K_k) per hit slot k: the distinct sim tracks of
    that hit, padded with HIT_UNMATCHED. The number of slots H and the widths K_k are read from
    the graph; slots are padded to the largest width K and stacked in slot order.
    """
    slots = sorted(
        (int(m.group(1)), key) for key in graph.keys() if (m := HIT_SLOT_FIELD.match(key))
    )
    if not slots:
        raise ValueError("Graph has no hit_{k}_simIdxs fields: it was made by an older version of transformer-datasets")
    if [k for k, _ in slots] != list(range(len(slots))):
        raise ValueError(f"Hit slot fields are not numbered 0..H-1: {[key for _, key in slots]}")
    tensors = [graph[key] for _, key in slots]
    width = max(t.shape[1] for t in tensors)
    padded = [F.pad(t, (0, width - t.shape[1]), value=HIT_UNMATCHED) for t in tensors]
    return torch.stack(padded, dim=1)


def hit_targets(cand, min_matched_hits):
    """
    Sim track and hit fraction of every T3, with LST's count rule.

    Input: cand (N, H, K) int64, the sim tracks of each of the T3's H hits, padded with
    HIT_UNMATCHED (see hit_candidates).
    Outputs: target (N,) int64 and frac (N,) float32.

    For every sim track, count the T3's hits with that sim track among their candidates
    (as matchedSimTrkIdxsAndFracsFromCounts in LST). The best sim track has the largest count,
    ties going to the larger sim index. frac = best count / H. target is the best sim track
    if its count >= min_matched_hits, otherwise -1 (noise). With H = 6 and min_matched_hits = 5,
    target equals LST's t3_simIdx (match fraction > 0.75).
    """
    n_t3, n_hits, _ = cand.shape
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
