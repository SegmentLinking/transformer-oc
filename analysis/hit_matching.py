"""
LST-style truth matching of track candidates (clusters of T3s) from the per-hit truth of the graphs.

A cluster's hits are the distinct hits (by hit_idx) of all its T3s. As in LST's
matchedSimTrkIdxsAndFracsFromCounts, every sim track is credited with the number of the cluster's
hits that have it among their sim tracks; the best sim track has the largest count (ties to the
larger sim index, as in src/utils/hit_targets.py), and its fraction is count / number of hits.
A cluster is matched to its best sim track if that fraction is > the purity threshold (LST: 0.75).
"""
import numpy as np

from constants import HIT_UNMATCHED


def match_clusters(labels, hit_idx, cand, hit_layer):
    """
    Inputs, for one event of N T3s with H hits each:
      labels (N,): cluster label of every T3, < 0 for T3s in no cluster
      hit_idx (N, H): event hit index of every hit of every T3
      cand (N, H, K): sim tracks of every hit, padded with HIT_UNMATCHED (hit_targets.hit_candidates)
      hit_layer (N, H): layer of every hit
    Output: dict of arrays with one entry per cluster, sorted by label:
      label, n_t3, n_hits (distinct hits), n_layers (distinct hit layers),
      sim (best sim track, -1 if no hit is matched), n_matched_hits (its count), frac (= n_matched_hits / n_hits)
    """
    labels = np.asarray(labels)
    hit_idx = np.asarray(hit_idx)
    cand = np.asarray(cand)
    hit_layer = np.asarray(hit_layer)
    n_hits_per_t3, width = cand.shape[1], cand.shape[2]

    sel = labels >= 0
    clusters, n_t3 = np.unique(labels[sel], return_counts=True)
    empty = dict(label=clusters, n_t3=n_t3, n_hits=np.zeros(0, np.int64), n_layers=np.zeros(0, np.int64),
                 sim=np.zeros(0, np.int64), n_matched_hits=np.zeros(0, np.int64), frac=np.zeros(0))
    if len(clusters) == 0:
        return empty

    lab = np.repeat(labels[sel], n_hits_per_t3).astype(np.int64)
    hit = hit_idx[sel].reshape(-1).astype(np.int64)
    hit_cand = cand[sel].reshape(-1, width)
    layer = hit_layer[sel].reshape(-1).astype(np.int64)

    # distinct (cluster, hit) pairs
    pairs, first = np.unique(np.stack([lab, hit], axis=1), axis=0, return_index=True)
    pair_lab = pairs[:, 0]
    _, n_hits = np.unique(pair_lab, return_counts=True)
    _, n_layers = np.unique(np.unique(np.stack([pair_lab, layer[first]], axis=1), axis=0)[:, 0], return_counts=True)

    # (cluster, sim) for every sim of every distinct hit; the sims of one hit are distinct, so the
    # number of (cluster, sim) rows is the number of the cluster's hits with that sim
    sims = hit_cand[first].reshape(-1)
    sim_lab = np.repeat(pair_lab, width)
    ok = sims != HIT_UNMATCHED
    best_sim = np.full(len(clusters), -1, dtype=np.int64)
    best_count = np.zeros(len(clusters), dtype=np.int64)
    if ok.any():
        cs, counts = np.unique(np.stack([sim_lab[ok], sims[ok]], axis=1), axis=0, return_counts=True)
        order = np.lexsort((cs[:, 1], counts, cs[:, 0]))  # by cluster, then count, then sim
        cs, counts = cs[order], counts[order]
        last = np.r_[cs[1:, 0] != cs[:-1, 0], True]      # best (count, sim) of each cluster
        where = np.searchsorted(clusters, cs[last, 0])
        best_sim[where] = cs[last, 1]
        best_count[where] = counts[last]

    return dict(label=clusters, n_t3=n_t3, n_hits=n_hits, n_layers=n_layers,
                sim=best_sim, n_matched_hits=best_count, frac=best_count / n_hits)


def select_sims(sim_features, pt_min=0.8, eta_max=2.4, vz_max=30.0, vxy_max=2.5):
    """
    Mask of the sim tracks in the efficiency denominator: charged, pT > pt_min, |eta| < eta_max,
    produced near the beam spot. sim_features columns: SIM_VARS of transformer-datasets
    (pt, eta, phi, q, vx, vy, vz, vtxperp). Row i is sim track i: the graphs' sim indices below
    len(sim_features) are the signal event's tracks, larger ones are pileup.
    """
    s = np.asarray(sim_features)
    pt, eta, q, vx, vy, vz = s[:, 0], s[:, 1], s[:, 3], s[:, 4], s[:, 5], s[:, 6]
    return (q != 0) & (pt > pt_min) & (np.abs(eta) < eta_max) & (np.abs(vz) < vz_max) & (np.hypot(vx, vy) < vxy_max)
