"""
Clustering of MDs into MD clusters, each one track candidate: DBSCAN on per-MD coordinates predicted by a
network (e.g. the sim kinematics, or a learned embedding).
"""

import numpy as np
from sklearn.cluster import DBSCAN

# DBSCAN's label for MDs in no cluster (noise); also given to MDs cut as fake
NO_MDCLUSTER = -1
# A track candidate needs at least this many MDs (DBSCAN min_samples)
MIN_MDS = 3


def find_mdclusters(
    md_coordinates: np.ndarray,
    md_isKept: np.ndarray,
    eps: float,
) -> np.ndarray:
    """
    md_mdclusterIdx, shape (n_md,): each MD's MD cluster index from DBSCAN(eps, min_samples=MIN_MDS) on the kept
    MDs' md_coordinates (n_md, n_coordinates), and NO_MDCLUSTER for noise and for MDs not kept.
    """
    md_mdclusterIdx = np.full(len(md_isKept), NO_MDCLUSTER)
    if not np.any(md_isKept):
        return md_mdclusterIdx
    # DBSCAN labels noise -1, which is NO_MDCLUSTER
    md_mdclusterIdx[md_isKept] = DBSCAN(
        eps=eps,
        min_samples=MIN_MDS,
    ).fit_predict(md_coordinates[md_isKept])
    return md_mdclusterIdx
