from constants import T3_HIT_SIMIDX, UNMATCHED
import awkward as ak
import numpy as np

from t3_processing import build_hit_table, resolve_t3, unique_t3_md_indices


def test_resolve_t3_all_same():
    hits = [[4], [4], [4], [4], [4], [4]]
    assert resolve_t3(hits) == [4, 4, 4, 4, 4, 4]


def test_resolve_t3_docstring_example():
    hits = [[7], [7], [3, 7], [3], [], [7, 3]]
    assert resolve_t3(hits) == [7, 7, 7, 3, UNMATCHED, 7]


def test_resolve_t3_single_sim_indices_unchanged():
    hits = [[5], [6], [5], [], [6], [8]]
    assert resolve_t3(hits) == [5, 6, 5, UNMATCHED, 6, 8]


def test_resolve_t3_tie_broken_by_first_appearance_for_every_hit():
    # 1 and 2 each appear in two hits; 1 appears first, so both hits take 1
    hits = [[1, 2], [2, 1]]
    assert resolve_t3(hits) == [1, 1]


def test_resolve_t3_duplicate_sim_index_in_one_hit_counted_once():
    # 4 appears twice in the first hit but in only one hit overall, so 9 (two hits) wins
    hits = [[4, 4, 9], [9]]
    assert resolve_t3(hits) == [9, 9]


def test_resolve_t3_all_unmatched():
    assert resolve_t3([[]] * 6) == [UNMATCHED] * 6


def test_unique_t3_md_indices_drops_shared_middle_md():
    # md_idx[t3, ls, md in ls]; LS0.md1 == LS1.md0 is the shared middle MD
    md_idx = np.array([[[10, 11], [11, 12]],
                       [[20, 21], [21, 22]]])
    assert unique_t3_md_indices(md_idx).tolist() == [[10, 11, 12], [20, 21, 22]]


def test_unique_t3_md_indices_rejects_unshared_middle_md():
    md_idx = np.array([[[10, 11], [13, 12]]])
    try:
        unique_t3_md_indices(md_idx)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def _one_event(per_slot_sims):
    """sim lists of shape (n_t3, 6, var) as the one-event awkward record of T3_HIT_SIMIDX fields."""
    return ak.Array([{b: [t3[i] for t3 in per_slot_sims] for i, b in enumerate(T3_HIT_SIMIDX)}])


def test_build_hit_table_shared_hits_get_one_id():
    # T3 1 shares its first two hits (one MD) with the last two hits of T3 0
    xyz = np.zeros((2, 6, 3), dtype=np.float32)
    xyz[0, :, 0] = [1, 2, 3, 4, 5, 6]
    xyz[1, :, 0] = [5, 6, 7, 8, 9, 10]
    detId = np.array([[1, 1, 2, 2, 3, 3], [3, 3, 4, 4, 5, 5]])
    layer = detId.copy()
    # sim lists as written by the ntuple: one entry per sim hit, with repeats and negatives
    sims = [[[7, 7], [7], [7, -1], [], [3, 7], [3]],
            [[3, 7], [3], [7], [7], [-1], [7, 7]]]
    hit_id, cand = build_hit_table(detId, xyz, _one_event(sims), layer)
    assert hit_id[0, 4] == hit_id[1, 0] and hit_id[0, 5] == hit_id[1, 1]
    assert len(np.unique(hit_id)) == 10
    assert cand.shape == (10, 2)
    assert cand[hit_id[0, 0]].tolist() == [7, UNMATCHED]  # repeats removed
    assert cand[hit_id[0, 2]].tolist() == [7, UNMATCHED]  # negatives removed
    assert cand[hit_id[0, 3]].tolist() == [UNMATCHED, UNMATCHED]
    assert cand[hit_id[0, 4]].tolist() == [7, 3]  # largest first


def test_build_hit_table_rejects_inconsistent_duplicates():
    xyz = np.zeros((1, 6, 3), dtype=np.float32)  # all six slots look like the same hit
    detId = np.ones((1, 6), dtype=int)
    sims = [[[1], [2], [1], [1], [1], [1]]]
    try:
        build_hit_table(detId, xyz, _one_event(sims), detId)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_"):
            func()
            print(f"{name}: passed")
