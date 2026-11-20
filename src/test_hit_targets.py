"""
test_hit_targets.py - tests for the hit-based training targets in utils/hit_targets.py

You can run me like:
> cd src && apptainer exec /ceph/users/atuna/ml-tracking.sif python3 test_hit_targets.py
"""
import torch
from torch_geometric.data import Data

from constants import HIT_UNMATCHED
from utils.hit_targets import hit_candidates, hit_targets as _hit_targets

U = HIT_UNMATCHED


def _table(rows):
    """A table of candidates per distinct hit, from a list of candidate lists, padded with HIT_UNMATCHED."""
    width = max(1, max(len(r) for r in rows))
    return torch.tensor([r + [U] * (width - len(r)) for r in rows], dtype=torch.int64)


def hit_targets(hit_id, table, min_matched_hits):
    """hit_targets on the (N, H, K) candidates of T3s whose hits are rows hit_id of the table."""
    return _hit_targets(table[hit_id], min_matched_hits=min_matched_hits)


def test_hit_candidates_pads_and_stacks_slots():
    # two T3s, three hit slots of different widths
    graph = Data(
        x=torch.zeros(2, 1),
        hit_0_simIdxs=torch.tensor([[7], [U]]),
        hit_2_simIdxs=torch.tensor([[5, 3, U], [U, U, U]]),
        hit_1_simIdxs=torch.tensor([[7, 2], [4, U]]),
    )
    cand = hit_candidates(graph)
    assert cand.shape == (2, 3, 3)
    assert cand[0].tolist() == [[7, U, U], [7, 2, U], [5, 3, U]]
    assert cand[1].tolist() == [[U, U, U], [4, U, U], [U, U, U]]
    target, frac = _hit_targets(cand, min_matched_hits=1)
    assert target.tolist() == [7, 4]
    assert torch.allclose(frac, torch.tensor([2 / 3, 1 / 3]))


def test_hit_candidates_missing_slots_raises():
    try:
        hit_candidates(Data(x=torch.zeros(1, 1), hit_1_simIdxs=torch.tensor([[1]])))
    except ValueError:
        return
    raise AssertionError("hit_candidates did not raise ValueError for slots not numbered from 0")


def test_counts_and_threshold():
    # hits 0-5: five on sim 7, one unmatched; hits 6-11: four on sim 3, two on sim 8
    cand = _table([[7], [7], [7], [7], [7], [], [3], [3], [3], [3], [8], [8]])
    hit_id = torch.tensor([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])
    target, frac = hit_targets(hit_id, cand, min_matched_hits=5)
    assert target.tolist() == [7, -1]
    assert torch.allclose(frac, torch.tensor([5 / 6, 4 / 6]))
    target, _ = hit_targets(hit_id, cand, min_matched_hits=4)
    assert target.tolist() == [7, 3]


def test_hit_counts_for_every_candidate():
    # every hit has sim 5; hits 0-2 also have sim 9: sim 5 is on all six hits
    cand = _table([[9, 5], [9, 5], [9, 5], [5], [5], [5]])
    target, frac = hit_targets(torch.arange(6).view(1, 6), cand, min_matched_hits=5)
    assert target.tolist() == [5] and frac.item() == 1.0


def test_shared_hits():
    # two T3s sharing hits 4 and 5 of the table
    cand = _table([[1], [1], [1], [1], [1], [1], [2], [2], [2], [2]])
    hit_id = torch.tensor([[0, 1, 2, 3, 4, 5], [4, 5, 6, 7, 8, 9]])
    target, frac = hit_targets(hit_id, cand, min_matched_hits=4)
    assert target.tolist() == [1, 2]
    assert torch.allclose(frac, torch.tensor([1.0, 4 / 6]))


def test_tie_goes_to_larger_sim_index():
    cand = _table([[4], [4], [4], [6], [6], [6]])
    target, frac = hit_targets(torch.arange(6).view(1, 6), cand, min_matched_hits=3)
    assert target.tolist() == [6] and abs(frac.item() - 0.5) < 1e-6


def test_all_unmatched_is_noise():
    cand = _table([[]] * 6)
    target, frac = hit_targets(torch.arange(6).view(1, 6), cand, min_matched_hits=1)
    assert target.tolist() == [-1] and frac.item() == 0.0


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_"):
            func()
            print(f"{name}: passed")
