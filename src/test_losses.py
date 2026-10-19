"""
test_losses.py - tests for the contrastive loss in losses.py

You can run me like:
> cd src && apptainer exec /ceph/users/atuna/ml-tracking.sif python3 test_losses.py
"""
import math

import torch

from losses import contrastive_loss, contrastive_loss_one_event

MARGIN = 1.0


def loss(coords, sim_index, batch=None):
    coords = torch.tensor(coords, dtype=torch.float32)
    sim_index = torch.tensor(sim_index)
    if batch is None:
        batch = torch.zeros(len(sim_index), dtype=torch.long)
    else:
        batch = torch.tensor(batch)
    attractive, repulsive = contrastive_loss(
        coords=coords,
        sim_index=sim_index,
        batch=batch,
        margin=MARGIN,
    )
    return attractive.item(), repulsive.item()


def test_positive_pair_is_distance_squared():
    attractive, repulsive = loss(coords=[[0.0, 0.0], [3.0, 4.0]], sim_index=[7, 7])
    assert math.isclose(attractive, 25.0)
    assert repulsive == 0.0


def test_unmatched_t3s_never_form_positive_pairs():
    # Both have sim index -1, but that does not make them a positive pair
    attractive, repulsive = loss(coords=[[0.0, 0.0], [0.5, 0.0]], sim_index=[-1, -1])
    assert attractive == 0.0
    assert math.isclose(repulsive, (MARGIN - 0.5) ** 2)


def test_negative_pair_beyond_margin_is_zero():
    _, repulsive = loss(coords=[[0.0, 0.0], [1.5, 0.0]], sim_index=[1, 2])
    assert repulsive == 0.0


def test_negative_pair_at_margin_is_zero():
    _, repulsive = loss(coords=[[0.0, 0.0], [MARGIN, 0.0]], sim_index=[1, 2])
    assert repulsive == 0.0


def test_terms_are_averaged_over_their_own_pairs():
    # A and B share a sim index (distance 0.4).
    # C has another sim index, at 0.3 from A and 0.5 from B.
    coords = [[0.0, 0.0], [0.4, 0.0], [0.0, 0.3]]
    attractive, repulsive = loss(coords=coords, sim_index=[5, 5, 6])
    assert math.isclose(attractive, 0.4**2, rel_tol=1e-6)
    expected_repulsive = ((MARGIN - 0.3) ** 2 + (MARGIN - 0.5) ** 2) / 2
    assert math.isclose(repulsive, expected_repulsive, rel_tol=1e-6)


def test_events_are_averaged():
    # Event 0: a positive pair at distance 2. Event 1: a positive pair at distance 4.
    # Sim index 3 appears in both events, but pairs never cross events.
    coords = [[0.0, 0.0], [2.0, 0.0], [10.0, 0.0], [14.0, 0.0]]
    sim_index = [3, 3, 3, 3]
    attractive, repulsive = loss(coords=coords, sim_index=sim_index, batch=[0, 0, 1, 1])
    assert math.isclose(attractive, (2.0**2 + 4.0**2) / 2)
    assert repulsive == 0.0


def test_gradient_is_finite_for_identical_points():
    # A positive pair at distance 0 must not give NaN gradients
    coords = torch.zeros(2, 3, requires_grad=True)
    attractive, repulsive = contrastive_loss(
        coords=coords,
        sim_index=torch.tensor([1, 1]),
        batch=torch.zeros(2, dtype=torch.long),
        margin=MARGIN,
    )
    (attractive + repulsive).backward()
    assert torch.isfinite(coords.grad).all()


def test_gradients_pull_positives_together_and_push_negatives_apart():
    coords = torch.tensor([[0.0, 0.0], [0.5, 0.0]], requires_grad=True)
    for sim_index, direction in [([1, 1], "together"), ([1, 2], "apart")]:
        coords.grad = None
        attractive, repulsive = contrastive_loss(
            coords=coords,
            sim_index=torch.tensor(sim_index),
            batch=torch.zeros(2, dtype=torch.long),
            margin=MARGIN,
        )
        (attractive + repulsive).backward()
        # A step against the gradient moves the second point along +x if it moves apart
        step_x = -coords.grad[1, 0].item()
        assert (step_x < 0) if direction == "together" else (step_x > 0)


def test_row_blocks_match_pair_by_pair_loop():
    # 11 T3s: sim indices with repeats, plus unmatched ones
    generator = torch.Generator().manual_seed(0)
    coords = torch.rand(11, 3, generator=generator)
    sim_index = torch.tensor([4, 4, 4, -1, 7, 7, -1, 9, 2, 2, -1])

    # Reference: plain loop over each pair once
    positive, negative = [], []
    for i in range(len(sim_index)):
        for j in range(i + 1, len(sim_index)):
            d = torch.dist(coords[i], coords[j])
            if sim_index[i] >= 0 and sim_index[i] == sim_index[j]:
                positive.append(d**2)
            else:
                negative.append(torch.relu(MARGIN - d) ** 2)
    expected = (sum(positive) / len(positive), sum(negative) / len(negative))

    gradients = []
    for block_size in [1, 3, 4, 1024]:
        coords_grad = coords.clone().requires_grad_(True)
        attractive, repulsive = contrastive_loss_one_event(
            coords=coords_grad,
            sim_index=sim_index,
            margin=MARGIN,
            block_size=block_size,
        )
        assert math.isclose(attractive.item(), expected[0].item(), rel_tol=1e-5)
        assert math.isclose(repulsive.item(), expected[1].item(), rel_tol=1e-5)
        (attractive + repulsive).backward()
        gradients.append(coords_grad.grad)
    for gradient in gradients[1:]:
        assert torch.allclose(gradient, gradients[0], atol=1e-6)


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_"):
            func()
            print(f"{name}: passed")
