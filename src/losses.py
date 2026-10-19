import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def contrastive_loss(coords, sim_index, batch, margin):
    """
    Contrastive loss of Hadsell, Chopra, and LeCun (2006) over all pairs of T3s in an
    event.

    A pair is positive if both T3s share a sim index >= 0, and contributes d^2, where d
    is the distance between their embeddings. Every other pair is negative, and
    contributes max(0, margin - d)^2. T3s with sim index < 0 are therefore never in a
    positive pair.

    Positive and negative pairs are very imbalanced, so each term is averaged over its
    own pairs within an event, then over the events in the batch.

    Returns (attractive, repulsive) loss tensors.
    """
    attractive = []
    repulsive = []
    for event in torch.unique(batch):
        in_event = batch == event
        attractive_event, repulsive_event = contrastive_loss_one_event(
            coords=coords[in_event],
            sim_index=sim_index[in_event],
            margin=margin,
        )
        attractive.append(attractive_event)
        repulsive.append(repulsive_event)
    return torch.stack(attractive).mean(), torch.stack(repulsive).mean()


def contrastive_loss_one_event(coords, sim_index, margin, block_size=1024):
    n = coords.shape[0]
    # Count pairs without an n x n matrix: each sim index shared by c T3s gives
    # c(c-1)/2 positive pairs, and every other pair is negative.
    _, counts = torch.unique(sim_index[sim_index >= 0], return_counts=True)
    n_positive = (counts * (counts - 1) // 2).sum().item()
    n_negative = n * (n - 1) // 2 - n_positive

    # Sum over blocks of rows. Checkpointing frees each block's distances after the
    # forward pass and recomputes them in the backward pass, so memory goes as
    # block_size * n instead of n * n.
    attractive = coords.sum() * 0  # zero which keeps the graph
    repulsive = coords.sum() * 0
    for start in range(0, n, block_size):
        attractive_block, repulsive_block = checkpoint(
            block_sums,
            coords,
            sim_index,
            start,
            min(start + block_size, n),
            margin,
            use_reentrant=False,
        )
        attractive = attractive + attractive_block
        repulsive = repulsive + repulsive_block

    if n_positive > 0:
        attractive = attractive / n_positive
    if n_negative > 0:
        repulsive = repulsive / n_negative
    return attractive, repulsive


def block_sums(coords, sim_index, start, end, margin):
    """Sums of d^2 over positive pairs and max(0, margin - d)^2 over negative pairs,
    for pairs (i, j) with start <= i < end and j > i."""
    rows = torch.arange(start, end, device=coords.device)
    columns = torch.arange(coords.shape[0], device=coords.device)
    upper = columns[None, :] > rows[:, None]
    matched = sim_index[rows] >= 0
    same = sim_index[rows][:, None] == sim_index[None, :]
    positive = same & matched[:, None] & upper
    negative = ~positive & upper

    # cdist always runs in float32, also under mixed-precision autocast
    distance = torch.cdist(coords[rows], coords)
    attractive = (distance.pow(2) * positive).sum()
    repulsive = (F.relu(margin - distance).pow(2) * negative).sum()
    return attractive, repulsive
