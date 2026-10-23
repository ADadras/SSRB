from __future__ import annotations
from typing import Any
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

def pair_cost(mask_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    expanded = target[:, None].expand_as(mask_logits)
    bce = F.binary_cross_entropy_with_logits(
        mask_logits, expanded, reduction="none"
    ).flatten(2).mean(dim=2)
    probability = mask_logits.sigmoid()
    intersection = (probability * expanded).flatten(2).sum(dim=2)
    denominator = probability.flatten(2).sum(dim=2) + expanded.flatten(2).sum(dim=2)
    dice_loss = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    return bce + dice_loss


def pair_diversity(mask_logits: torch.Tensor) -> torch.Tensor:
    probability = mask_logits.sigmoid().flatten(2)
    if probability.shape[1] < 2:
        return probability.new_tensor(0.0)
    numerator = 2.0 * torch.einsum("bkp,blp->bkl", probability, probability) + 1.0
    mass = probability.sum(dim=2)
    denominator = mass[:, :, None] + mass[:, None, :] + 1.0
    overlap = numerator / denominator
    triangle = torch.triu(
        torch.ones_like(overlap, dtype=torch.bool), diagonal=1
    )
    return overlap[triangle].mean()


def assignment_weights(costs: torch.Tensor, *, method: str = "capacity_wta", capacity_offset: int = 0):
    if costs.ndim != 2 or min(costs.shape) < 1 or not torch.isfinite(costs).all():
        raise ValueError("assignment requires finite nonempty B,K costs")
    batch, queries = costs.shape
    if method == "hard_mcl":
        winners = costs.argmin(dim=1)
        return F.one_hot(winners, num_classes=queries).float(), winners
    if method == "capacity_wta":
        # Expand every head into floor(B/K) or ceil(B/K) capacity slots, then
        # solve the minimum-cost batch assignment.  The remainder rotates over
        # heads between batches so a short final batch has no fixed-head bias.
        base, remainder = divmod(int(batch), int(queries))
        offset = int(capacity_offset) % int(queries)
        capacities = torch.full((queries,), base, dtype=torch.long)
        if remainder:
            indices = (torch.arange(remainder) + offset) % queries
            capacities[indices] += 1
        slot_heads = torch.repeat_interleave(torch.arange(queries), capacities)
        if int(slot_heads.numel()) != int(batch):
            raise AssertionError("capacity slots do not sum to the batch size")
        expanded = costs.detach().cpu().double()[:, slot_heads]
        row_indices, slot_indices = linear_sum_assignment(expanded.numpy())
        winners = torch.empty(batch, dtype=torch.long)
        winners[torch.as_tensor(row_indices, dtype=torch.long)] = slot_heads[
            torch.as_tensor(slot_indices, dtype=torch.long)
        ]
        winners = winners.to(costs.device)
        return F.one_hot(winners, num_classes=queries).float(), winners
    raise ValueError(f"unsupported assignment {method}")


def objective(mask_logits, rank_logits, target, *, training=True, capacity_offset=0):
    costs = pair_cost(mask_logits, target)
    weights, winners = assignment_weights(costs, method="capacity_wta" if training else "hard_mcl", capacity_offset=capacity_offset)
    mask_loss = (weights.detach() * costs).sum(dim=1).mean()
    rank_loss = F.binary_cross_entropy_with_logits(rank_logits, weights.detach())
    diversity = pair_diversity(mask_logits)
    loss = mask_loss + .30 * rank_loss + 0.0 * diversity
    return loss, {"mask_loss": float(mask_loss.detach()), "rank_loss": float(rank_loss.detach()), "winners": winners.detach()}
