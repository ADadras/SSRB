"""Original-grid mask-bank metrics used by the accepted SSRB experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F


DEFAULT_PREFIXES = (1, 2, 4, 8)
EVALUATION_DICE_SMOOTH = 1.0
WASTE_DICE_THRESHOLD = 0.30
ACTIVE_DICE_THRESHOLD = 0.50
UNIQUE_DICE_THRESHOLD = 0.95


def as_path_list(value: Any) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            return [str(item) for item in json.loads(stripped)]
        return [value]
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise TypeError(f"cannot interpret mask paths from {type(value).__name__}")


def load_binary_mask(path: str | Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.float32)
    if array.max(initial=0.0) > 1.0:
        array = array / 255.0
    return torch.from_numpy(array > 0.5)


def load_same_grid_masks(paths: Iterable[str | Path]) -> torch.Tensor:
    masks = [load_binary_mask(path) for path in paths]
    if not masks:
        raise ValueError("at least one reference mask is required")
    shapes = {tuple(mask.shape) for mask in masks}
    if len(shapes) != 1:
        raise ValueError(f"reference masks do not share an annotation grid: {shapes}")
    return torch.stack(masks, dim=0)


def upsample_probability(
    probability: torch.Tensor, output_hw: tuple[int, int]
) -> torch.Tensor:
    if probability.ndim != 3:
        raise ValueError("probability bank must have shape K,H,W")
    return F.interpolate(
        probability[None].float(),
        size=(int(output_hw[0]), int(output_hw[1])),
        mode="bilinear",
        align_corners=False,
    )[0]


def threshold_probability(probability: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return probability >= float(threshold)


def dice_matrix(
    predictions: torch.Tensor,
    references: torch.Tensor,
    *,
    smooth: float = EVALUATION_DICE_SMOOTH,
) -> torch.Tensor:
    prediction = predictions.bool().flatten(1).float()
    reference = references.bool().flatten(1).float()
    intersection = torch.einsum("kp,tp->kt", prediction, reference)
    denominator = prediction.sum(dim=1)[:, None] + reference.sum(dim=1)[None, :]
    return (2.0 * intersection + float(smooth)) / (denominator + float(smooth))


def pairwise_dice(
    predictions: torch.Tensor, *, smooth: float = EVALUATION_DICE_SMOOTH
) -> torch.Tensor:
    matrix = dice_matrix(predictions, predictions, smooth=smooth)
    keep = torch.triu(torch.ones_like(matrix, dtype=torch.bool), diagonal=1)
    return matrix[keep]


def containment(predictions: torch.Tensor) -> float:
    flat = predictions.bool().flatten(1).float()
    values: list[float] = []
    for left in range(flat.shape[0]):
        for right in range(left + 1, flat.shape[0]):
            intersection = float((flat[left] * flat[right]).sum())
            minimum = min(float(flat[left].sum()), float(flat[right].sum()))
            values.append(intersection / max(1.0, minimum))
    return float(np.mean(values)) if values else 1.0


def unique_clusters(predictions: torch.Tensor, threshold: float = 0.95) -> int:
    """Count connected components under a high-Dice duplicate relation."""

    count = int(predictions.shape[0])
    if count <= 1:
        return count
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    pair = dice_matrix(predictions, predictions)
    for left in range(count):
        for right in range(left + 1, count):
            if float(pair[left, right]) >= float(threshold):
                union(left, right)
    return len({find(index) for index in range(count)})


def encode_binary_rle(mask: torch.Tensor) -> dict[str, Any]:
    """Encode a binary mask as lossless zero-based row-major start/length runs."""

    array = mask.detach().cpu().bool().numpy().reshape(-1).astype(np.uint8)
    padded = np.pad(array, (1, 1), constant_values=0)
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    starts = changes[0::2]
    ends = changes[1::2]
    return {
        "shape": [int(mask.shape[-2]), int(mask.shape[-1])],
        "starts": starts.astype(np.int64).tolist(),
        "lengths": (ends - starts).astype(np.int64).tolist(),
        "order": "C",
        "index_base": 0,
    }


def decode_binary_rle(payload: dict[str, Any]) -> torch.Tensor:
    height, width = (int(value) for value in payload["shape"])
    flat = np.zeros(height * width, dtype=np.bool_)
    for start, length in zip(payload["starts"], payload["lengths"], strict=True):
        flat[int(start) : int(start) + int(length)] = True
    return torch.from_numpy(flat.reshape(height, width))


def image_metrics(
    predictions: torch.Tensor,
    references: torch.Tensor,
    *,
    max_k: int = 8,
) -> dict[str, float]:
    if predictions.ndim != 3 or references.ndim != 3:
        raise ValueError("predictions and references must have shape N,H,W")
    if tuple(predictions.shape[-2:]) != tuple(references.shape[-2:]):
        raise ValueError("predictions and references must share an evaluation grid")
    effective_k = min(int(max_k), int(predictions.shape[0]))
    if effective_k < 1:
        raise ValueError("at least one prediction is required")
    prediction = predictions[:effective_k].bool()
    dice = dice_matrix(prediction, references)
    prefixes: list[float] = []
    output: dict[str, float] = {}
    for prefix in range(1, effective_k + 1):
        quality = dice[:prefix].max(dim=0).values
        value = float(quality.mean())
        prefixes.append(value)
        output[f"MLQ@{prefix}"] = value
        output[f"Worst@{prefix}"] = float(quality.min())
    for requested in DEFAULT_PREFIXES:
        if requested > effective_k:
            output[f"MLQ@{requested}"] = float("nan")
            output[f"Worst@{requested}"] = float("nan")
    prediction_best = dice.max(dim=1).values
    pair = pairwise_dice(prediction)
    output.update(
        {
            "AUC": float(np.mean(prefixes)),
            f"Waste@{effective_k}": float(
                (prediction_best < WASTE_DICE_THRESHOLD).float().mean()
            ),
            f"PairDice@{effective_k}": float(pair.mean()) if pair.numel() else 0.0,
            "Containment": containment(prediction),
            "Unique95": float(unique_clusters(prediction, UNIQUE_DICE_THRESHOLD)),
            "active_head_count": float(
                (prediction_best >= ACTIVE_DICE_THRESHOLD).sum()
            ),
            "effective_k": float(effective_k),
        }
    )
    for prefix in (2, 4, 8):
        if prefix <= effective_k:
            output[f"Gain@{prefix}"] = output[f"MLQ@{prefix}"] - output[
                f"MLQ@{prefix // 2}"
            ]
        else:
            output[f"Gain@{prefix}"] = float("nan")
    return output


def macro_average(rows: list[dict[str, Any]], value_keys: Iterable[str]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key in value_keys:
        values = np.asarray([float(row[key]) for row in rows if key in row], dtype=float)
        finite = values[np.isfinite(values)]
        output[key] = float(finite.mean()) if len(finite) else float("nan")
    return output


def iou_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_flat = left.bool().flatten(1).float()
    right_flat = right.bool().flatten(1).float()
    intersection = torch.einsum("ip,jp->ij", left_flat, right_flat)
    union = left_flat.sum(dim=1)[:, None] + right_flat.sum(dim=1)[None, :] - intersection
    return 1.0 - (intersection + 1.0) / (union + 1.0)

def ged2(predictions: torch.Tensor, references: torch.Tensor) -> float:
    return float(
        2.0 * iou_distance(predictions, references).mean()
        - iou_distance(predictions, predictions).mean()
        - iou_distance(references, references).mean()
    )
