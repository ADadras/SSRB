"""Accepted crossed bootstrap for the last-block DINO comparison.

Scientific functions retain source SHA-256
2d33a758ef3af9ebad66d63afcf05829f9158e2b9e35fac6dffabdd2253e83ea.
"""
from __future__ import annotations
import math
from typing import Any
import numpy as np
import pandas as pd

REPLICATES = 10_000

SEED = 20_260_817

METHODS = ("partial_dino_capacity", "frozen_capacity", "ssrb")

METRICS = ("MLQ@8", "Waste@8", "MLQ@1", "AUC", "Unique95")

REGIME_ORDER = ("LIDC", "LIDC-hard", "BSDS")

CONTRASTS = {
    "A": ("partial_dino_capacity", "frozen_capacity"),
    "B": ("ssrb", "partial_dino_capacity"),
}

def average_visible_memberships(full: pd.DataFrame) -> pd.DataFrame:
    group = [
        "regime",
        "optimizer_seed",
        "source_id",
        "image_id",
        "method",
    ]
    units = full.groupby(group, as_index=False).agg(
        **{metric: (metric, "mean") for metric in METRICS},
        visible_memberships=("visible_seed", "nunique"),
    )
    keys = ["regime", "optimizer_seed", "source_id", "image_id"]
    membership = units.pivot(index=keys, columns="method", values="visible_memberships")
    if membership.isna().any().any():
        raise ValueError("paired method missing after visible-membership averaging")
    if not np.array_equal(
        membership.to_numpy(), membership.iloc[:, :1].to_numpy().repeat(3, axis=1)
    ):
        raise ValueError("methods average different visible memberships")
    full_counts = set(
        units.loc[units["regime"] != "LIDC-hard", "visible_memberships"].astype(int)
    )
    if full_counts != {3}:
        raise ValueError(
            f"full LIDC/BSDS units require three visible memberships: {full_counts}"
        )
    hard_counts = set(
        units.loc[units["regime"] == "LIDC-hard", "visible_memberships"].astype(int)
    )
    if not hard_counts or not hard_counts.issubset({1, 2, 3}):
        raise ValueError(
            f"LIDC-hard units require one to three visible memberships: {hard_counts}"
        )
    if set(units["optimizer_seed"].astype(int)) != {0, 1, 2}:
        raise ValueError("optimizer seed matrix is incomplete")
    return units

def paired_wide(units: pd.DataFrame) -> pd.DataFrame:
    keys = ["regime", "optimizer_seed", "source_id", "image_id"]
    pieces = []
    for metric in METRICS:
        pivot = units.pivot(index=keys, columns="method", values=metric)
        if set(pivot.columns) != set(METHODS) or pivot.isna().any().any():
            raise ValueError(f"pairing failed for {metric}")
        pivot.columns = [f"{metric}|{method}" for method in pivot.columns]
        pieces.append(pivot)
    return pd.concat(pieces, axis=1).reset_index()

def bootstrap_contrasts(
    paired: pd.DataFrame, *, replicates: int, seed: int
) -> dict[str, dict[str, dict[str, Any]]]:
    rng = np.random.default_rng(int(seed))
    optimizer_draws = rng.integers(0, 3, size=(int(replicates), 3))
    lidc_sources = sorted(
        paired.loc[paired["regime"] == "LIDC", "source_id"].astype(str).unique()
    )
    bsds_sources = sorted(
        paired.loc[paired["regime"] == "BSDS", "source_id"].astype(str).unique()
    )
    if not lidc_sources or not bsds_sources:
        raise ValueError("full LIDC or BSDS source universe is empty")
    source_universes = {"lidc": lidc_sources, "bsds": bsds_sources}
    source_draws_by_family = {
        family: rng.multinomial(
            len(sources),
            np.full(len(sources), 1.0 / len(sources)),
            size=int(replicates),
        )
        for family, sources in source_universes.items()
    }
    output: dict[str, dict[str, dict[str, Any]]] = {
        contrast: {} for contrast in CONTRASTS
    }
    for regime in REGIME_ORDER:
        group = paired[paired["regime"] == regime].copy()
        if group.empty:
            raise ValueError(f"empty regime: {regime}")
        family = "lidc" if regime in {"LIDC", "LIDC-hard"} else "bsds"
        # LIDC and LIDC-hard share the exact full-LIDC source multiplicities.
        # Hard-only rows contribute zero for a sampled patient without a hard
        # membership, preserving the nested-regime correlation for max-|T|.
        sources = source_universes[family]
        source_index = {source: index for index, source in enumerate(sources)}
        unknown_sources = sorted(
            set(group["source_id"].astype(str)).difference(source_index)
        )
        if unknown_sources:
            raise ValueError(
                f"{regime} contains sources outside its full-regime universe"
            )
        active_grid = (
            group.groupby(["source_id", "optimizer_seed"])
            .size()
            .unstack("optimizer_seed", fill_value=0)
            .reindex(columns=[0, 1, 2], fill_value=0)
        )
        if active_grid.empty or (active_grid.to_numpy() <= 0).any():
            raise ValueError(
                f"crossed source/optimizer grid is incomplete for {regime}"
            )
        source_draws = source_draws_by_family[family]
        for contrast, (left, right) in CONTRASTS.items():
            sums = np.zeros((len(sources), 3, len(METRICS)), dtype=float)
            counts = np.zeros((len(sources), 3), dtype=float)
            for row in group.to_dict("records"):
                s = source_index[str(row["source_id"])]
                o = int(row["optimizer_seed"])
                counts[s, o] += 1.0
                for metric_index, metric in enumerate(METRICS):
                    sums[s, o, metric_index] += float(
                        row[f"{metric}|{left}"] - row[f"{metric}|{right}"]
                    )
            if (counts.sum(axis=0) == 0).any():
                raise ValueError(f"optimizer pairing incomplete: {contrast} {regime}")
            samples = np.zeros((int(replicates), len(METRICS)), dtype=float)
            for replicate in range(int(replicates)):
                weights = source_draws[replicate].astype(float)
                numerator = np.zeros(len(METRICS), dtype=float)
                denominator = 0.0
                for optimizer_seed in optimizer_draws[replicate]:
                    numerator += np.einsum(
                        "s,sm->m", weights, sums[:, int(optimizer_seed)]
                    )
                    denominator += float(
                        np.einsum(
                            "s,s->", weights, counts[:, int(optimizer_seed)]
                        )
                    )
                if denominator <= 0:
                    raise ValueError(f"empty bootstrap replicate: {regime}")
                samples[replicate] = numerator / denominator
            point = np.asarray(
                [
                    float(
                        (
                            group[f"{metric}|{left}"]
                            - group[f"{metric}|{right}"]
                        ).mean()
                    )
                    for metric in METRICS
                ]
            )
            output[contrast][regime] = {
                "point": point,
                "samples": samples,
                "images": int(group["image_id"].nunique()),
                "sources": int(group["source_id"].nunique()),
                "paired_units": len(group),
            }
    return output

def effect_table(
    bootstrap: dict[str, dict[str, dict[str, Any]]]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for contrast, (left, right) in CONTRASTS.items():
        for regime in REGIME_ORDER:
            info = bootstrap[contrast][regime]
            for metric_index, metric in enumerate(METRICS):
                draws = info["samples"][:, metric_index]
                rows.append(
                    {
                        "contrast": contrast,
                        "left": left,
                        "right": right,
                        "regime": regime,
                        "metric": metric,
                        "effect": float(info["point"][metric_index]),
                        "ci95_low": float(np.quantile(draws, 0.025)),
                        "ci95_high": float(np.quantile(draws, 0.975)),
                        "simultaneous_ci95_low": math.nan,
                        "simultaneous_ci95_high": math.nan,
                        "images": int(info["images"]),
                        "source_clusters": int(info["sources"]),
                        "paired_units_after_visible_average": int(
                            info["paired_units"]
                        ),
                    }
                )
        for metric in ("MLQ@8", "Waste@8"):
            metric_index = METRICS.index(metric)
            matrix = np.stack(
                [
                    bootstrap[contrast][regime]["samples"][:, metric_index]
                    for regime in REGIME_ORDER
                ],
                axis=1,
            )
            point = np.asarray(
                [
                    bootstrap[contrast][regime]["point"][metric_index]
                    for regime in REGIME_ORDER
                ]
            )
            scale = matrix.std(axis=0, ddof=1)
            scale = np.where(scale > 1e-12, scale, 1.0)
            max_t = np.max(
                np.abs((matrix - point[None, :]) / scale[None, :]), axis=1
            )
            critical = float(np.quantile(max_t, 0.95))
            standard_error = matrix.std(axis=0, ddof=1)
            for regime_index, regime in enumerate(REGIME_ORDER):
                target = next(
                    row
                    for row in rows
                    if row["contrast"] == contrast
                    and row["regime"] == regime
                    and row["metric"] == metric
                )
                target["simultaneous_ci95_low"] = float(
                    target["effect"] - critical * standard_error[regime_index]
                )
                target["simultaneous_ci95_high"] = float(
                    target["effect"] + critical * standard_error[regime_index]
                )
                target["max_abs_t_critical"] = critical
    return pd.DataFrame(rows)
