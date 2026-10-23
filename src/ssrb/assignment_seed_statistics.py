"""Preserve the accepted assignment comparison and resampling calculation."""
from __future__ import annotations
from typing import Any
import json
import numpy as np
import pandas as pd

VARIANTS = ("dmb_wta", "soft_tau005", "dmb_rn", "capacity_wta")

VISIBLE_SEEDS = (0, 1, 2)

OPTIMIZER_SEEDS = (0, 1, 2)

REGIMES = ("LIDC", "LIDC-hard")

METRICS = (
    "MLQ@8",
    "AUC",
    "Worst@8",
    "Waste@8",
    "PairDice@8",
    "Containment",
    "Unique95",
    "active_head_count",
)

COMPARISONS = (
    ("dmb_rn", "dmb_wta"),
    ("dmb_rn", "soft_tau005"),
    ("dmb_rn", "capacity_wta"),
)

def two_way_bootstrap(
    units: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if set(units["optimizer_seed"].astype(int)) != set(OPTIMIZER_SEEDS):
        raise ValueError("two-way bootstrap requires all optimizer seeds")
    aggregated = (
        units.groupby(["optimizer_seed", "source_id"], as_index=False)["difference"]
        .agg(["sum", "count"])
        .reset_index()
    )
    sources = sorted(aggregated["source_id"].astype(str).unique())
    sums = np.zeros((len(OPTIMIZER_SEEDS), len(sources)), dtype=np.float64)
    counts = np.zeros_like(sums)
    source_index = {value: index for index, value in enumerate(sources)}
    seed_index = {value: index for index, value in enumerate(OPTIMIZER_SEEDS)}
    for row in aggregated.to_dict("records"):
        i = seed_index[int(row["optimizer_seed"])]
        j = source_index[str(row["source_id"])]
        sums[i, j] = float(row["sum"])
        counts[i, j] = float(row["count"])
    if (counts <= 0).any():
        raise ValueError("optimizer seed/source matrix is incomplete")
    generator = np.random.default_rng(int(seed))
    seed_draws = generator.integers(
        0, len(OPTIMIZER_SEEDS), size=(int(replicates), len(OPTIMIZER_SEEDS))
    )
    source_draws = generator.integers(
        0, len(sources), size=(int(replicates), len(sources))
    )
    sampled_sums = sums[seed_draws[:, :, None], source_draws[:, None, :]].sum(
        axis=(1, 2)
    )
    sampled_counts = counts[
        seed_draws[:, :, None], source_draws[:, None, :]
    ].sum(axis=(1, 2))
    boot = sampled_sums / sampled_counts
    seed_effects = {
        str(optimizer_seed): float(
            units.loc[
                units["optimizer_seed"].astype(int) == optimizer_seed,
                "difference",
            ].mean()
        )
        for optimizer_seed in OPTIMIZER_SEEDS
    }
    values = np.asarray(list(seed_effects.values()), dtype=np.float64)
    return {
        "effect": float(units["difference"].mean()),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "ci90_low": float(np.quantile(boot, 0.05)),
        "ci90_high": float(np.quantile(boot, 0.95)),
        "bootstrap_mean": float(boot.mean()),
        "bootstrap_replicates": int(replicates),
        "source_clusters": int(len(sources)),
        "optimizer_seed_count": int(len(OPTIMIZER_SEEDS)),
        "evaluation_units": int(len(units)),
        "seed_effects": json.dumps(seed_effects, sort_keys=True),
        "seed_effect_min": float(values.min()),
        "seed_effect_max": float(values.max()),
        "seed_effect_std": float(values.std(ddof=1)),
    }

def decision(effect: dict[str, Any], delta: float) -> str:
    seed_effects = np.asarray(
        list(json.loads(str(effect["seed_effects"])).values()), dtype=np.float64
    )
    if (
        float(effect["effect"]) >= float(delta)
        and float(effect["ci95_low"]) > 0.0
        and (seed_effects > 0.0).all()
    ):
        return "supported_improvement_all_seeds"
    if (
        float(effect["effect"]) <= -float(delta)
        and float(effect["ci95_high"]) < 0.0
        and (seed_effects < 0.0).all()
    ):
        return "supported_degradation_all_seeds"
    if (
        float(effect["ci90_low"]) > -float(delta)
        and float(effect["ci90_high"]) < float(delta)
    ):
        return "practical_equivalence_two_way"
    return "inconclusive"

from types import SimpleNamespace
base=SimpleNamespace(OPTIMIZER_SEEDS=OPTIMIZER_SEEDS,VARIANTS=VARIANTS,REGIMES=REGIMES,METRICS=METRICS,COMPARISONS=COMPARISONS,two_way_bootstrap=two_way_bootstrap,decision=decision)

def paired_units(
    frame: pd.DataFrame,
    *,
    method_a: str,
    method_b: str,
    regime: str,
    metric: str,
) -> pd.DataFrame:
    subset = frame[frame["regime"].astype(str) == regime]
    identity = [
        "endpoint",
        "subset",
        "optimizer_seed",
        "visible_seed",
        "image_id",
        "source_id",
        "regime",
    ]
    left = subset[subset["variant"] == method_a][identity + [metric]]
    right = subset[subset["variant"] == method_b][identity + [metric]]
    paired = left.merge(
        right,
        on=identity,
        how="outer",
        suffixes=("_a", "_b"),
        validate="one_to_one",
        indicator=True,
    )
    if paired.empty or set(paired["_merge"].astype(str)) != {"both"}:
        raise ValueError("paired robustness methods do not share exact protocol rows")
    paired["difference"] = (
        pd.to_numeric(paired[f"{metric}_a"], errors="coerce")
        - pd.to_numeric(paired[f"{metric}_b"], errors="coerce")
    ).replace([np.inf, -np.inf], np.nan)
    if paired["difference"].isna().any():
        raise ValueError("paired robustness metric is not finite on every protocol row")
    unit_identity = [
        "endpoint",
        "subset",
        "optimizer_seed",
        "image_id",
        "source_id",
        "regime",
    ]
    units = (
        paired.groupby(unit_identity, as_index=False)
        .agg(
            difference=("difference", "mean"),
            visible_seed_pairs=("visible_seed", "nunique"),
        )
    )
    pair_counts = set(units["visible_seed_pairs"].astype(int))
    if not pair_counts or not pair_counts.issubset({1, 2, 3}):
        raise ValueError("paired robustness units have invalid visible-seed coverage")
    if regime == "LIDC" and pair_counts != {3}:
        raise ValueError("full LIDC units do not contain all visible seeds")
    if set(units["optimizer_seed"].astype(int)) != set(base.OPTIMIZER_SEEDS):
        raise ValueError("paired robustness units do not contain all optimizer seeds")
    optimizer_identity = ["optimizer_seed", "image_id", "source_id"]
    optimizer_sets = [
        set(map(tuple, group[optimizer_identity[1:]].to_records(index=False)))
        for _, group in units.groupby("optimizer_seed", sort=True)
    ]
    if any(value != optimizer_sets[0] for value in optimizer_sets[1:]):
        raise ValueError("optimizer seeds do not share the same evaluation units")
    return units

def analyze(
    frame: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    delta: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = (
        frame.groupby(["variant", "regime", "optimizer_seed"], as_index=False)[
            list(base.METRICS)
        ]
        .mean(numeric_only=True)
    )
    effects: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for method_a, method_b in base.COMPARISONS:
        for regime in base.REGIMES:
            for metric in base.METRICS:
                units = paired_units(
                    frame,
                    method_a=method_a,
                    method_b=method_b,
                    regime=regime,
                    metric=metric,
                )
                inference = base.two_way_bootstrap(
                    units,
                    replicates=int(bootstrap_replicates),
                    seed=int(bootstrap_seed),
                )
                effects.append(
                    {
                        "method_a": method_a,
                        "method_b": method_b,
                        "contrast": f"{method_a}_minus_{method_b}",
                        "regime": regime,
                        "metric": metric,
                        **inference,
                        "minimum_visible_seed_pairs": int(
                            units["visible_seed_pairs"].min()
                        ),
                        "maximum_visible_seed_pairs": int(
                            units["visible_seed_pairs"].max()
                        ),
                        "practical_margin": (
                            float(delta) if metric == "MLQ@8" else None
                        ),
                        "decision": (
                            base.decision(inference, float(delta))
                            if metric == "MLQ@8"
                            else "descriptive"
                        ),
                    }
                )
                coverage.append(
                    {
                        "method_a": method_a,
                        "method_b": method_b,
                        "regime": regime,
                        "metric": metric,
                        "available": True,
                        "evaluation_units": int(len(units)),
                        "source_clusters": int(units["source_id"].nunique()),
                        "minimum_visible_seed_pairs": int(
                            units["visible_seed_pairs"].min()
                        ),
                        "maximum_visible_seed_pairs": int(
                            units["visible_seed_pairs"].max()
                        ),
                    }
                )
    return summary, pd.DataFrame(effects), pd.DataFrame(coverage)
