"""Preserve the accepted assignment comparison and resampling calculation."""
from __future__ import annotations
from typing import Any
import json
import numpy as np
import pandas as pd

PRIMARY_REGIMES = (
    "IMA++",
    "LIDC",
    "LIDC-hard",
    "RIGA-disc",
    "RIGA-cup",
    "BSDS",
)

def paired_units(
    frame: pd.DataFrame,
    *,
    method_a: str,
    method_b: str,
    regime: str,
    metric: str,
) -> pd.DataFrame:
    subset = frame[frame["regime"].astype(str) == str(regime)]
    identity = [
        "endpoint",
        "subset",
        "visible_seed",
        "image_id",
        "source_id",
        "regime",
    ]
    left = subset[subset["method"].astype(str) == str(method_a)][identity + [metric]]
    right = subset[subset["method"].astype(str) == str(method_b)][identity + [metric]]
    if left.empty or right.empty:
        return pd.DataFrame()
    paired = left.merge(right, on=identity, how="inner", suffixes=("_a", "_b"), validate="one_to_one")
    paired["difference"] = paired[f"{metric}_a"].astype(float) - paired[f"{metric}_b"].astype(float)
    # Native-K methods legitimately leave unmatched budget metrics as NaN.
    # Such rows are unavailable for that contrast, not zero-valued evidence.
    paired["difference"] = paired["difference"].replace([np.inf, -np.inf], np.nan)
    paired = paired.dropna(subset=["difference"])
    if paired.empty:
        return pd.DataFrame()
    # The visible-mask seed is a protocol replicate, not an independent source.
    unit_identity = ["endpoint", "subset", "image_id", "source_id", "regime"]
    averaged = (
        paired.groupby(unit_identity, as_index=False)
        .agg(difference=("difference", "mean"), visible_seed_pairs=("visible_seed", "nunique"))
    )
    return averaged

def pooled_riga_units(
    frame: pd.DataFrame,
    *,
    method_a: str,
    method_b: str,
    metric: str,
) -> pd.DataFrame:
    pieces = [
        paired_units(
            frame,
            method_a=method_a,
            method_b=method_b,
            regime=regime,
            metric=metric,
        )
        for regime in ("RIGA-disc", "RIGA-cup")
    ]
    pieces = [piece for piece in pieces if not piece.empty]
    if len(pieces) != 2:
        return pd.DataFrame()
    output = pd.concat(pieces, ignore_index=True)
    output["regime"] = "RIGA-pooled"
    return output

def cluster_bootstrap(
    units: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    if units.empty:
        raise ValueError("cannot bootstrap an empty paired table")
    grouped = units.groupby("source_id", sort=True)["difference"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(dtype=np.float64)
    counts = grouped["count"].to_numpy(dtype=np.float64)
    generator = np.random.default_rng(int(seed))
    draws = generator.integers(0, len(grouped), size=(int(replicates), len(grouped)))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {
        "effect": float(units["difference"].mean()),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "ci90_low": float(np.quantile(boot, 0.05)),
        "ci90_high": float(np.quantile(boot, 0.95)),
        "bootstrap_mean": float(boot.mean()),
        "bootstrap_replicates": int(replicates),
        "source_clusters": int(len(grouped)),
        "evaluation_units": int(len(units)),
        "minimum_visible_seed_pairs": int(units["visible_seed_pairs"].min()),
        "maximum_visible_seed_pairs": int(units["visible_seed_pairs"].max()),
    }

def mlq_decision(effect: dict[str, float], delta: float) -> str:
    if effect["effect"] >= float(delta) and effect["ci95_low"] > 0.0:
        return "supported_improvement"
    if effect["effect"] <= -float(delta) and effect["ci95_high"] < 0.0:
        return "supported_degradation"
    if effect["ci90_low"] > -float(delta) and effect["ci90_high"] < float(delta):
        return "practical_equivalence"
    return "inconclusive"

def method_summaries(frame: pd.DataFrame, metrics: tuple[str, ...]) -> pd.DataFrame:
    available = [metric for metric in metrics if metric in frame.columns]
    per_seed = (
        frame.groupby(["method", "regime", "visible_seed"], as_index=False)[available]
        .mean(numeric_only=True)
    )
    aggregate = per_seed.groupby(["method", "regime"], as_index=False)[available].agg(
        ["mean", "std"]
    )
    aggregate.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in aggregate.columns
    ]
    return aggregate
