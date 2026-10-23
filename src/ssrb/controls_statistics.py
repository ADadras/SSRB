"""Accepted analysis profile; use only after frozen-prediction evaluation.

The paper's LIDC and LIDC-hard draws are independent in this published profile.
See statistics.py for linked draws for new analyses.
"""
from __future__ import annotations
import math
from typing import Any
import numpy as np
import pandas as pd

METHODS = ("ssrb", "posthoc_additive", "native_union_4_4", "conventional_residual", "wide_raw", "raw_capacity", "dino_capacity")

CONTROL_METHODS = ("posthoc_additive", "native_union_4_4", "conventional_residual", "wide_raw")

METRICS = ("MLQ@8", "Waste@8", "AUC", "MLQ@1", "MLQ@2", "MLQ@3", "MLQ@4", "MLQ@5", "MLQ@6", "MLQ@7", "PairDice@8", "Unique95")

def bootstrap(units: pd.DataFrame, replicates: int, seed: int) -> dict[str, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    results: dict[str, dict[str, Any]] = {}
    optimizer_draws = rng.integers(0, 3, size=(replicates, 3))
    shared_riga_source_draws: np.ndarray | None = None
    shared_riga_sources: list[str] | None = None
    available_regimes = set(units["regime"].unique())
    regime_order = [
        regime for regime in ("LIDC", "LIDC-hard", "IMA++", "BSDS", "RIGA-disc", "RIGA-cup")
        if regime in available_regimes
    ]
    for regime in regime_order:
        group = units[units["regime"] == regime]
        sources = sorted(group["source_id"].astype(str).unique())
        source_index = {value: index for index, value in enumerate(sources)}
        sums = np.zeros((len(sources), 3, len(METHODS), len(METRICS)), dtype=float)
        counts = np.zeros((len(sources), 3, len(METHODS)), dtype=float)
        method_index = {method: index for index, method in enumerate(METHODS)}
        for _, row in group.iterrows():
            s = source_index[str(row["source_id"])]
            o = int(row["optimizer_seed"])
            m = method_index[str(row["method_canonical"])]
            sums[s, o, m] += np.asarray([float(row[value]) for value in METRICS])
            counts[s, o, m] += 1
        if not np.array_equal(counts, counts[:, :, :1].repeat(len(METHODS), axis=2)):
            raise ValueError(f"pairing incomplete for {regime}")
        if regime == "RIGA-cup" and shared_riga_source_draws is not None:
            if sources != shared_riga_sources:
                raise ValueError("RIGA disc/cup source clusters do not align")
            source_draws = shared_riga_source_draws
        else:
            source_draws = rng.multinomial(
                len(sources), np.full(len(sources), 1 / len(sources)), size=replicates
            )
            if regime == "RIGA-disc":
                shared_riga_source_draws = source_draws
                shared_riga_sources = sources
        samples = np.zeros((replicates, len(METHODS), len(METRICS)), dtype=float)
        for r in range(replicates):
            numerator = np.zeros((len(METHODS), len(METRICS)), dtype=float)
            denominator = np.zeros(len(METHODS), dtype=float)
            weights = source_draws[r].astype(float)
            for optimizer_seed in optimizer_draws[r]:
                numerator += np.einsum("s,smq->mq", weights, sums[:, optimizer_seed])
                denominator += np.einsum("s,sm->m", weights, counts[:, optimizer_seed])
            samples[r] = numerator / denominator[:, None]
        point = np.asarray([
            [float(group[group["method_canonical"] == method][metric].mean()) for metric in METRICS]
            for method in METHODS
        ])
        results[regime] = {
            "point": point, "samples": samples,
            "methods": method_index, "metrics": {value: index for index, value in enumerate(METRICS)},
            "images": int(group["image_id"].nunique()), "sources": len(sources),
        }
    return results

def effect_table(boot: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for comparator in CONTROL_METHODS:
        for metric in METRICS:
            family_rows: list[dict[str, Any]] = []
            samples: list[np.ndarray] = []
            for regime in sorted(boot):
                info = boot[regime]
                left = info["methods"]["ssrb"]
                right = info["methods"][comparator]
                q = info["metrics"][metric]
                draws = info["samples"][:, left, q] - info["samples"][:, right, q]
                effect = float(info["point"][left, q] - info["point"][right, q])
                row = {
                    "regime": regime, "left": "ssrb", "right": comparator, "metric": metric,
                    "effect": effect, "ci95_low": float(np.quantile(draws, .025)),
                    "ci95_high": float(np.quantile(draws, .975)),
                    "simultaneous_ci95_low": math.nan, "simultaneous_ci95_high": math.nan,
                }
                family_rows.append(row)
                samples.append(draws)
            matrix = np.stack(samples, axis=1)
            point = np.asarray([row["effect"] for row in family_rows])
            scale = matrix.std(axis=0, ddof=1)
            scale = np.where(scale > 1e-12, scale, 1.0)
            max_t = np.max(np.abs((matrix - point[None]) / scale[None]), axis=1)
            critical = float(np.quantile(max_t, .95))
            for index, row in enumerate(family_rows):
                se = float(matrix[:, index].std(ddof=1))
                row["simultaneous_ci95_low"] = float(row["effect"] - critical * se)
                row["simultaneous_ci95_high"] = float(row["effect"] + critical * se)
            rows.extend(family_rows)
    return pd.DataFrame(rows)
