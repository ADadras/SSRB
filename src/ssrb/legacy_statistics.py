"""Representation-comparison bootstrap used for the paper.

This paper-specified calculation draws LIDC and LIDC-hard separately. Use the
linked-source calculation in statistics.py for new analyses requiring shared
parent/subset draws. Do not silently replace the published intervals.
"""
from __future__ import annotations
import math
from typing import Any
import numpy as np
import pandas as pd
METHODS = ("raw_residual_fusion", "dino_capacity", "raw_capacity")
METRICS = ("MLQ@8", "Waste@8")

def bootstrap_means(
    units: pd.DataFrame, *, replicates: int, seed: int
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    rng = np.random.default_rng(int(seed))
    regimes = sorted(units["regime"].unique())
    shared_riga_draws: np.ndarray | None = None
    results: dict[str, dict[str, Any]] = {}
    draws_record: dict[str, dict[str, Any]] = {}
    for regime in regimes:
        group = units[units["regime"] == regime].copy()
        sources = sorted(group["source_id"].astype(str).unique())
        source_index = {value: index for index, value in enumerate(sources)}
        sums = np.zeros((len(sources), 3, len(METHODS), len(METRICS)), dtype=np.float64)
        counts = np.zeros((len(sources), 3, len(METHODS)), dtype=np.float64)
        method_index = {value: index for index, value in enumerate(METHODS)}
        for _, row in group.iterrows():
            s = source_index[str(row["source_id"])]
            o = int(row["optimizer_seed"])
            m = method_index[str(row["method_canonical"])]
            sums[s, o, m, 0] += float(row["MLQ@8"])
            sums[s, o, m, 1] += float(row["Waste@8"])
            counts[s, o, m] += 1.0
        common_counts = counts[:, :, 0]
        if not np.array_equal(counts, common_counts[:, :, None].repeat(len(METHODS), axis=2)):
            raise ValueError(f"method pairing grid is incomplete for {regime}")
        if regime == "RIGA-cup" and shared_riga_draws is not None:
            source_draws = shared_riga_draws
        else:
            source_draws = rng.multinomial(
                len(sources), np.full(len(sources), 1.0 / len(sources)), size=int(replicates)
            )
            if regime == "RIGA-disc":
                shared_riga_draws = source_draws
        optimizer_draws = rng.integers(0, 3, size=(int(replicates), 3))
        bootstrap = np.zeros((int(replicates), len(METHODS), len(METRICS)), dtype=np.float64)
        for replicate in range(int(replicates)):
            numerator = np.zeros((len(METHODS), len(METRICS)), dtype=np.float64)
            denominator = np.zeros(len(METHODS), dtype=np.float64)
            weights = source_draws[replicate].astype(np.float64)
            for optimizer_seed in optimizer_draws[replicate]:
                numerator += np.einsum("s,smq->mq", weights, sums[:, optimizer_seed])
                denominator += np.einsum("s,sm->m", weights, counts[:, optimizer_seed])
            if (denominator <= 0).any():
                raise ValueError(f"empty bootstrap replicate for {regime}")
            bootstrap[replicate] = numerator / denominator[:, None]
        point = np.zeros((len(METHODS), len(METRICS)), dtype=np.float64)
        for m, method in enumerate(METHODS):
            subset = group[group["method_canonical"] == method]
            point[m] = [float(subset[metric].mean()) for metric in METRICS]
        results[regime] = {
            "point": point,
            "bootstrap": bootstrap,
            "methods": method_index,
            "metrics": {value: index for index, value in enumerate(METRICS)},
            "images": int(group["image_id"].nunique()),
            "sources": int(len(sources)),
        }
        draws_record[regime] = {
            "source_clusters": int(len(sources)),
            "optimizer_draw_shape": list(optimizer_draws.shape),
        }
    return results, draws_record

def effect_row(
    boot: dict[str, dict[str, Any]], regime: str, comparator: str, metric: str
) -> tuple[dict[str, Any], np.ndarray]:
    info = boot[regime]
    left = info["methods"]["raw_residual_fusion"]
    right = info["methods"][comparator]
    metric_index = info["metrics"][metric]
    samples = info["bootstrap"][:, left, metric_index] - info["bootstrap"][:, right, metric_index]
    effect = float(info["point"][left, metric_index] - info["point"][right, metric_index])
    return (
        {
            "regime": regime,
            "candidate": "raw_residual_fusion",
            "comparator": comparator,
            "metric": metric,
            "effect": effect,
            "ci95_low": float(np.quantile(samples, 0.025)),
            "ci95_high": float(np.quantile(samples, 0.975)),
            "simultaneous_ci95_low": math.nan,
            "simultaneous_ci95_high": math.nan,
        },
        samples,
    )

def simultaneous(rows: list[dict[str, Any]], samples: list[np.ndarray]) -> float:
    point = np.asarray([float(row["effect"]) for row in rows])
    matrix = np.stack(samples, axis=1)
    scale = matrix.std(axis=0, ddof=1)
    scale = np.where(scale > 1e-12, scale, 1.0)
    max_t = np.max(np.abs((matrix - point[None]) / scale[None]), axis=1)
    critical = float(np.quantile(max_t, 0.95))
    for index, row in enumerate(rows):
        se = float(matrix[:, index].std(ddof=1))
        row["simultaneous_ci95_low"] = float(row["effect"] - critical * se)
        row["simultaneous_ci95_high"] = float(row["effect"] + critical * se)
    return critical


def analyze_stage(
    boot: dict[str, dict[str, Any]], stage: str
) -> tuple[pd.DataFrame, pd.DataFrame, str, dict[str, float]]:
    if stage == "stage_a":
        mlq_specs = [
            ("LIDC-hard", "dino_capacity"),
            ("LIDC-hard", "raw_capacity"),
            ("BSDS", "dino_capacity"),
        ]
        waste_specs = [
            ("LIDC-hard", "dino_capacity"),
            ("BSDS", "dino_capacity"),
        ]
    else:
        mlq_specs = [
            (regime, comparator)
            for regime in ("IMA++", "RIGA-disc", "RIGA-cup")
            for comparator in ("dino_capacity", "raw_capacity")
        ]
        waste_specs = list(mlq_specs)
    rows: list[dict[str, Any]] = []
    mlq_rows: list[dict[str, Any]] = []
    mlq_samples: list[np.ndarray] = []
    for regime, comparator in mlq_specs:
        row, samples = effect_row(boot, regime, comparator, "MLQ@8")
        mlq_rows.append(row)
        mlq_samples.append(samples)
    waste_rows: list[dict[str, Any]] = []
    waste_samples: list[np.ndarray] = []
    for regime, comparator in waste_specs:
        row, samples = effect_row(boot, regime, comparator, "Waste@8")
        waste_rows.append(row)
        waste_samples.append(samples)
    critical = {
        "MLQ": simultaneous(mlq_rows, mlq_samples),
        "Waste": simultaneous(waste_rows, waste_samples),
    }
    rows.extend(mlq_rows)
    rows.extend(waste_rows)
    # Every candidate-versus-baseline endpoint/metric is also exported descriptively.
    present = {(row["regime"], row["comparator"], row["metric"]) for row in rows}
    for regime in sorted(boot):
        for comparator in ("dino_capacity", "raw_capacity"):
            for metric in METRICS:
                key = (regime, comparator, metric)
                if key not in present:
                    row, _ = effect_row(boot, regime, comparator, metric)
                    rows.append(row)
    effects = pd.DataFrame(rows)
    gate_rows: list[dict[str, Any]] = []
    if stage == "stage_a":
        lookup = effects.set_index(["regime", "comparator", "metric"])
        definitions = [
            ("LIDC-hard gain vs DINO", "LIDC-hard", "dino_capacity", "MLQ@8", ">=point_and_low", 0.010),
            ("LIDC-hard noninferior vs raw", "LIDC-hard", "raw_capacity", "MLQ@8", ">low", -0.010),
            ("LIDC-hard Waste vs DINO", "LIDC-hard", "dino_capacity", "Waste@8", "<=high", 0.050),
            ("BSDS noninferior vs DINO", "BSDS", "dino_capacity", "MLQ@8", ">low", -0.010),
            ("BSDS Waste vs DINO", "BSDS", "dino_capacity", "Waste@8", "<=high", 0.050),
        ]
    else:
        lookup = effects.set_index(["regime", "comparator", "metric"])
        definitions = []
        for regime in ("IMA++", "RIGA-disc", "RIGA-cup"):
            for comparator in ("dino_capacity", "raw_capacity"):
                definitions.append((f"{regime} MLQ vs {comparator}", regime, comparator, "MLQ@8", ">low", -0.010))
                definitions.append((f"{regime} Waste vs {comparator}", regime, comparator, "Waste@8", "<=high", 0.050))
    for name, regime, comparator, metric, predicate, threshold in definitions:
        row = lookup.loc[(regime, comparator, metric)]
        if predicate == ">=point_and_low":
            passed = float(row["effect"]) >= threshold and float(row["simultaneous_ci95_low"]) > 0.0
        elif predicate == ">low":
            passed = float(row["simultaneous_ci95_low"]) > threshold
        else:
            passed = float(row["simultaneous_ci95_high"]) <= threshold
        gate_rows.append(
            {
                "gate": name,
                "regime": regime,
                "comparator": comparator,
                "metric": metric,
                "effect": float(row["effect"]),
                "simultaneous_ci95_low": float(row["simultaneous_ci95_low"]),
                "simultaneous_ci95_high": float(row["simultaneous_ci95_high"]),
                "threshold": float(threshold),
                "passed": bool(passed),
            }
        )
    gates = pd.DataFrame(gate_rows)
    passed = bool(gates["passed"].all())
    decision = (
        "Fusion Stage-A PASS"
        if stage == "stage_a" and passed
        else "Fusion F3: Stage-A failure"
        if stage == "stage_a"
        else "Fusion F1: broad rescue"
        if passed
        else "Fusion F2: transfer failure"
    )
    return effects, gates, decision, critical

def summary_table(units: pd.DataFrame) -> pd.DataFrame:
    return (
        units.groupby(["regime", "method_canonical"], as_index=False, sort=True)
        .agg(
            MLQ8=("MLQ@8", "mean"),
            Waste8=("Waste@8", "mean"),
            images=("image_id", "nunique"),
            sources=("source_id", "nunique"),
        )
    )
