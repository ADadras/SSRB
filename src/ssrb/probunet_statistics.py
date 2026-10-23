"""Accepted analysis profile; use only after frozen-prediction evaluation.

The paper's LIDC and LIDC-hard draws are independent in this published profile.
See statistics.py for linked draws for new analyses.
"""
from __future__ import annotations
import math
from typing import Any
import numpy as np
import pandas as pd

METRICS = ("MLQ@8", "AUC", "Waste@8", "PairDice@8", "Unique95")

PREFIX_METRICS = tuple(f"MLQ@{k}" for k in range(1, 9))

ALL_METRICS = tuple(dict.fromkeys((*METRICS, *PREFIX_METRICS)))

REPLICATES = 10_000

BOOTSTRAP_SEED = 20_260_814

def paired_units(full: pd.DataFrame) -> pd.DataFrame:
    candidate = full[full["method_canonical"] == "raw_residual_fusion"].copy()
    prob = full[full["method_canonical"] == "probunet"].copy()
    keys = ["regime", "visible_seed", "image_id", "source_id", "subset"]
    prob = prob[keys + list(ALL_METRICS)].rename(
        columns={metric: f"probunet__{metric}" for metric in ALL_METRICS}
    )
    merged = candidate.merge(prob, on=keys, how="inner", validate="many_to_one")
    if len(merged) != len(candidate):
        raise ValueError("candidate and Prob-U do not pair exactly")
    for metric in ALL_METRICS:
        merged[f"delta__{metric}"] = merged[metric] - merged[f"probunet__{metric}"]
    aggregations: dict[str, tuple[str, str]] = {
        f"delta__{metric}": (f"delta__{metric}", "mean")
        for metric in ALL_METRICS
    }
    aggregations.update(
        {
            f"candidate__{metric}": (metric, "mean")
            for metric in ALL_METRICS
        }
    )
    aggregations.update(
        {
            f"probunet__{metric}": (f"probunet__{metric}", "mean")
            for metric in ALL_METRICS
        }
    )
    units = (
        merged.groupby(
            ["regime", "optimizer_seed", "image_id", "source_id"],
            as_index=False,
            sort=True,
        )
        .agg(**aggregations, visible_memberships=("visible_seed", "nunique"))
    )
    if set(units["optimizer_seed"].astype(int)) != {0, 1, 2}:
        raise ValueError("optimizer-seed pairing is incomplete")
    return units

def bootstrap_regimes(
    units: pd.DataFrame, *, replicates: int, seed: int
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    all_metrics = ALL_METRICS
    outputs: dict[str, dict[str, Any]] = {}
    draw_record: dict[str, Any] = {}
    riga_source_draws: np.ndarray | None = None
    for regime in ("LIDC", "LIDC-hard", "IMA++", "RIGA-disc", "RIGA-cup"):
        group = units[units["regime"] == regime].copy()
        if group.empty:
            raise ValueError(f"no paired units for {regime}")
        sources = sorted(group["source_id"].astype(str).unique())
        source_index = {source: index for index, source in enumerate(sources)}
        sums = np.zeros((len(sources), 3, len(all_metrics)), dtype=np.float64)
        counts = np.zeros((len(sources), 3), dtype=np.float64)
        for row in group.to_dict("records"):
            source = source_index[str(row["source_id"])]
            optimizer = int(row["optimizer_seed"])
            counts[source, optimizer] += 1.0
            for metric_index, metric in enumerate(all_metrics):
                sums[source, optimizer, metric_index] += float(row[f"delta__{metric}"])
        if (counts <= 0).any():
            raise ValueError(f"incomplete optimizer/source grid for {regime}")
        if regime == "RIGA-cup" and riga_source_draws is not None:
            if len(sources) != riga_source_draws.shape[1]:
                raise ValueError("RIGA disc/cup source grids differ")
            source_draws = riga_source_draws
        else:
            source_draws = rng.multinomial(
                len(sources), np.full(len(sources), 1.0 / len(sources)), size=replicates
            )
            if regime == "RIGA-disc":
                riga_source_draws = source_draws
        optimizer_draws = rng.integers(0, 3, size=(replicates, 3))
        samples = np.zeros((replicates, len(all_metrics)), dtype=np.float64)
        for replicate in range(replicates):
            weights = source_draws[replicate].astype(np.float64)
            numerator = np.zeros(len(all_metrics), dtype=np.float64)
            denominator = 0.0
            for optimizer in optimizer_draws[replicate]:
                numerator += np.einsum("s,sm->m", weights, sums[:, optimizer])
                denominator += float(np.dot(weights, counts[:, optimizer]))
            samples[replicate] = numerator / denominator
        point = np.asarray(
            [float(group[f"delta__{metric}"].mean()) for metric in all_metrics],
            dtype=np.float64,
        )
        outputs[regime] = {
            "point": point,
            "samples": samples,
            "metric_index": {metric: index for index, metric in enumerate(all_metrics)},
            "images": int(group["image_id"].nunique()),
            "sources": int(len(sources)),
        }
        draw_record[regime] = {
            "sources": int(len(sources)),
            "images": int(group["image_id"].nunique()),
        }
    return outputs, draw_record

def effect_table(boot: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for regime, info in boot.items():
        for metric, index in info["metric_index"].items():
            samples = info["samples"][:, index]
            rows.append(
                {
                    "regime": regime,
                    "contrast": "raw_residual_fusion_minus_probunet",
                    "metric": metric,
                    "effect": float(info["point"][index]),
                    "ci95_low": float(np.quantile(samples, 0.025)),
                    "ci95_high": float(np.quantile(samples, 0.975)),
                    "simultaneous_ci95_low": np.nan,
                    "simultaneous_ci95_high": np.nan,
                }
            )
    frame = pd.DataFrame(rows)
    for metric in ("MLQ@8", "Waste@8"):
        mask = frame["metric"].eq(metric)
        chosen = frame[mask].copy()
        matrix = np.stack(
            [boot[row.regime]["samples"][:, boot[row.regime]["metric_index"][metric]]
             for row in chosen.itertuples(index=False)],
            axis=1,
        )
        point = chosen["effect"].to_numpy(np.float64)
        scale = matrix.std(axis=0, ddof=1)
        scale = np.where(scale > 1e-12, scale, 1.0)
        critical = float(np.quantile(np.max(np.abs((matrix - point) / scale), axis=1), 0.95))
        se = matrix.std(axis=0, ddof=1)
        frame.loc[mask, "simultaneous_ci95_low"] = point - critical * se
        frame.loc[mask, "simultaneous_ci95_high"] = point + critical * se
    return frame

def method_summary(units: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for regime, group in units.groupby("regime", sort=True):
        for method, prefix in (
            ("raw_residual_fusion", "candidate"), ("probunet", "probunet")
        ):
            row: dict[str, Any] = {"regime": regime, "method_canonical": method}
            for metric in METRICS:
                label = metric.replace("@", "")
                row[label] = float(group[f"{prefix}__{metric}"].mean())
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary["redundant_head_fraction_95"] = 1.0 - summary["Unique95"] / 8.0
    return summary
