"""Accepted analysis profile; use only after frozen-prediction evaluation.

The paper's LIDC and LIDC-hard draws are independent in this published profile.
See statistics.py for linked draws for new analyses.
"""
from __future__ import annotations
import math
from typing import Any
import numpy as np
import pandas as pd

REPS = 10_000

SEED = 20_260_814

METRICS = ("MLQ@8", "Waste@8")

def bootstrap(units: pd.DataFrame) -> dict[str, dict[str, Any]]:
    rng = np.random.default_rng(SEED)
    out = {}
    for regime in ("LIDC", "LIDC-hard", "BSDS"):
        group = units[units.regime == regime]
        sources = sorted(group.source_id.astype(str).unique())
        si = {s: i for i, s in enumerate(sources)}
        methods = {"dinov2_fusion": 0, "dinov2_capacity": 1}
        sums = np.zeros((len(sources), 3, 2, 2)); counts = np.zeros((len(sources), 3, 2))
        for row in group.to_dict("records"):
            s, o, m = si[str(row["source_id"])], int(row["optimizer_seed"]), methods[row["method"]]
            counts[s,o,m] += 1
            for q, metric in enumerate(METRICS): sums[s,o,m,q] += float(row[metric])
        if not np.array_equal(counts[:,:,0], counts[:,:,1]): raise ValueError(f"pairing failed: {regime}")
        source_draws = rng.multinomial(len(sources), np.full(len(sources), 1/len(sources)), size=REPS)
        os_draws = rng.integers(0,3,size=(REPS,3)); samples=np.zeros((REPS,2))
        for r in range(REPS):
            num=np.zeros((2,2)); den=np.zeros(2); w=source_draws[r].astype(float)
            for o in os_draws[r]:
                num += np.einsum("s,smq->mq",w,sums[:,o]); den += np.einsum("s,sm->m",w,counts[:,o])
            means=num/den[:,None]; samples[r]=means[0]-means[1]
        point=np.asarray([group[group.method=="dinov2_fusion"][m].mean()-group[group.method=="dinov2_capacity"][m].mean() for m in METRICS])
        out[regime]={"point":point,"samples":samples,"images":int(group.image_id.nunique()),"sources":len(sources)}
    return out

def simultaneous(boot: dict[str, dict[str, Any]], metric_index: int) -> tuple[float, np.ndarray, np.ndarray]:
    regimes=("LIDC", "LIDC-hard", "BSDS")
    matrix=np.stack([boot[r]["samples"][:,metric_index] for r in regimes],axis=1)
    point=np.asarray([boot[r]["point"][metric_index] for r in regimes]); scale=matrix.std(0,ddof=1); scale=np.where(scale>1e-12,scale,1)
    critical=float(np.quantile(np.max(np.abs((matrix-point)/scale),axis=1),.95)); se=matrix.std(0,ddof=1)
    return critical, point-critical*se, point+critical*se
