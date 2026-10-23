"""Accepted support-specificity contrasts, retaining the original early draw order."""
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from .analysis import load_evaluations
from .data import write_json
from .sharing import METHODS

REGIMES=('LIDC','LIDC-hard','BSDS')
OPTIMIZER_SEEDS=(0,1,2)
PREFIXES=tuple(range(1,9))


MLQ_METRICS = tuple(f"MLQ@{prefix}" for prefix in PREFIXES)

DESCRIPTIVE_METRICS = (
    *MLQ_METRICS,
    "Waste@8",
    "AUC",
    "Worst@8",
    "PairDice@8",
    "Unique95",
    "active_head_count",
    "Containment",
    "Gain@2",
    "Gain@4",
    "Gain@8",
)

def average_visible_memberships(image: pd.DataFrame) -> pd.DataFrame:
    """Pair first, then average available visible memberships per image/optimizer.

    Full endpoints must contribute all three visible seeds.  LIDC-hard may
    contribute one, two, or three because its membership is defined separately
    for each visible annotation.
    """

    keys = [
        "regime", "endpoint", "optimizer_seed", "image_id", "source_id",
        "method_canonical",
    ]
    units = image.groupby(keys, as_index=False)[list(DESCRIPTIVE_METRICS)].mean()
    counts = (
        image.groupby(keys, as_index=False)["visible_seed"]
        .nunique()
        .rename(columns={"visible_seed": "visible_seed_memberships"})
    )
    units = units.merge(counts, on=keys, validate="one_to_one")
    full = units[units["regime"] != "LIDC-hard"]
    if set(full["visible_seed_memberships"].astype(int)) != {3}:
        raise ValueError("a full-endpoint unit does not have all three visible memberships")
    hard_counts = set(
        units.loc[units["regime"] == "LIDC-hard", "visible_seed_memberships"].astype(int)
    )
    if not hard_counts or not hard_counts.issubset({1, 2, 3}):
        raise ValueError(f"invalid LIDC-hard visible membership counts: {hard_counts}")
    identity = ["regime", "optimizer_seed", "image_id", "source_id"]
    membership = units.groupby(identity, as_index=False)["method_canonical"].nunique()
    if set(membership["method_canonical"].astype(int)) != {len(METHODS)}:
        raise ValueError("visible-averaged units do not pair over all four support methods")
    if units.duplicated([*identity, "method_canonical"]).any():
        raise ValueError("duplicate visible-averaged method units")
    return units

def make_draws(
    units: pd.DataFrame, *, replicates: int, seed: int
) -> dict[str, Any]:
    if int(replicates) < 100:
        raise ValueError("bootstrap-replicates must be at least 100")
    rng = np.random.default_rng(int(seed))
    optimizer_counts = rng.multinomial(3, np.full(3, 1.0 / 3.0), size=int(replicates)).astype(np.int16)
    sources = {
        regime: sorted(units.loc[units["regime"] == regime, "source_id"].astype(str).unique())
        for regime in REGIMES
    }
    source_counts: dict[str, np.ndarray] = {}
    generated: dict[str, np.ndarray] = {}
    for regime in REGIMES:
        key = "RIGA-shared" if regime.startswith("RIGA-") else regime
        if key not in generated:
            count = len(sources[regime])
            generated[key] = rng.multinomial(
                count, np.full(count, 1.0 / count), size=int(replicates)
            ).astype(np.int16)
        source_counts[regime] = generated[key]
    return {
        "replicates": int(replicates),
        "seed": int(seed),
        "optimizer_counts": optimizer_counts,
        "sources": sources,
        "source_counts": source_counts,
    }

def bootstrap_method_means(
    units: pd.DataFrame, *, metrics: tuple[str, ...], draws: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    method_index = {method: index for index, method in enumerate(METHODS)}
    for regime in REGIMES:
        local = units[units["regime"] == regime]
        sources = draws["sources"][regime]
        source_index = {source: index for index, source in enumerate(sources)}
        sums = np.zeros(
            (len(sources), 3, len(METHODS), len(metrics)), dtype=np.float64
        )
        counts = np.zeros((len(sources), 3, len(METHODS)), dtype=np.float64)
        aggregated = local.groupby(
            ["source_id", "optimizer_seed", "method_canonical"],
            as_index=False,
        ).agg(
            **{f"sum__{metric}": (metric, "sum") for metric in metrics},
            image_count=("image_id", "count"),
        )
        for row in aggregated.to_dict("records"):
            source = source_index[str(row["source_id"])]
            optimizer = int(row["optimizer_seed"])
            method = method_index[str(row["method_canonical"])]
            sums[source, optimizer, method] = [
                float(row[f"sum__{metric}"]) for metric in metrics
            ]
            counts[source, optimizer, method] = float(row["image_count"])
        common_counts = counts[..., 0]
        for method in range(1, len(METHODS)):
            if not np.array_equal(counts[..., method], common_counts):
                raise ValueError(f"method membership differs within {regime}")
        if (common_counts <= 0).any():
            raise ValueError(f"crossed source/optimizer grid is incomplete for {regime}")

        replicates = int(draws["replicates"])
        boot_sum = np.zeros((replicates, len(METHODS) * len(metrics)), dtype=np.float64)
        boot_count = np.zeros(replicates, dtype=np.float64)
        source_weights = draws["source_counts"][regime].astype(np.float64, copy=False)
        optimizer_weights = draws["optimizer_counts"].astype(np.float64, copy=False)
        flattened = sums.reshape(len(sources), 3, len(METHODS) * len(metrics))
        for optimizer in OPTIMIZER_SEEDS:
            factor = optimizer_weights[:, optimizer]
            sampled_sum = source_weights @ flattened[:, optimizer, :]
            sampled_count = source_weights @ common_counts[:, optimizer]
            boot_sum += factor[:, None] * sampled_sum
            boot_count += factor * sampled_count
        if np.any(boot_count <= 0):
            raise ValueError(f"a crossed bootstrap replicate is empty for {regime}")
        boot = (boot_sum / boot_count[:, None]).reshape(
            replicates, len(METHODS), len(metrics)
        )
        point = np.empty((len(METHODS), len(metrics)), dtype=np.float64)
        for method, method_position in method_index.items():
            point[method_position] = local.loc[
                local["method_canonical"] == method, list(metrics)
            ].mean().to_numpy(dtype=np.float64)
        output[regime] = {
            "point": point,
            "boot": boot,
            "method_index": method_index,
            "metric_index": {metric: index for index, metric in enumerate(metrics)},
            "source_clusters": len(sources),
            "evaluation_units_per_method": int(len(local) / len(METHODS)),
        }
    return output

def summarize(full,*,replicates=10000,seed=20260814):
    if set(full.regime)!=set(REGIMES) or set(full.method_canonical)!=set(METHODS):
        raise ValueError('requires all four support controls on LIDC, LIDC-hard and BSDS')
    for method in METHODS:
        for endpoint in ('lidc','bsds'):
            for optimizer in range(3):
                cell=full[(full.method_canonical==method)&(full.endpoint==endpoint)&(full.optimizer_seed==optimizer)]
                if set(cell.visible_seed)!={0,1,2}:
                    raise ValueError('incomplete visible/optimizer family')
    units=average_visible_memberships(full)
    draws=make_draws(units,replicates=replicates,seed=seed)
    boot=bootstrap_method_means(units,metrics=('MLQ@8',),draws=draws)
    rows=[]
    for candidate,shuffled in (('capacity_sa_dmb','capacity_sa_dmb_shuffled'),('sa_dmb','sa_dmb_shuffled')):
        effects=[];samples=[]
        for regime in ('LIDC-hard','BSDS'):
            info=boot[regime];left=info['method_index'][candidate];right=info['method_index'][shuffled]
            point=float(info['point'][left,0]-info['point'][right,0])
            sample=info['boot'][:,left,0]-info['boot'][:,right,0]
            effects.append(point);samples.append(sample)
            rows.append(dict(candidate=candidate,component=regime,effect=point,
                ci95_low=float(np.quantile(sample,.025)),ci95_high=float(np.quantile(sample,.975)),
                threshold=-.005,predicate_pass=bool(point>=-.005)))
        point=float(np.mean(effects));sample=np.stack(samples,axis=1).mean(axis=1)
        low,high=float(np.quantile(sample,.025)),float(np.quantile(sample,.975))
        rows.append(dict(candidate=candidate,component='M=0.5*(LIDC-hard+BSDS)',effect=point,
            ci95_low=low,ci95_high=high,threshold=.005,
            predicate_pass=bool(point>=.005 and low>0. and min(effects)>=-.005)))
    return pd.DataFrame(rows)


def analyze(evaluations,output,*,replicates=10000,seed=20260814):
    output=Path(output)
    if output.exists():
        raise FileExistsError(output)
    full,hashes=load_evaluations(evaluations)
    full['endpoint']=full.endpoint.replace({'lidc_hard':'lidc'})
    result=summarize(full,replicates=replicates,seed=seed)
    output.mkdir(parents=True)
    result.to_csv(output/'support_specificity.csv',index=False)
    write_json(output/'analysis.json',dict(status='complete',replicates=replicates,seed=seed,
        rows=len(result),evaluations=hashes,resampling='original optimizer multinomial followed by LIDC, LIDC-hard and BSDS source draws'))
    return str(output)
