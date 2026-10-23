"""Residual summaries using the accepted crossed source/optimizer bootstrap."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .data import sha256,write_json

ENDPOINTS=('lidc','bsds','ima','riga_disc','riga_cup')
DIAGNOSTIC_METRICS = (
    "residual_mean_abs", "residual_rms", "relative_residual_rms",
    "anchor_fused_dice", "signed_area_change", "absolute_area_change",
    "absolute_component_change", "symmetric_boundary_displacement_px",
    "residual_magnitude_in_boundary_band", "boundary_to_nonboundary_magnitude_ratio",
)


def summarize(full, *, replicates=10000, seed=20260815):
    if replicates<1:
        raise ValueError('bootstrap replicates must be positive')
    endpoints=ENDPOINTS
    keys=['endpoint','visible_seed','optimizer_seed','image_id','source_id','head_id']
    if set(full.endpoint)!=set(endpoints) or full.duplicated(keys).any():
        raise ValueError('requires five tasks without duplicate heads')
    if not full.head_id.isin(range(8)).all() or not np.isfinite(full[list(DIAGNOSTIC_METRICS)].to_numpy(float)).all():
        raise ValueError('invalid head identity or diagnostic value')
    groups=full.groupby(['endpoint','visible_seed','optimizer_seed','image_id','source_id'])
    if not (groups.size()==8).all() or not (groups.head_id.nunique()==8).all():
        raise ValueError('each image requires eight distinct heads')
    for endpoint in endpoints:
        group=full[full.endpoint==endpoint]
        cells=set(zip(group.visible_seed,group.optimizer_seed))
        if cells!={(v,o) for v in range(3) for o in range(3)}:
            raise ValueError('requires every visible and optimizer seed')
    if not (full.groupby(['endpoint','optimizer_seed','image_id','source_id']).visible_seed.nunique()==3).all():
        raise ValueError('each image requires all three visible choices')
    per_image = full.groupby(["endpoint", "visible_seed", "optimizer_seed", "image_id", "source_id"], as_index=False).agg(
        **{metric: (metric, "mean") for metric in DIAGNOSTIC_METRICS}
    )
    averaged = per_image.groupby(["endpoint", "optimizer_seed", "image_id", "source_id"], as_index=False).agg(
        **{metric: (metric, "mean") for metric in DIAGNOSTIC_METRICS}, visible_memberships=("visible_seed", "nunique")
    )
    rng = np.random.default_rng(int(seed) + 991)
    replicates = int(replicates)
    optimizer_draws = rng.integers(0, 3, size=(replicates, 3))
    shared_riga_source_draws: np.ndarray | None = None
    shared_riga_sources: list[str] | None = None
    rows = []
    for endpoint in endpoints:
        group = averaged[averaged["endpoint"] == endpoint]
        sources = sorted(group["source_id"].astype(str).unique())
        source_index = {value: index for index, value in enumerate(sources)}
        sums = np.zeros((len(sources), 3, len(DIAGNOSTIC_METRICS)), dtype=float)
        counts = np.zeros((len(sources), 3), dtype=float)
        for _, row in group.iterrows():
            source = source_index[str(row["source_id"])]
            optimizer_seed = int(row["optimizer_seed"])
            sums[source, optimizer_seed] += np.asarray(
                [float(row[metric]) for metric in DIAGNOSTIC_METRICS]
            )
            counts[source, optimizer_seed] += 1
        if (counts <= 0).any():
            raise ValueError(f"incomplete diagnostic source/optimizer grid: {endpoint}")
        if endpoint == "riga_cup" and shared_riga_source_draws is not None:
            if sources != shared_riga_sources:
                raise ValueError("RIGA diagnostic source clusters do not align")
            source_draws = shared_riga_source_draws
        else:
            source_draws = rng.multinomial(
                len(sources), np.full(len(sources), 1 / len(sources)), size=replicates
            )
            if endpoint == "riga_disc":
                shared_riga_source_draws = source_draws
                shared_riga_sources = sources
        weighted_sums = np.einsum("rs,som->rom", source_draws, sums)
        weighted_counts = np.einsum("rs,so->ro", source_draws, counts)
        row_indices = np.arange(replicates)[:, None]
        numerator = weighted_sums[row_indices, optimizer_draws].sum(axis=1)
        denominator = weighted_counts[row_indices, optimizer_draws].sum(axis=1)
        samples = numerator / denominator[:, None]
        for metric_index, metric in enumerate(DIAGNOSTIC_METRICS):
            rows.append({
                "endpoint": endpoint, "metric": metric, "mean": float(group[metric].mean()),
                "ci95_low": float(np.quantile(samples[:, metric_index], .025)),
                "ci95_high": float(np.quantile(samples[:, metric_index], .975)),
            })
    return pd.DataFrame(rows)


def analyze(diagnostics, output, *, replicates=10000, seed=20260815):
    manifest,output=Path(diagnostics),Path(output)
    if output.exists():
        raise FileExistsError(output)
    names=json.loads(manifest.read_text())
    if not isinstance(names,list) or not names or not all(isinstance(x,str) for x in names):
        raise ValueError('diagnostic list requires directory paths')
    paths=[(Path(n) if Path(n).is_absolute() else manifest.resolve().parent/n).resolve() for n in names]
    if len(set(paths))!=len(paths):
        raise ValueError('duplicate diagnostic directory')
    frames=[];inputs=[]
    for path in paths:
        record=json.loads((path/'provenance.json').read_text())
        rows=json.loads((path/'head_diagnostics.json').read_text())
        if record.get('reference_masks_read') is not False or record.get('rows')!=len(rows):
            raise ValueError('invalid diagnostic provenance')
        frames.append(pd.DataFrame(rows))
        inputs.append(dict(head_diagnostics_sha256=sha256(path/'head_diagnostics.json'),
                           provenance_sha256=sha256(path/'provenance.json')))
    result=summarize(pd.concat(frames,ignore_index=True),replicates=replicates,seed=seed)
    output.mkdir(parents=True)
    result.to_csv(output/'residual_diagnostic_summary.csv',index=False)
    write_json(output/'analysis.json',dict(status='complete',replicates=replicates,seed=seed,
        bootstrap_rng_seed=seed+991,shared_riga_source_draws=True,rows=len(result),
        diagnostic_list_sha256=sha256(manifest),inputs=inputs))
    return str(output)
