"""Build paired analysis units from portable evaluator outputs."""
from pathlib import Path
import json
import pandas as pd
from .data import sha256, write_json

REGIMES = dict(lidc='LIDC', lidc_hard='LIDC-hard', ima='IMA++', bsds='BSDS',
               riga_disc='RIGA-disc', riga_cup='RIGA-cup')


def load_evaluations(path):
    specs = json.loads(Path(path).read_text())
    frames, hashes = [], []
    for spec in specs:
        if set(spec) != {'evaluation', 'method', 'optimizer_seed'}:
            raise ValueError('evaluation spec requires evaluation, method, optimizer_seed')
        directory = Path(spec['evaluation'])
        if not directory.is_absolute():
            directory = Path(path).resolve().parent / directory
        filename = directory / 'per_image.json'
        frame = pd.DataFrame(json.loads(filename.read_text()))
        frame['regime'] = frame['endpoint'].map(REGIMES)
        frame['method_canonical'] = spec['method']
        frame['optimizer_seed'] = int(spec['optimizer_seed'])
        frame['subset'] = frame['endpoint'].map(lambda e: 'visible_dice_lt_0.80' if e=='lidc_hard' else 'all_hidden')
        if frame['regime'].isna().any():
            raise ValueError('unknown endpoint')
        frames.append(frame)
        hashes.append(dict(method=spec['method'], optimizer_seed=spec['optimizer_seed'], sha256=sha256(filename)))
    full = pd.concat(frames, ignore_index=True)
    keys = ['regime', 'method_canonical', 'visible_seed', 'optimizer_seed', 'image_id', 'source_id']
    if full.duplicated(keys).any():
        raise ValueError('duplicate result cells')
    return full, hashes


def run(evaluations, output, *, profile='controls', replicates=10000, seed=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    full, hashes = load_evaluations(evaluations)
    if profile == 'probunet':
        from . import probunet_statistics as stats
        units = stats.paired_units(full)
        boot, _ = stats.bootstrap_regimes(units, replicates=replicates, seed=20260814 if seed is None else seed)
        effects = stats.effect_table(boot)
        summary = stats.method_summary(units)
    else:
        if profile == 'controls':
            from . import controls_statistics as stats
            metrics, methods = stats.METRICS, stats.METHODS
        elif profile == 'dinov2':
            from . import dinov2_statistics as stats
            metrics, methods = stats.METRICS, ('dinov2_fusion','dinov2_capacity')
        elif profile == 'partial':
            from . import partial_statistics as stats
            metrics, methods = stats.METRICS, stats.METHODS
        else:
            raise ValueError('unknown accepted analysis profile')
        selected = full[full.method_canonical.isin(methods)]
        if selected[list(metrics)].isna().any().any():
            raise ValueError('missing required metrics')
        units = selected.groupby(['regime','optimizer_seed','image_id','source_id','method_canonical'],as_index=False).agg(
            **{m:(m,'mean') for m in metrics},visible_memberships=('visible_seed','nunique'))
        # Hard references depend on the visible mask. An image can therefore
        # contribute one, two or three visible choices to LIDC-hard.
        ordinary = units.regime != 'LIDC-hard'
        if not (units.loc[ordinary,'visible_memberships']==3).all() or set(units.optimizer_seed)!={0,1,2}:
            raise ValueError('requires all three visible and optimizer seeds for parent endpoints')
        parents = ('lidc','bsds') if profile in {'dinov2','partial'} else ('lidc','bsds','ima','riga_disc','riga_cup')
        if not selected.visible_seed.isin([0,1,2]).all():
            raise ValueError('unexpected visible seed')
        for method in methods:
            for endpoint in parents:
                for optimizer in range(3):
                    cell=selected[(selected.method_canonical==method) & (selected.endpoint==endpoint) &
                                  (selected.optimizer_seed==optimizer)]
                    if set(cell.visible_seed)!={0,1,2}:
                        raise ValueError('incomplete parent endpoint, method or seed family')
        paired=units.groupby(['regime','optimizer_seed','image_id','source_id']).method_canonical.nunique()
        if not (paired==len(methods)).all():
            raise ValueError('methods do not pair after visible-choice averaging')
        if profile == 'controls':
            boot = stats.bootstrap(units,replicates,20260815 if seed is None else seed)
            effects = stats.effect_table(boot)
        elif profile == 'partial':
            units = stats.average_visible_memberships(selected.rename(columns={'method_canonical':'method'}))
            boot = stats.bootstrap_contrasts(stats.paired_wide(units), replicates=replicates,
                                             seed=stats.SEED if seed is None else seed)
            effects = stats.effect_table(boot)
        else:
            if replicates != stats.REPS or (seed is not None and seed != stats.SEED):
                raise ValueError('DINOv2 profile fixes its resampling configuration')
            units = units.rename(columns={'method_canonical':'method'})
            boot = stats.bootstrap(units)
            rows=[]
            for q, metric in enumerate(metrics):
                _, low, high = stats.simultaneous(boot,q)
                for j, regime in enumerate(('LIDC','LIDC-hard','BSDS')):
                    import numpy as np
                    draws=boot[regime]['samples'][:,q]
                    rows.append(dict(regime=regime,metric=metric,effect=boot[regime]['point'][q],
                                     ci95_low=np.quantile(draws,.025),ci95_high=np.quantile(draws,.975),
                                     simultaneous_ci95_low=low[j],simultaneous_ci95_high=high[j]))
            effects=pd.DataFrame(rows)
        summary=units.groupby(['regime','method_canonical' if profile=='controls' else 'method'],as_index=False)[list(metrics)].mean()
    output.mkdir(parents=True)
    units.to_csv(output/'units.csv',index=False)
    effects.to_csv(output/'paired_effects.csv',index=False)
    summary.to_csv(output/'method_summary.csv',index=False)
    write_json(output/'analysis.json',dict(profile=profile,replicates=replicates,seed=seed if seed is not None else
               (20260815 if profile=='controls' else stats.SEED if profile=='partial' else 20260814),evaluations=hashes,
               evaluation_list_sha256=sha256(evaluations),resampling='paper resampling procedure'))
    return str(output)
