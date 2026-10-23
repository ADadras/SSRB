"""Rebuild the reported assignment comparisons from frozen evaluations."""
from pathlib import Path
import pandas as pd
from .analysis import load_evaluations
from .data import sha256, write_json

METRICS = ('MLQ@8','AUC','Worst@8','Waste@8','PairDice@8','Containment','Unique95','active_head_count')


def validate_grid(frame, methods, endpoints, optimizers):
    selected = frame[frame.method_canonical.isin(methods)].copy()
    selected['endpoint'] = selected.endpoint.replace({'lidc_hard':'lidc'})
    observed = set(map(tuple, selected[['method_canonical','endpoint','visible_seed','optimizer_seed']].drop_duplicates().to_records(index=False)))
    expected = {(m,e,v,o) for m in methods for e in endpoints for v in range(3) for o in optimizers}
    if observed != expected:
        raise ValueError('assignment method, endpoint or seed family is incomplete or unexpected')
    if selected[list(METRICS)].isna().any().any():
        raise ValueError('assignment comparisons require finite metrics')
    selected['method'] = selected.method_canonical
    selected['variant'] = selected.method_canonical
    return selected


def run(evaluations, output, *, profile, replicates=10000, seed=None):
    output=Path(output)
    if output.exists():
        raise FileExistsError(output)
    if replicates < 10000:
        raise ValueError('paper assignment analysis requires at least 10000 draws')
    frame, hashes=load_evaluations(evaluations)
    if profile=='lidc':
        from . import assignment_seed_statistics as stats
        selected=validate_grid(frame, stats.VARIANTS, ('lidc',), range(3))
        seed=3664 if seed is None else seed
        summary,effects,coverage=stats.analyze(selected,bootstrap_replicates=replicates,bootstrap_seed=seed,delta=.005)
    elif profile in ('visible','capacity'):
        from . import assignment_cluster_statistics as stats
        methods=('dmb_rn','dmb_wta','soft_tau005') if profile=='visible' else ('dmb_rn','cb_wta')
        endpoints=('lidc','ima','bsds','riga_disc','riga_cup') if profile=='visible' else ('riga_disc','riga_cup')
        selected=validate_grid(frame,methods,endpoints,(0,))
        seed=(3450 if profile=='visible' else 3633) if seed is None else seed
        summary=stats.method_summaries(selected,METRICS)
        effects=[];coverage=[]
        regimes=list(stats.PRIMARY_REGIMES)+['RIGA-pooled'] if profile=='visible' else ['RIGA-disc','RIGA-cup','RIGA-pooled']
        for comparator in methods[1:]:
            for regime in regimes:
                for metric in METRICS:
                    fn=stats.pooled_riga_units if regime=='RIGA-pooled' else stats.paired_units
                    kwargs=dict(method_a='dmb_rn',method_b=comparator,metric=metric)
                    if regime!='RIGA-pooled':
                        kwargs['regime']=regime
                    units=fn(selected,**kwargs)
                    if units.empty:
                        raise ValueError('assignment contrast has no paired observations')
                    inference=stats.cluster_bootstrap(units,replicates=replicates,seed=seed)
                    effects.append(dict(method_a='dmb_rn',method_b=comparator,contrast='dmb_rn_minus_'+comparator,regime=regime,metric=metric,**inference,practical_margin=.005 if metric=='MLQ@8' else None,decision=stats.mlq_decision(inference,.005) if metric=='MLQ@8' else 'descriptive'))
                    coverage.append(dict(method_a='dmb_rn',method_b=comparator,regime=regime,metric=metric,available=True))
        effects=pd.DataFrame(effects);coverage=pd.DataFrame(coverage)
    else:
        raise ValueError('unknown assignment analysis profile')
    output.mkdir(parents=True)
    summary.to_csv(output/'method_summary.csv',index=False)
    effects.to_csv(output/'paired_effects.csv',index=False)
    coverage.to_csv(output/'comparison_coverage.csv',index=False)
    write_json(output/'analysis.json',dict(profile=profile,replicates=replicates,seed=seed,evaluation_list_sha256=sha256(evaluations),evaluations=hashes,resampling='accepted assignment profile'))
    return str(output)
