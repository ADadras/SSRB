"""Run both reported stages of the representation comparison."""
from pathlib import Path
from .analysis import load_evaluations
from . import legacy_statistics as stats
from .data import sha256,write_json


def run(evaluations,output,*,replicates=10000,seed=20260814):
    output=Path(output)
    if output.exists():
        raise FileExistsError(output)
    if replicates < 10000:
        raise ValueError('paper representation analysis requires at least 10000 draws')
    full,hashes=load_evaluations(evaluations)
    full=full[full.method_canonical.isin(stats.METHODS)]
    expected={(m,e,v,o) for m in stats.METHODS for e in ('lidc','bsds','ima','riga_disc','riga_cup') for v in range(3) for o in range(3)}
    observed=set(map(tuple,full[full.endpoint!='lidc_hard'][['method_canonical','endpoint','visible_seed','optimizer_seed']].drop_duplicates().to_records(index=False)))
    if observed!=expected or full[list(stats.METRICS)].isna().any().any():
        raise ValueError('representation comparison requires all methods, tasks, seeds and metrics')
    units=full.groupby(['regime','endpoint','optimizer_seed','image_id','source_id','method_canonical'],as_index=False,sort=True).agg(**{m:(m,'mean') for m in stats.METRICS},visible_memberships=('visible_seed','nunique'))
    if not (units.loc[units.regime!='LIDC-hard','visible_memberships']==3).all():
        raise ValueError('parent endpoints require three visible choices')
    if not (units.groupby(['regime','optimizer_seed','image_id','source_id']).method_canonical.nunique()==3).all():
        raise ValueError('representation methods do not pair')
    output.mkdir(parents=True)
    records={}
    for stage,regimes in [('stage_a',('LIDC','LIDC-hard','BSDS')),('stage_b',('IMA++','RIGA-disc','RIGA-cup'))]:
        selected=units[units.regime.isin(regimes)]
        boot,draws=stats.bootstrap_means(selected,replicates=replicates,seed=seed)
        effects,gates,decision,critical=stats.analyze_stage(boot,stage)
        directory=output/stage;directory.mkdir()
        selected.to_csv(directory/'units.csv',index=False)
        stats.summary_table(selected).to_csv(directory/'method_summary.csv',index=False)
        effects.to_csv(directory/'paired_effects.csv',index=False)
        gates.to_csv(directory/'decision_gates.csv',index=False)
        records[stage]=dict(decision=decision,critical_values=critical,draws=draws)
    write_json(output/'analysis.json',dict(replicates=replicates,seed=seed,stages=records,evaluations=hashes,evaluation_list_sha256=sha256(evaluations),resampling='accepted representation profile; separate RNG initialization for each stage'))
    return str(output)
