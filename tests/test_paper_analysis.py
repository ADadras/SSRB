import json
import pandas as pd
import numpy as np
from ssrb import controls_statistics as stats
from ssrb.figures import results


def test_controls_known_paired_constant_and_vector_plot(tmp_path):
    rows=[]
    for regime in ('LIDC','LIDC-hard','IMA++','BSDS','RIGA-disc','RIGA-cup'):
        for optimizer in range(3):
            for image in range(5):
                for method in stats.METHODS:
                    value=.4+.01*image+.001*optimizer+(0.05 if method=='ssrb' else 0.)
                    rows.append(dict(regime=regime,source_id=f's{image//2}',image_id=f'i{image}',
                                     optimizer_seed=optimizer,method_canonical=method,
                                     **{metric:value for metric in stats.METRICS}))
    units=pd.DataFrame(rows)
    boot=stats.bootstrap(units,replicates=20,seed=20260815)
    effects=stats.effect_table(boot)
    np.testing.assert_allclose(effects.effect,.05,atol=1e-14)
    np.testing.assert_allclose(effects.simultaneous_ci95_low,.05,atol=1e-14)
    frame=units.groupby(['regime','method_canonical'],as_index=False)[list(stats.METRICS)].mean()
    filename=tmp_path/'summary.csv';frame.to_csv(filename,index=False)
    results(filename,tmp_path/'plots')
    assert len(list((tmp_path/'plots').glob('*.pdf')))==6
    assert all(p.read_bytes().startswith(b'%PDF') for p in (tmp_path/'plots').glob('*.pdf'))
    assert json.loads((tmp_path/'plots/provenance.json').read_text())['summary_sha256']


def test_full_analysis_keeps_hard_cases_with_fewer_visible_choices(tmp_path):
    from ssrb.analysis import run
    import pytest
    methods=stats.METHODS
    specifications=[]
    for endpoint in ('lidc','bsds','ima','riga_disc','riga_cup'):
        for method in methods:
            for optimizer in range(3):
                for visible in range(3):
                    rows=[]
                    for image in range(2):
                        value=.4+.01*image+.001*optimizer+.02*visible+(.05 if method=='ssrb' else 0.)
                        row=dict(endpoint=endpoint,visible_seed=visible,image_id=f'i{image}',source_id=f's{image}',
                                 **{metric:value for metric in stats.METRICS})
                        rows.append(row)
                        if endpoint=='lidc' and (image==0 or visible==0):
                            rows.append(row|dict(endpoint='lidc_hard'))
                    directory=tmp_path/f'{endpoint}_{method}_{optimizer}_{visible}';directory.mkdir()
                    (directory/'per_image.json').write_text(json.dumps(rows))
                    specifications.append(dict(evaluation=str(directory),method=method,optimizer_seed=optimizer))
    spec=tmp_path/'evaluations.json';spec.write_text(json.dumps(specifications))
    run(spec,tmp_path/'analysis',profile='controls',replicates=20)
    units=pd.read_csv(tmp_path/'analysis/units.csv')
    hard=units[(units.regime=='LIDC-hard') & (units.method_canonical=='ssrb')]
    assert set(hard.visible_memberships)=={1,3}
    assert abs(hard['MLQ@8'].mean()-.466)<1e-14
    spec.write_text(json.dumps(specifications[1:]))
    with pytest.raises(ValueError,match='three visible|incomplete'):
        run(spec,tmp_path/'incomplete',profile='controls',replicates=20)
