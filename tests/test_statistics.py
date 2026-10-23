import numpy as np
from ssrb.statistics import cluster_draws, resample_means


def test_parent_subset_and_fundus_pairing():
    sources = {'lidc':['p1','p2','p3'], 'lidc_hard':['p1','p3'],
               'riga_disc':['f1','f2'], 'riga_cup':['f1','f2']}
    draws, optimizer = cluster_draws(sources,replicates=20)
    assert optimizer.shape==(20,3)
    for p in sources['lidc_hard']:
        assert np.array_equal(draws['lidc'][p],draws['lidc_hard'][p])
    for f in sources['riga_disc']:
        assert np.array_equal(draws['riga_disc'][f],draws['riga_cup'][f])
    assert np.array_equal(sum(draws['lidc'].values()),np.full(20,3))
    # Every image in a patient shares one weight, including multiple nodules.
    sums=np.ones((3,3,2,1))*5
    counts=np.ones((3,3,2))
    actual=resample_means(sums,counts,sources['lidc'],draws['lidc'],optimizer)
    assert np.all(actual==5)


def test_accepted_bootstrap_parity_and_command(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path
    import pandas as pd
    from ssrb.legacy_statistics import bootstrap_means, effect_row, simultaneous
    fixture=json.loads((Path(__file__).parent/'fixtures/statistics_reference.json').read_text())
    units=pd.DataFrame(fixture['units'])
    actual,_=bootstrap_means(units,replicates=37,seed=20260814)
    rows,samples=[],[]
    for regime,info in actual.items():
        assert np.array_equal(info['point'],np.asarray(fixture['bootstrap'][regime]['point']))
        assert np.array_equal(info['bootstrap'],np.asarray(fixture['bootstrap'][regime]['bootstrap']))
        row,sample=effect_row(actual,regime,'dino_capacity','MLQ@8')
        rows.append(row);samples.append(sample)
    assert np.isfinite(simultaneous(rows,samples))
    assert all(r['simultaneous_ci95_low']<=r['effect']<=r['simultaneous_ci95_high'] for r in rows)
    units.to_csv(tmp_path/'units.csv',index=False)
    result=subprocess.run([sys.executable,'-I','-m','ssrb.cli','bootstrap-representation','--units',str(tmp_path/'units.csv'),'--output',str(tmp_path/'bootstrap.json')],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    value=json.loads((tmp_path/'bootstrap.json').read_text())
    assert len(value['BSDS']['bootstrap'])==10000
