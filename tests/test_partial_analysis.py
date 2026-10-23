import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ssrb.analysis import run
from ssrb.data import write_json


def test_partial_analysis_matches_original_crossed_bootstrap(tmp_path):
    reference=json.loads((Path(__file__).parent/'fixtures/partial_analysis_reference.json').read_text())
    full=pd.DataFrame(reference['full'])
    parent=full.regime.map({'LIDC':'lidc','LIDC-hard':'lidc','BSDS':'bsds'})
    specs=[]
    for index,((endpoint,method,seed),group) in enumerate(full.groupby([parent,'method','optimizer_seed'])):
        directory=tmp_path/f'evaluation_{index}'
        directory.mkdir()
        rows=[]
        for value in group.to_dict('records'):
            value['endpoint']={'LIDC':'lidc','LIDC-hard':'lidc_hard','BSDS':'bsds'}[value.pop('regime')]
            value.pop('method');value.pop('optimizer_seed')
            rows.append(value)
        write_json(directory/'per_image.json',rows)
        specs.append(dict(evaluation=str(directory),method=method,optimizer_seed=int(seed)))
    manifest=tmp_path/'evaluations.json'
    write_json(manifest,specs)
    run(manifest,tmp_path/'analysis',profile='partial',replicates=reference['replicates'])
    actual=pd.read_csv(tmp_path/'analysis/paired_effects.csv')
    expected=pd.DataFrame(reference['effects'])
    keys=['contrast','left','right','regime','metric']
    paired=actual.merge(expected,on=keys,suffixes=('_actual','_original'),validate='one_to_one')
    assert len(paired)==len(actual)==len(expected)==30
    for field in ['effect','ci95_low','ci95_high','simultaneous_ci95_low','simultaneous_ci95_high']:
        np.testing.assert_allclose(paired[field+'_actual'],paired[field+'_original'],atol=1e-12,rtol=0,equal_nan=True)
    assert json.loads((tmp_path/'analysis/analysis.json').read_text())['seed']==reference['seed']
    # Removing a whole optimizer/parent cell must fail before computing intervals.
    write_json(manifest,[s for s in specs if not (s['method']=='ssrb' and s['optimizer_seed']==2)])
    with pytest.raises(ValueError,match='incomplete parent'):
        run(manifest,tmp_path/'incomplete',profile='partial',replicates=reference['replicates'])
