import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from ssrb import assignment_cluster_statistics as cluster, assignment_seed_statistics as seeds
from ssrb.assignment_statistics import validate_grid
from ssrb.paper_assignment_plot import choose_panels,portable_frame


def test_original_assignment_intervals_and_example_selection():
    reference=json.loads((Path(__file__).parent/'fixtures/assignment_reference.json').read_text())
    frame=pd.DataFrame(reference['rows']);single=frame[frame.optimizer_seed==0]
    for row in reference['cluster']:
        kwargs=dict(method_a='dmb_rn',method_b=row['comparator'],metric='MLQ@8')
        units=cluster.pooled_riga_units(single,**kwargs) if row['regime']=='RIGA-pooled' else cluster.paired_units(single,regime=row['regime'],**kwargs)
        actual=cluster.cluster_bootstrap(units,replicates=reference['replicates'],seed=3450)
        for key,value in row['values'].items():
            assert actual[key]==pytest.approx(value,abs=1e-12)
    _,actual,_=seeds.analyze(frame[frame.endpoint=='lidc'],bootstrap_replicates=reference['replicates'],bootstrap_seed=3664,delta=.005)
    expected=pd.DataFrame(reference['lidc_effects'])
    numeric=expected.select_dtypes(include=[np.number]).columns
    np.testing.assert_allclose(actual[numeric],expected[numeric],atol=1e-12,rtol=0)
    assert actual.decision.tolist()==expected.decision.tolist()
    assert choose_panels(single)[1]==reference['panels']
    frame['method_canonical']=frame.variant
    with pytest.raises(ValueError,match='incomplete'):
        validate_grid(frame[~((frame.variant=='capacity_wta') & (frame.optimizer_seed==2))],seeds.VARIANTS,('lidc',),range(3))


def test_portable_endpoints_preserve_original_example_selection(tmp_path):
    reference=json.loads((Path(__file__).parent/'fixtures/assignment_reference.json').read_text())
    frame=pd.DataFrame(reference['rows']);frame=frame[(frame.optimizer_seed==0)&frame.method.isin(['dmb_rn','dmb_wta'])].copy()
    frame.loc[frame.subset=='visible_dice_lt_0.80','endpoint']='lidc_hard'
    frames=[]
    for method,group in frame.groupby('method'):
        directory=tmp_path/method;directory.mkdir()
        columns=['endpoint','visible_seed','image_id','source_id','MLQ@8']
        (directory/'per_image.json').write_text(json.dumps(group[columns].to_dict('records')))
        frames.append(portable_frame(directory,method))
    panels=choose_panels(pd.concat(frames,ignore_index=True))[1]
    for panel,expected in reference['panels'].items():
        assert panels[panel]['source_id']==expected['source_id']
        assert panels[panel]['source_effect']==expected['source_effect']
        assert [r['image_id'] for r in panels[panel]['images']]==[r['image_id'] for r in expected['images']]
