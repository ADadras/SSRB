import json

import pytest

from ssrb.comparisons import train
from ssrb.data import write_json
from ssrb.runtime import benchmark
from ssrb.synthetic import prepare_demo


def test_runtime_smoke_executes_real_bank_and_preserves_protocol_label(tmp_path):
    demo=prepare_demo(tmp_path/'demo');trainer=demo/'prepared/trainer';images=demo/'prepared/inference'
    config=demo/'tiny_raw.json'
    write_json(config,dict(base=2,image_size=32,epochs=1,patience=1,batch_size=4,workers=0))
    train('raw_capacity',trainer,trainer/'visible.json',demo/'raw',config=config)
    record=benchmark(images,images/'images.json',demo/'timing',method='raw_capacity',
        checkpoint=demo/'raw/checkpoint.pt',device='cpu',subset_size=2,warmup_runs=1,timed_runs=3,smoke=True)
    assert record['status']=='smoke-complete'
    assert record['complexity']['precision']=='float32'
    assert record['timing']['warmup_runs']==1 and record['timing']['timed_runs']==3
    assert record['timing']['seconds_per_run_mean']>0
    assert len(json.loads((demo/'timing/samples.json').read_text()))==3
    assert record['subset_size']==2
    assert json.loads((demo/'timing/progress.json').read_text())==dict(phase='timed',completed=3,total=3)
    with pytest.raises(ValueError,match='recorded sources'):
        benchmark(images,images/'images.json',demo/'not_paper',method='raw_capacity',
                  checkpoint=demo/'raw/checkpoint.pt',device='cpu',warmup_runs=1,timed_runs=3)
    with pytest.raises(RuntimeError,match='NVIDIA A40'):
        benchmark(images,images/'images.json',demo/'wrong_device',method='raw_capacity',
                  checkpoint=demo/'raw/checkpoint.pt',device='cpu')
