import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from ssrb.data import read_rows, validate_visible, within, verify_inputs, write_json
from ssrb.synthetic import prepare_demo, smoke
from ssrb.training import train


def test_complete_synthetic_pipeline(tmp_path):
    result=smoke(tmp_path/'smoke')
    assert result['synthetic']['effective_k']==8
    root=tmp_path/'smoke'
    semantic=torch.load(root/'semantic/checkpoint.pt',weights_only=True)
    assert semantic['best_epoch']>=5
    history=json.loads((root/'semantic/history.json').read_text())
    assert all(not r['eligible'] for r in history[:4])
    assert (root/'predictions/frozen.json').is_file()
    assert (root/'evaluation/evaluation.json').is_file()
    from ssrb.diagnostics import run as diagnose
    images=root/'prepared/inference'
    report=diagnose(root/'residual/checkpoint.pt',images,images/'images.json',
                    images/'test_features.pt',root/'diagnostics')
    assert report==dict(images=4,heads=8,rows=32)
    assert json.loads((root/'diagnostics/provenance.json').read_text())['reference_masks_read'] is False


@pytest.mark.parametrize('cache_inputs', [False, True])
def test_hidden_changes_do_not_affect_training(tmp_path, cache_inputs):
    from ssrb.inference import infer
    from ssrb.data import sha256
    root=prepare_demo(tmp_path/'demo')
    trainer=root/'prepared/trainer'
    image_root=root/'prepared/inference'
    def run(name):
        sem,hsem=train('semantic',trainer,trainer/'visible.json',root/name/'semantic',config=root/'tiny.json',cache_inputs=cache_inputs)
        raw,hraw=train('residual',trainer,trainer/'visible.json',root/name/'residual',config=root/'tiny.json',anchor=root/name/'semantic/checkpoint.pt',cache_inputs=cache_inputs)
        pred=infer(root/name/'residual/checkpoint.pt',image_root,image_root/'images.json',image_root/'test_features.pt',root/name/'predictions')
        return sem,raw,hsem,hraw,pred
    first=run('a')
    for name in (root/'prepared/evaluator/masks').glob('*.png'):
        name.write_bytes(b'changed hidden data, unreadable as an image')
        name.chmod(0)
    second=run('b')
    assert first[2:4]==second[2:4]
    for i in (0,1):
        assert first[i]['best_epoch']==second[i]['best_epoch']
        assert first[i]['input_seal']==second[i]['input_seal']
        assert all(torch.equal(v,second[i]['state_dict'][k]) for k,v in first[i]['state_dict'].items())
    for key in first[4]:
        assert torch.equal(first[4][key]['probability'],second[4][key]['probability'])
        assert torch.equal(first[4][key]['scores'],second[4][key]['scores'])
    for file in ('semantic/checkpoint.pt','residual/checkpoint.pt','predictions/predictions.pt'):
        assert sha256(root/'a'/file)==sha256(root/'b'/file)


def test_reject_hidden_columns_and_escaping_paths(tmp_path):
    root=prepare_demo(tmp_path/'demo')
    trainer=root/'prepared/trainer'
    rows=read_rows(trainer/'visible.json')
    rows[0]['heldout_mask_paths']=['hidden.png']
    with pytest.raises(ValueError,match='permitted'):
        validate_visible(rows)
    with pytest.raises(ValueError,match='escapes'):
        within(trainer,'../evaluator/hidden.json')
    with pytest.raises(ValueError,match='relative'):
        within(trainer,'/etc/passwd')
    (trainer/'escape').symlink_to(root/'prepared/evaluator')
    with pytest.raises(ValueError,match='escapes'):
        within(trainer,'escape/hidden.json')
    result=subprocess.run([sys.executable,'-I','-m','ssrb.cli','train-semantic','--root',str(trainer),'--manifest',str(trainer/'visible.json'),'--output',str(root/'rejected'),'--hidden-masks','bad'],capture_output=True)
    assert result.returncode!=0
    assert b'unrecognized arguments: --hidden-masks' in result.stderr
    manifest=trainer/'visible.json'
    text=manifest.read_text()
    manifest.write_text(text+' ')
    with pytest.raises(ValueError,match='seal'):
        verify_inputs(trainer,manifest)


def test_kernel_read_isolation(tmp_path):
    allowed=tmp_path/'allowed';allowed.mkdir()
    hidden=tmp_path/'hidden';hidden.mkdir()
    (allowed/'visible.txt').write_text('yes')
    (hidden/'hidden.txt').write_text('no')
    code='''
from pathlib import Path
from ssrb.isolation import restrict_reads
import sys
abi=restrict_reads([sys.prefix,sys.base_prefix,"/usr","/lib","/lib64",sys.argv[1]])
assert (Path(sys.argv[1])/"visible.txt").read_text()=="yes"
try:
    (Path(sys.argv[2])/"hidden.txt").read_text()
except PermissionError:
    print("DENIED",abi)
else:
    raise AssertionError("hidden file remained readable")
'''
    r=subprocess.run([sys.executable,'-I','-c',code,str(allowed),str(hidden)],capture_output=True,text=True)
    if 'Function not implemented' in r.stderr or 'Operation not supported' in r.stderr:
        pytest.skip('kernel does not support Landlock; read isolation cannot be verified')
    assert r.returncode==0,r.stderr
    assert 'DENIED' in r.stdout


def test_isolated_training_command(tmp_path):
    root=prepare_demo(tmp_path/'demo')
    trainer=root/'prepared/trainer'
    config=json.loads((root/'tiny.json').read_text())
    config['workers']=1
    write_json(root/'tiny.json',config)
    result=subprocess.run([sys.executable,'-I','-m','ssrb.cli','train-semantic','--root',str(trainer),
        '--manifest',str(trainer/'visible.json'),'--output',str(root/'isolated'),
        '--config',str(root/'tiny.json'),'--isolate'],capture_output=True,text=True,
        env=os.environ|{'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    if 'Function not implemented' in result.stderr or 'Operation not supported' in result.stderr:
        pytest.skip('kernel does not support Landlock; read isolation cannot be verified')
    assert result.returncode==0,result.stderr
    assert 'Landlock' in result.stdout
    assert (root/'isolated/checkpoint.pt').is_file()


def test_inference_batching_and_metadata(tmp_path):
    from ssrb.inference import infer
    root=tmp_path/'batches'
    smoke(root)
    images=root/'prepared/inference'
    checkpoint=root/'residual/checkpoint.pt'
    one=infer(checkpoint,images,images/'images.json',images/'test_features.pt',root/'batch1',batch_size=1)
    many=infer(checkpoint,images,images/'images.json',images/'test_features.pt',root/'batch12',batch_size=12)
    for key in one:
        torch.testing.assert_close(one[key]['probability'],many[key]['probability'],atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(one[key]['scores'],many[key]['scores'],atol=2e-6,rtol=2e-6)
    metadata=json.loads((root/'batch12/frozen.json').read_text())
    assert metadata['batch_size']==12 and metadata['device']=='cpu'


def test_input_cache_preserves_training_checkpoints_and_predictions(tmp_path):
    from ssrb.inference import infer
    from ssrb.data import sha256
    demo=prepare_demo(tmp_path/'cache_demo');root=demo/'prepared/trainer';images=demo/'prepared/inference'
    histories=[];states=[]
    for name,cached in [('ordinary',False),('cached',True)]:
        sem,h1=train('semantic',root,root/'visible.json',demo/name/'semantic',config=demo/'tiny.json',cache_inputs=cached)
        raw,h2=train('residual',root,root/'visible.json',demo/name/'residual',config=demo/'tiny.json',
                     anchor=demo/name/'semantic/checkpoint.pt',cache_inputs=cached)
        infer(demo/name/'residual/checkpoint.pt',images,images/'images.json',images/'test_features.pt',demo/name/'predictions')
        histories.append((h1,h2));states.append((sem,raw))
    assert histories[0]==histories[1]
    for stage in range(2):
        assert all(torch.equal(value,states[1][stage]['state_dict'][key]) for key,value in states[0][stage]['state_dict'].items())
    for filename in ('semantic/checkpoint.pt','residual/checkpoint.pt','predictions/predictions.pt'):
        assert sha256(demo/'ordinary'/filename)==sha256(demo/'cached'/filename)
    run=json.loads((demo/'cached/residual/run.json').read_text())
    assert run['cache_inputs'] and all(v>0 for v in run['cached_input_bytes'].values())
