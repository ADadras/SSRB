import torch
import numpy as np
from ssrb.proposals import rank_candidates, native_metrics
from ssrb.partial import configure_last_block, quantized_features


def test_proposal_ranking_and_empty_bank():
    candidates=[dict(predicted_iou=.9,stability_score=.8,original_index=2),
                dict(predicted_iou=.9,stability_score=.9,original_index=1),
                dict(predicted_iou=.9,stability_score=.9,original_index=0)]
    assert [r['original_index'] for r in rank_candidates(candidates)]==[0,1,2]
    references=torch.ones(2,8,8,dtype=torch.bool)
    zero=native_metrics(torch.empty(0,8,8),references)
    assert zero['effective_k']==0 and zero['MLQ@8']==0 and zero['Waste_native']==1
    two=native_metrics(torch.ones(2,8,8,dtype=torch.bool),references)
    assert two['effective_k']==2 and two['MLQ_native']==1 and np.isnan(two['MLQ@8'])


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks=torch.nn.ModuleList([torch.nn.Linear(3,3) for _ in range(12)])
        self.seen=[]
    def get_intermediate_layers(self,image,n,**kwargs):
        self.seen.append(len(image));result=[]
        for j,block in enumerate(self.blocks):
            image=block(image)
            if j in n:result.append((image,torch.zeros(1)))
        return result


def test_partial_block_scope_microbatch_and_gradients():
    encoder=Encoder()
    names=configure_last_block(encoder)
    assert names and all(n.startswith('blocks.11.') for n in names)
    assert not encoder.training
    images=torch.randn(32,4,3)
    features=quantized_features(encoder,images,16)
    assert encoder.seen==[16,16] and features.shape==(32,3,4,3)
    features.sum().backward()
    assert encoder.blocks[11].weight.grad.abs().sum()>0
    assert all(p.grad is None for block in encoder.blocks[:11] for p in block.parameters())
    with torch.no_grad():
        encoder.seen=[]
        full=quantized_features(encoder,images,32)
    torch.testing.assert_close(features,full,atol=0,rtol=0)


def test_feature_extraction_two_backbones_multiple_batches(tmp_path, monkeypatch):
    from ssrb.features import extract
    from ssrb.synthetic import prepare_demo
    from ssrb.data import read_rows
    demo=prepare_demo(tmp_path/'demo');root=demo/'prepared/inference'
    weights=tmp_path/'weights.pt';torch.save({},weights)
    class FakeDino(torch.nn.Module):
        def __init__(self,backbone):
            super().__init__();self.backbone=backbone;self.calls=0
        def get_intermediate_layers(self,images,n,**kwargs):
            assert n==([5,8,11] if self.backbone=='dinov3' else [3,6,11])
            self.calls+=1
            shape=(len(images),784,768) if self.backbone=='dinov3' else (len(images),1024,384)
            return [(torch.ones(shape)*(i + .1234567),torch.zeros(len(images),shape[-1])) for i in n]
    for backbone in ('dinov3','dinov2'):
        model=FakeDino(backbone)
        monkeypatch.setattr(torch.hub,'load',lambda *args,**kwargs:model)
        extract(root,root/'images.json',tmp_path/backbone,tmp_path,weights,device='cpu',batch_size=1,backbone=backbone)
        payload=torch.load(tmp_path/backbone/'test_features.pt',weights_only=True)
        assert model.calls==len(read_rows(root/'images.json'))==4
        assert payload['grid_hw']==((28,28) if backbone=='dinov3' else (32,32))
        assert set(payload)=={'features','image_ids','grid_hw'}
        expected_dtype = torch.float16 if backbone == 'dinov3' else torch.float32
        assert payload['features'].dtype == expected_dtype
        reference = torch.tensor((5 if backbone == 'dinov3' else 3) + .1234567, dtype=expected_dtype)
        assert payload['features'][0, 0, 0, 0] == reference


def test_shared_cache_filter_and_hidden_manifest_rejection(tmp_path):
    import pytest
    from ssrb.features import import_cache
    from ssrb.data import read_rows,write_json
    from ssrb.synthetic import prepare_demo
    demo=prepare_demo(tmp_path/'demo');root=demo/'prepared/inference'
    value=torch.load(root/'test_features.pt',weights_only=True)
    expected=value['features'].clone()
    value['features']=torch.cat([value['features'],torch.zeros_like(value['features'][:1])])
    value['image_ids'].append('another_endpoint')
    shared=tmp_path/'test_features.pt';torch.save(value,shared)
    imported=tmp_path/'filtered/test_features.pt'
    import_cache(shared,imported,root/'images.json',split='test')
    actual=torch.load(imported,weights_only=True)
    assert actual['image_ids']==[r['image_id'] for r in sorted(read_rows(root/'images.json'),key=lambda r:r['image_id'])]
    assert torch.equal(actual['features'],expected)
    with pytest.raises(ValueError):
        import_cache(shared,tmp_path/'bad.pt',demo/'prepared/evaluator/hidden.json',split='test')
    rows=read_rows(root/'images.json');rows[0]['image_id']='missing'
    write_json(tmp_path/'missing.json',rows)
    with pytest.raises(ValueError,match='missing'):
        import_cache(shared,tmp_path/'missing.pt',tmp_path/'missing.json',split='test')


def test_binary_rle_predictions_preserve_native_evaluation(tmp_path):
    import json
    from ssrb.data import pack_binary_prediction,prediction_probability,read_rows,write_json,sha256
    from ssrb.evaluation import evaluate
    from ssrb.synthetic import prepare_demo
    demo=prepare_demo(tmp_path/'demo');prepared=demo/'prepared'
    rows=read_rows(prepared/'inference/images.json')
    dense={};encoded={};generator=torch.Generator().manual_seed(3550)
    for r,k in zip(rows,(0,1,4,8),strict=True):
        masks=torch.rand(k,32,32,generator=generator)>.7
        scores=torch.arange(k,dtype=torch.float32)
        dense[r['image_id']]=dict(probability=masks,scores=scores)
        encoded[r['image_id']]=pack_binary_prediction(masks,scores)
        assert torch.equal(prediction_probability(encoded[r['image_id']]),masks)
    for name,bank in [('dense',dense),('rle',encoded)]:
        output=tmp_path/name;output.mkdir();torch.save(bank,output/'predictions.pt')
        write_json(output/'frozen.json',dict(predictions_sha256=sha256(output/'predictions.pt'),binary_proposals=True))
        evaluate(output,prepared/'evaluator',prepared/'evaluator/hidden.json',tmp_path/(name+'_eval'))
    for filename in ('per_image.json','summary.json'):
        assert (tmp_path/'dense_eval'/filename).read_bytes()==(tmp_path/'rle_eval'/filename).read_bytes()
    assert prediction_probability(encoded[rows[0]['image_id']]).shape==(0,32,32)


def test_combined_manifest_preserves_split_batching(tmp_path, monkeypatch):
    import json
    from ssrb.features import extract
    from ssrb.synthetic import prepare_demo
    from ssrb.data import read_rows, write_json
    demo=prepare_demo(tmp_path/'demo');root=demo/'prepared/inference'
    original=read_rows(root/'images.json')
    rows=[dict(original[j%len(original)],image_id=f'image_{j:02d}',split=('train' if j%2==0 else 'val')) for j in range(6)]
    write_json(tmp_path/'combined.json',rows)
    weights=tmp_path/'weights.pt';torch.save({},weights)
    class BatchSensitiveEncoder(torch.nn.Module):
        def get_intermediate_layers(self,images,n,**kwargs):
            # A batch-dependent output exposes accidental changes in batch shape.
            return [torch.full((len(images),1024,384),float(len(images))) for _ in n]
    monkeypatch.setattr(torch.hub,'load',lambda *args,**kwargs:BatchSensitiveEncoder())
    extract(root,tmp_path/'combined.json',tmp_path/'combined',tmp_path,weights,device='cpu',batch_size=2,backbone='dinov2')
    for split in ('train','val'):
        write_json(tmp_path/(split+'.json'),[r for r in rows if r['split']==split])
        extract(root,tmp_path/(split+'.json'),tmp_path/split,tmp_path,weights,device='cpu',batch_size=2,backbone='dinov2')
        combined=torch.load(tmp_path/'combined'/f'{split}_features.pt',weights_only=True)
        separate=torch.load(tmp_path/split/f'{split}_features.pt',weights_only=True)
        assert combined['image_ids']==separate['image_ids']
        assert torch.equal(combined['features'],separate['features'])
        assert combined['features'][:,0,0,0].tolist()==[2.,2.,1.]
    provenance=json.loads((tmp_path/'combined/feature_provenance.json').read_text())
    assert provenance['batch_size']==2 and provenance['batches_separated_by_split']
    assert provenance['execution']['torch']==str(torch.__version__)


def test_feature_reference_guard_detects_changed_values_and_order(tmp_path, capsys):
    import pytest
    from ssrb.features import fingerprint, verify
    from ssrb.cli import main
    from ssrb.data import write_json
    payload=dict(features=torch.arange(24,dtype=torch.float32).reshape(2,3,4,1),image_ids=['a','b'],grid_hw=(2,2))
    source=tmp_path/'features.pt';torch.save(payload,source)
    entry=dict(endpoint='test',split='train',backbone='dinov2',**fingerprint(source))
    reference=tmp_path/'reference.json';write_json(reference,dict(entries=[entry]))
    assert verify(source,reference,'test','train','dinov2')['status']=='verified-reference-feature-cache'
    main(['verify-features','--source',str(source),'--reference',str(reference),'--endpoint','test','--split','train','--backbone','dinov2'])
    assert 'verified-reference-feature-cache' in capsys.readouterr().out
    changed=payload|dict(features=payload['features'].flip(0))
    torch.save(changed,source)
    with pytest.raises(ValueError,match='tensor_sha256'):
        verify(source,reference,'test','train','dinov2')
    torch.save(payload|dict(image_ids=['b','a']),source)
    with pytest.raises(ValueError,match='image_ids_sha256'):
        verify(source,reference,'test','train','dinov2')
    with pytest.raises(ValueError,match='one verified reference'):
        verify(source,reference,'test','val','dinov2')


def test_explicit_dinov3_weights_override_existing_hub_cache(tmp_path, monkeypatch):
    from ssrb.features import load_local_dinov3
    from pathlib import Path
    weights = tmp_path/'approved'/'weights.pth'
    weights.parent.mkdir()
    weights.write_bytes(b'approved local weights')
    previous = torch.hub.get_dir()
    stale = tmp_path/'old_cache'
    (stale/'checkpoints').mkdir(parents=True)
    (stale/'checkpoints'/weights.name).write_bytes(b'other weights with the same filename')
    torch.hub.set_dir(str(stale))

    def upstream(repo, model, **kwargs):
        assert kwargs['weights'] == str(weights.resolve())
        cached = Path(torch.hub.get_dir())/'checkpoints'/weights.name
        assert cached.read_bytes() == weights.read_bytes()
        assert not cached.is_relative_to(stale)
        return 'loaded the supplied weights'

    monkeypatch.setattr(torch.hub, 'load', upstream)
    try:
        assert load_local_dinov3(tmp_path, weights) == 'loaded the supplied weights'
        assert torch.hub.get_dir() == str(stale)
        assert (stale/'checkpoints'/weights.name).read_bytes().startswith(b'other weights')
    finally:
        torch.hub.set_dir(previous)
