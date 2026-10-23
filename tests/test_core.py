import itertools
import json
from pathlib import Path
import pytest
import torch
from ssrb.models import DirectMaskBank, FrozenDinoRawResidual, KHeadValidityUNet
from ssrb.losses import assignment_weights, objective, pair_cost
from ssrb import metrics


def test_published_parameter_counts():
    a = DirectMaskBank(layers=3,dim=768,queries=8)
    assert sum(p.numel() for p in a.parameters()) == 466707
    assert sum(p.numel() for p in KHeadValidityUNet(heads=8,base=24).parameters()) == 1094512


def test_residual_freeze_and_alignment():
    torch.manual_seed(4)
    a = DirectMaskBank(layers=3,dim=8,queries=8,hidden=16)
    f = FrozenDinoRawResidual(a,heads=8,base=2).train()
    x, im = torch.randn(2,3,4,8), torch.randn(2,3,16,16)
    before = {k: v.clone() for k, v in a.state_dict().items()}
    m, r = a(x)
    z, q = f(x,im)
    assert torch.equal(z,torch.nn.functional.interpolate(m,size=(16,16),mode='bilinear',align_corners=False))
    assert torch.equal(r,q)
    assert not f.anchor.training
    loss, _ = objective(z,q,(torch.rand(2,16,16)>.5).float())
    opt = torch.optim.Adam(f.raw.parameters(),lr=.001)
    loss.backward()
    assert all(p.grad is None for p in a.parameters())
    assert f.raw.decoder.out.weight.grad.abs().sum() > 0
    assert f.raw.validity[-1].weight.grad.abs().sum() > 0
    opt.step()
    assert all(torch.equal(v,a.state_dict()[k]) for k,v in before.items())
    f.eval()
    z, q = f(x,im)
    raw = f.raw(im)
    assert torch.equal(z,torch.nn.functional.interpolate(a(x)[0],size=(16,16),mode='bilinear',align_corners=False)+raw['mask_logits'])
    assert torch.equal(q,a(x)[1]+raw['validity_logits'])
    with pytest.raises(ValueError,match='head counts'):
        FrozenDinoRawResidual(a,heads=7,base=2)


@pytest.mark.parametrize('batch,heads,offset',[(4,2,0),(5,3,1),(2,8,4),(9,8,7)])
def test_capacity_including_remainder(batch,heads,offset):
    costs = torch.arange(batch*heads).reshape(batch,heads).float().sin()
    w, winners = assignment_weights(costs,capacity_offset=offset)
    assert torch.equal(w.sum(1),torch.ones(batch))
    base, rem = divmod(batch,heads)
    expected = torch.full((heads,),base)
    expected[(torch.arange(rem)+offset)%heads] += 1
    assert torch.equal(torch.bincount(winners,minlength=heads),expected)
    if batch <= 5:
        feasible = [v for v in itertools.product(range(heads),repeat=batch)
                    if torch.equal(torch.bincount(torch.tensor(v),minlength=heads),expected)]
        optimum = min(sum(float(costs[i,v[i]]) for i in range(batch)) for v in feasible)
        assert abs(float((w*costs).sum())-optimum)<1e-6
    assert torch.equal(w,assignment_weights(costs,capacity_offset=offset)[0])


def test_cost_ties_and_known_loss():
    costs = torch.zeros(5,3)
    first = assignment_weights(costs,capacity_offset=1)[1]
    assert torch.equal(first,assignment_weights(costs,capacity_offset=1)[1])
    assert torch.equal(assignment_weights(costs,method='hard_mcl')[1],torch.zeros(5,dtype=torch.long))
    logits = torch.zeros(2,8,2,2)
    target = torch.zeros(2,2,2)
    expected = torch.log(torch.tensor(2.)) + 2/3
    torch.testing.assert_close(pair_cost(logits,target),torch.full((2,8),expected))
    loss,_ = objective(logits,torch.zeros(2,8),target)
    torch.testing.assert_close(loss,expected+.3*torch.log(torch.tensor(2.)))


def test_numerical_model_reference():
    ref = json.loads((Path(__file__).parent/'fixtures/model_reference.json').read_text())
    torch.manual_seed(17)
    a = DirectMaskBank(layers=3,dim=8,queries=8,hidden=16,dropout=0.)
    f = FrozenDinoRawResidual(a,heads=8,base=2).eval()
    torch.manual_seed(19)
    for layer in (f.raw.decoder.out,f.raw.validity[-1]):
        torch.nn.init.normal_(layer.weight,std=.03)
        torch.nn.init.normal_(layer.bias,std=.03)
    torch.manual_seed(20)
    x=torch.randn(2,3,4,8)
    im=torch.randn(2,3,16,16)
    target=(torch.rand(2,2,2)>.5).float()
    with torch.no_grad():
        sm,sr=a(x)
        raw=f.raw(im)
        zm,zr=f(x,im)
    outputs=dict(semantic_masks=sm,semantic_ranks=sr,raw_masks=raw['mask_logits'],raw_ranks=raw['validity_logits'],fused_masks=zm,fused_ranks=zr,pair_cost=pair_cost(sm,target))
    for key,value in outputs.items():
        torch.testing.assert_close(value,torch.tensor(ref[key]),rtol=1e-6,atol=1e-7)
    loss,_=objective(sm,sr,target,capacity_offset=3)
    # CPU reduction order can move this scalar by one float32 ULP.
    # Keep the original reference and allow two rounding units at this scale.
    torch.testing.assert_close(loss, torch.tensor(ref['loss']), rtol=0,
                               atol=2*torch.finfo(torch.float32).eps)
    torch.manual_seed(21)
    pred=torch.rand(8,5,7)>.5
    refs=torch.rand(3,5,7)>.5
    assert metrics.image_metrics(pred,refs)==ref['metrics']


def test_native_grid_metrics_and_serialization(tmp_path):
    empty=torch.zeros(1,4,5,dtype=torch.bool)
    assert metrics.dice_matrix(empty,empty).item()==1
    masks=torch.zeros(8,4,5,dtype=torch.bool)
    masks[0,:2]=True
    masks[1,2:]=True
    references=masks[:2]
    a=metrics.image_metrics(masks,references)
    b=metrics.image_metrics(masks.flip(0),references)
    assert a['MLQ@8']==b['MLQ@8']==1
    assert a['MLQ@1']!=b['MLQ@1']
    assert a['Waste@8']==.75
    assert metrics.unique_clusters(masks)==3
    for mask in masks:
        assert torch.equal(mask,metrics.decode_binary_rle(metrics.encode_binary_rle(mask)))
    p=torch.randn(8,3,4).sigmoid()
    torch.save(p,tmp_path/'p.pt')
    q=torch.load(tmp_path/'p.pt',weights_only=True)
    assert torch.equal(metrics.threshold_probability(metrics.upsample_probability(p,(9,11))),
                       metrics.threshold_probability(metrics.upsample_probability(q,(9,11))))
