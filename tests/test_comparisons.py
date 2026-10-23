import json
import torch
import pytest
from ssrb.data import write_json
from ssrb.synthetic import prepare_demo
from ssrb.training import train as core_train
from ssrb.comparisons import train, infer, fixed_validation_noise, configuration
from ssrb.posthoc import lexicographic_assignment, combine, align, infer as infer_posthoc
from ssrb.evaluation import evaluate


def test_all_standalone_comparisons_and_posthoc(tmp_path):
    demo = prepare_demo(tmp_path/'demo')
    root = demo/'prepared/trainer'
    test = demo/'prepared/inference'
    core_train('semantic', root, root/'visible.json', demo/'semantic', config=demo/'tiny.json')
    with pytest.raises(ValueError,match='anchor'):
        core_train('residual',root,root/'visible.json',demo/'wrong_core_seed',
                   config=demo/'tiny.json',anchor=demo/'semantic/checkpoint.pt',seed=1)
    with pytest.raises(ValueError,match='anchor'):
        train('conventional_residual',root,root/'visible.json',demo/'wrong_control_seed',
              anchor=demo/'semantic/checkpoint.pt',seed=1)
    config = demo/'comparison.json'
    write_json(config, dict(base=2, hidden=16, image_size=32, epochs=2, patience=1,
                           batch_size=4, workers=0, warmup_epochs=1))
    for method in ('raw_capacity', 'wide_raw', 'conventional_residual', 'probunet',
                   'dino_wta', 'dino_soft_wta', 'dino_rank_neighborhood'):
        ck, history = train(method, root, root/'visible.json', demo/method, config=config,
                            anchor=demo/'semantic/checkpoint.pt')
        assert len(history) == 2
        assert ck['best_epoch'] > (1 if method == 'dino_rank_neighborhood' else 0)
        prediction = infer(demo/method/'checkpoint.pt', test, test/'images.json', demo/(method+'_pred'),
                           feature_file=test/'test_features.pt')
        assert all(v['probability'].shape == (8, 2, 2) if method.startswith('dino_')
                   else v['probability'].shape == (8, 32, 32) for v in prediction.values())
        summary = evaluate(demo/(method+'_pred'), demo/'prepared/evaluator',
                           demo/'prepared/evaluator/hidden.json', demo/(method+'_eval'))
        assert summary['synthetic']['effective_k'] == 8
    alignment = demo/'alignment.json'
    result = align(demo/'semantic/checkpoint.pt', demo/'raw_capacity/checkpoint.pt', root,
                   root/'visible.json', alignment)
    assert sorted(result['permutation']) == list(range(8))
    for mode in ('additive', 'union'):
        prediction = infer_posthoc(demo/'semantic/checkpoint.pt', demo/'raw_capacity/checkpoint.pt', test,
                                   test/'images.json', test/'test_features.pt', demo/mode,
                                   alignment=alignment if mode=='additive' else None)
        assert len(prediction) == 4
        assert all(v['probability'].shape == (8, 32, 32) for v in prediction.values())
    # Stochastic inference keeps the same seed/model-construction/loader sequence.
    again = infer(demo/'probunet/checkpoint.pt', test, test/'images.json', demo/'repeat')
    original = torch.load(demo/'probunet_pred/predictions.pt', weights_only=True)
    assert all(torch.equal(again[k]['probability'], original[k]['probability']) for k in again)


def test_assignment_ties_and_union_order():
    assert lexicographic_assignment(torch.zeros(8, 8))[0] == list(range(8))
    cost = torch.ones(8, 8)
    cost[torch.arange(8), torch.arange(7, -1, -1)] = 0
    assert lexicographic_assignment(cost)[0] == list(range(7, -1, -1))
    a = torch.arange(8.).view(1, 8, 1, 1)
    b = -a
    ar = torch.arange(8.).view(1, 8)
    br = -ar
    probability, _ = combine(a, ar, b, br)
    torch.testing.assert_close(probability[:, :4], b[:, :4].sigmoid())
    torch.testing.assert_close(probability[:, 4:], a[:, [7, 6, 5, 4]].sigmoid())
    # Distinct large logits can round to identical sigmoid probabilities.
    # The accepted union chooses each top four from logits before concatenation.
    saturated = torch.arange(20., 28.).view(1, 8)
    assert torch.equal(saturated.sigmoid(), torch.ones_like(saturated))
    probability, _ = combine(a, saturated, b, saturated)
    torch.testing.assert_close(probability[:, :4], b[:, [7, 6, 5, 4]].sigmoid())
    torch.testing.assert_close(probability[:, 4:], a[:, [7, 6, 5, 4]].sigmoid())
    with pytest.raises(ValueError):
        combine(a, ar, b, br, permutation=[0]*8)


def test_diffusion_validation_determinism():
    a, b = fixed_validation_noise('synthetic', image_size=16, diffusion_steps=1000)
    c, d = fixed_validation_noise('synthetic', image_size=16, diffusion_steps=1000)
    assert a == c and torch.equal(b, d)
    assert configuration('cimd')['batch_size'] == 20
    assert configuration('wide_raw')['base'] == 29


def test_original_baseline_numerical_reference():
    from pathlib import Path
    import numpy as np
    from comparison_fixture import probunet_case, cimd_case
    from ssrb.probabilistic import ProbUNet, dice_loss, kl_normal
    from ssrb.cimd import training_objective
    reference=json.loads((Path(__file__).parent/'fixtures/comparison_reference.json').read_text())
    def compare(actual,expected):
        if isinstance(expected,dict):
            assert set(actual)==set(expected)
            for key in expected:compare(actual[key],expected[key])
        else:
            np.testing.assert_allclose(actual,expected,atol=2e-7,rtol=2e-6)
    compare(probunet_case(ProbUNet,dice_loss,kl_normal),reference['probunet'])
    compare(cimd_case(training_objective),reference['cimd'])
    t,noise=fixed_validation_noise('synthetic',image_size=16,diffusion_steps=1000)
    assert t==reference['noise']['timestep']
    np.testing.assert_array_equal(noise.flatten()[:12].numpy(),reference['noise']['values'])


def test_single_dropout_and_fixed_extents(tmp_path):
    from ssrb.extent_banks import extents, ensemble
    from ssrb.metrics import ged2
    from ssrb.data import sha256
    demo=prepare_demo(tmp_path/'demo');root=demo/'prepared/trainer';test=demo/'prepared/inference'
    config=demo/'single.json'
    write_json(config,dict(hidden=16,epochs=2,patience=1,batch_size=4,workers=0))
    for method in ('dino_single','dino_mc_dropout'):
        ck,_=train(method,root,root/'visible.json',demo/method,config=config)
        assert ck['model_args']['queries']==1
        value=infer(demo/method/'checkpoint.pt',test,test/'images.json',demo/(method+'_pred'),feature_file=test/'test_features.pt')
        assert all(len(v['probability'])==(8 if method=='dino_mc_dropout' else 1) for v in value.values())
    dirs=[]
    single=torch.load(demo/'dino_single_pred/predictions.pt',weights_only=True)
    for j in range(8):
        directory=demo/f'member_{j}';directory.mkdir();dirs.append(directory)
        torch.save(single,directory/'predictions.pt')
        write_json(directory/'frozen.json',dict(predictions_sha256=sha256(directory/'predictions.pt'),images_sha256=sha256(test/'images.json')))
    ensemble(dirs,demo/'ensemble')
    paths=extents(demo/'dino_single_pred',test,test/'images.json',demo/'extents')
    assert len(paths)==3
    for method in paths:
        summary=evaluate(demo/'extents'/method,demo/'prepared/evaluator',demo/'prepared/evaluator/hidden.json',demo/(method+'_eval'))
        assert summary['synthetic']['effective_k']==8
    masks=torch.ones(2,8,8,dtype=torch.bool)
    assert ged2(masks,masks)==0


def test_deterministic_cli_repeat_and_execution_record(tmp_path):
    from ssrb.cli import main
    demo=prepare_demo(tmp_path/'demo');root=demo/'prepared/trainer'
    config=demo/'comparison.json'
    write_json(config,dict(base=2,image_size=32,epochs=2,patience=1,batch_size=4,workers=0))
    before=(torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic,
            torch.backends.cudnn.benchmark)
    try:
        for name in ('first','second'):
            main(['train-comparison','--method','probunet','--root',str(root),
                  '--manifest',str(root/'visible.json'),'--output',str(demo/name),
                  '--config',str(config),'--deterministic'])
        a=torch.load(demo/'first/checkpoint.pt',weights_only=True)
        b=torch.load(demo/'second/checkpoint.pt',weights_only=True)
        assert all(torch.equal(a['state_dict'][k],b['state_dict'][k]) for k in a['state_dict'])
        assert (demo/'first/history.json').read_bytes()==(demo/'second/history.json').read_bytes()
        run=json.loads((demo/'first/run.json').read_text())
        assert run['execution']['deterministic_algorithms'] is True
        assert run['execution']['cudnn_deterministic'] is True
        assert run['execution']['cudnn_benchmark'] is False
        images=demo/'prepared/inference'
        main(['infer-comparison','--checkpoint',str(demo/'first/checkpoint.pt'),'--root',str(images),
              '--manifest',str(images/'images.json'),'--output',str(demo/'prediction'),'--deterministic'])
        frozen=json.loads((demo/'prediction/frozen.json').read_text())
        assert frozen['execution']['deterministic_algorithms'] is True
    finally:
        torch.use_deterministic_algorithms(before[0])
        torch.backends.cudnn.deterministic=before[1]
        torch.backends.cudnn.benchmark=before[2]


def test_original_raw_and_wide_training_order(tmp_path, monkeypatch):
    """Check the actual loader sequence against a trace from the original models."""
    from pathlib import Path
    import ssrb.comparisons as comparisons
    reference = json.loads((Path(__file__).parent/'fixtures/raw_shuffle_reference.json').read_text())
    demo = prepare_demo(tmp_path/'demo')
    root = demo/'prepared/trainer'
    config = demo/'order.json'
    write_json(config, dict(image_size=32, epochs=2, patience=8, batch_size=4, workers=0))
    for method in ('raw_capacity', 'wide_raw'):
        observed = []

        def trace(model, loader, device, name, c, epoch, optimizer=None, diffusion=None):
            order = [int(value.rsplit('_', 1)[1]) for batch in loader for value in batch['image_id']]
            if optimizer is not None:
                observed.append(order)
            return 1.0

        monkeypatch.setattr(comparisons, 'epoch_run', trace)
        comparisons.train(method, root, root/'visible.json', demo/method, config=config, seed=0)
        assert observed == reference['shuffle_order'][method]
