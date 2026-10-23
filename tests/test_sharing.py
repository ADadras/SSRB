import json
from pathlib import Path

import torch

from ssrb.sharing import factorial_objective,factorial_assignment_weights,deterministic_deranged_rows
from ssrb.losses import pair_cost


def test_sharing_controls_match_original_loss_gradients_and_derangement():
    ref=json.loads((Path(__file__).parent/'fixtures/sharing_reference.json').read_text())
    target=torch.tensor(ref['target']);features=torch.tensor(ref['features'])
    for case in ref['cases']:
        masks=torch.tensor(ref['masks'],requires_grad=True);ranks=torch.tensor(ref['ranks'],requires_grad=True)
        loss,_=factorial_objective(masks,ranks,target,features,ref['image_ids'],method=case['method'],**ref['arguments'])
        loss.backward()
        # Float32 scalar reductions can differ by one ULP across execution states.
        # The separate full-fit gates require exact original/release states.
        torch.testing.assert_close(loss,torch.tensor(case['loss']),atol=1e-6,rtol=0)
        torch.testing.assert_close(masks.grad,torch.tensor(case['mask_gradient']),atol=1e-7,rtol=0)
        torch.testing.assert_close(ranks.grad,torch.tensor(case['rank_gradient']),atol=1e-7,rtol=0)
        kwargs={k:v for k,v in ref['arguments'].items() if k not in ('q_loss_weight','pair_diversity_weight')}
        weights,anchors,_=factorial_assignment_weights(pair_cost(masks,target),masks,features,ref['image_ids'],method=case['method'],**kwargs)
        torch.testing.assert_close(weights,torch.tensor(case['weights']),atol=2e-7,rtol=0)
        assert torch.equal(anchors,torch.tensor(case['anchors']))
    affinity=torch.arange(24,dtype=torch.float32).reshape(3,8);anchors=torch.tensor([0,3,7])
    rng=torch.get_rng_state().clone()
    shuffled,mappings=deterministic_deranged_rows(affinity,anchors,ref['image_ids'],endpoint='bsds',visible_seed=1,salt='test')
    assert torch.equal(torch.get_rng_state(),rng)
    assert torch.equal(shuffled.sort(dim=1).values,affinity.sort(dim=1).values)
    for row,anchor in enumerate(anchors):
        assert mappings[row][anchor]==anchor
        assert all(donor!=receiver for receiver,donor in enumerate(mappings[row]) if receiver!=anchor)
