import json
from pathlib import Path
import numpy as np
import torch
from ssrb.diagnostics import head_metrics, symmetric_boundary_distance


def test_original_diagnostic_regression():
    torch.manual_seed(3964)
    anchor=torch.randn(1,8,16,16)
    residual=torch.randn_like(anchor)*.7
    expected=json.loads((Path(__file__).parent/'fixtures/diagnostic_reference.json').read_text())
    for head in range(8):
        actual=head_metrics(anchor[0,head].numpy(),residual[0,head].numpy(),
                            torch.linspace(-1,1,8)[head],torch.linspace(1,-1,8)[head])
        assert actual.keys()==expected[head].keys()
        for key,value in actual.items():
            if isinstance(value,str):
                assert value==expected[head][key]
            else:
                np.testing.assert_allclose(value,expected[head][key],atol=1e-7,rtol=1e-7)
    empty=np.zeros((16,16),dtype=bool)
    assert symmetric_boundary_distance(empty,empty)==0
    assert symmetric_boundary_distance(empty,~empty)==float(np.hypot(16,16))
