import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ssrb.sharing_statistics import summarize


def test_support_specificity_reproduces_original_full_factorial_intervals():
    ref=json.loads((Path(__file__).parent/'fixtures/sharing_statistics_reference.json').read_text())
    full=pd.DataFrame(ref['full'])
    actual=summarize(full,replicates=ref['replicates'],seed=ref['seed'])
    expected=pd.DataFrame(ref['effects'])
    paired=actual.merge(expected,on=['candidate','component'],suffixes=('_release','_original'),validate='one_to_one')
    assert len(paired)==6
    for column in ('effect','ci95_low','ci95_high','threshold'):
        np.testing.assert_allclose(paired[column+'_release'],paired[column+'_original'],atol=1e-12,rtol=0)
    assert (paired.predicate_pass_release==paired.predicate_pass_original).all()
    with pytest.raises(ValueError,match='incomplete visible/optimizer'):
        summarize(full[~((full.visible_seed==2)&(full.optimizer_seed==2)&(full.endpoint=='bsds'))])
