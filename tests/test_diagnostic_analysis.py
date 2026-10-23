import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ssrb.diagnostic_statistics import summarize


def test_diagnostic_intervals_match_original_crossed_bootstrap():
    fixture=json.loads((Path(__file__).parent/'fixtures/diagnostic_summary_reference.json').read_text())
    full=pd.DataFrame(fixture['full'])
    actual=summarize(full,replicates=fixture['replicates'],seed=fixture['seed'])
    expected=pd.DataFrame(fixture['summary'])
    joined=actual.merge(expected,on=['endpoint','metric'],suffixes=('_actual','_original'),validate='one_to_one')
    assert len(joined)==50
    for field in ('mean','ci95_low','ci95_high'):
        np.testing.assert_allclose(joined[field+'_actual'],joined[field+'_original'],atol=1e-12,rtol=0)
    with pytest.raises(ValueError,match='visible and optimizer'):
        summarize(full[~((full.endpoint=='lidc') & (full.visible_seed==0) & (full.optimizer_seed==0))])
    with pytest.raises(ValueError,match='eight distinct heads'):
        summarize(full.iloc[1:])
