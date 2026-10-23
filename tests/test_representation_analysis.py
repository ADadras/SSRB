import json
from pathlib import Path
import numpy as np
import pandas as pd
from ssrb import legacy_statistics as stats


def test_original_representation_stage_intervals_and_decisions():
    reference=json.loads((Path(__file__).parent/'fixtures/representation_reference.json').read_text())
    units=pd.DataFrame(reference['units'])
    for stage,regimes in [('stage_a',('BSDS','LIDC','LIDC-hard')),('stage_b',('IMA++','RIGA-disc','RIGA-cup'))]:
        boot,_=stats.bootstrap_means(units[units.regime.isin(regimes)],replicates=reference['replicates'],seed=20260814)
        effects,gates,decision,critical=stats.analyze_stage(boot,stage)
        expected=reference['stages'][stage]
        columns=['effect','ci95_low','ci95_high','simultaneous_ci95_low','simultaneous_ci95_high']
        np.testing.assert_allclose(effects[columns],pd.DataFrame(expected['effects'])[columns].astype(float),atol=1e-12,rtol=0,equal_nan=True)
        assert gates.passed.tolist()==[r['passed'] for r in expected['gates']]
        assert decision==expected['decision']
        assert critical==expected['critical']
