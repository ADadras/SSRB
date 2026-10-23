import json
import pytest
from ssrb.tables import aggregate


def test_visible_choices_are_averaged_before_images(tmp_path):
    entries=[]
    for optimizer in range(3):
        for visible in range(3):
            path=tmp_path/f'o{optimizer}v{visible}';path.mkdir()
            rows=[dict(endpoint='lidc_hard',optimizer_seed=optimizer,visible_seed=visible,image_id='a',source_id='p1',**{'MLQ@8':.8})]
            if visible==0:
                rows.append(dict(endpoint='lidc_hard',optimizer_seed=optimizer,visible_seed=visible,image_id='b',source_id='p2',**{'MLQ@8':.2}))
            # optimizer_seed belongs to the directory metadata in real files.
            for row in rows:row.pop('optimizer_seed')
            (path/'per_image.json').write_text(json.dumps(rows))
            entries.append(dict(evaluation=str(path),optimizer_seed=optimizer))
    (tmp_path/'inputs.json').write_text(json.dumps(entries))
    rows=aggregate(tmp_path/'inputs.json',tmp_path/'table.csv','partial')
    assert rows[0]['analysis_units']==6
    assert abs(rows[0]['MLQ@8']-.5)<1e-15
    descriptive=aggregate(tmp_path/'inputs.json',tmp_path/'descriptive.csv','partial',averaging='run-means')
    assert descriptive[0]['averaging']=='run-means'
    assert descriptive[0]['analysis_units']==9
    assert descriptive[0]['MLQ@8']==pytest.approx(.7)


def _single_seed_evaluations(tmp_path, visible_choices):
    entries=[]
    for visible in visible_choices:
        path=tmp_path/f'v{visible}';path.mkdir()
        rows=[dict(endpoint='bsds',visible_seed=visible,image_id='a',source_id='s1',
                   **{'MLQ@8':.2+.1*visible})]
        (path/'per_image.json').write_text(json.dumps(rows))
        entries.append(dict(evaluation=str(path),optimizer_seed=0))
    manifest=tmp_path/'inputs.json';manifest.write_text(json.dumps(entries))
    return manifest


def test_single_optimizer_seed_requires_an_explicit_declaration(tmp_path):
    manifest=_single_seed_evaluations(tmp_path,range(3))
    with pytest.raises(ValueError,match='every declared optimizer seed'):
        aggregate(manifest,tmp_path/'default.csv','partial')
    rows=aggregate(manifest,tmp_path/'baseline.csv','new-training-reproduction',optimizer_seeds=[0])
    assert rows[0]['analysis_units']==1
    assert rows[0]['MLQ@8']==pytest.approx(.3)


def test_aggregate_rejects_a_missing_visible_choice(tmp_path):
    manifest=_single_seed_evaluations(tmp_path,[0,2])
    with pytest.raises(ValueError,match='all three visible choices'):
        aggregate(manifest,tmp_path/'incomplete.csv','partial',optimizer_seeds=[0])
