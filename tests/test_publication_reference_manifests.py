from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image
from ssrb.data import write_json,sha256
from ssrb.publication_plots import run


def test_qualitative_render_uses_separate_image_and_reference_manifests(tmp_path):
    cases=[]
    for endpoint in ('lidc','ima','riga_disc','bsds'):
        root=tmp_path/endpoint;image_root=root/'images';refs=root/'references';diag=root/'diagnostics';pred=root/'predictions'
        for p in (image_root,refs,diag,pred):
            p.mkdir(parents=True)
        identity=dict(image_id='synthetic_'+endpoint,source_id='synthetic_source',endpoint=endpoint,visible_seed=0,split='test')
        rgb=np.full((48,48,3),127,dtype=np.uint8)
        Image.fromarray(rgb).save(image_root/'image.png')
        mask=np.zeros((48,48),dtype=np.uint8);mask[12:36,14:34]=255
        Image.fromarray(mask).save(refs/'mask.png')
        write_json(image_root/'images.json',[identity|dict(image_path='image.png')])
        write_json(refs/'hidden.json',[identity|dict(visible_mask_path='mask.png',heldout_mask_paths=['mask.png'])])
        write_json(diag/'head_diagnostics.json',[dict(endpoint=endpoint,visible_seed=0,optimizer_seed=0,image_id=identity['image_id'],head_id=h,anchor_fused_dice=.9) for h in range(8)])
        write_json(diag/'provenance.json',dict(reference_masks_read=False,checkpoint_sha256='synthetic-checkpoint'))
        probability=torch.from_numpy(mask>0).float().repeat(8,1,1)
        torch.save({identity['image_id']:dict(probability=probability,scores=torch.arange(8,0,-1).float())},pred/'predictions.pt')
        write_json(pred/'frozen.json',dict(predictions_sha256=sha256(pred/'predictions.pt'),checkpoint_sha256='synthetic-checkpoint'))
        cases.append(dict(endpoint=endpoint,root=str(image_root),reference_root=str(refs),manifest=str(refs/'hidden.json'),diagnostics=str(diag),predictions=str(pred)))
    source=tmp_path/'cases.json';write_json(source,cases)
    output=tmp_path/'figure.pdf';run('qualitative',source,output)
    record=json.loads(output.with_suffix('.record.json').read_text())
    assert output.read_bytes().startswith(b'%PDF')
    assert record['reference_masks_used_for_selection'] is False
    assert len(record['cases'])==4
    assert all(c['selection_fixed_before_reference_manifest_read'] for c in record['cases'])
