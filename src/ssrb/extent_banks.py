"""Prespecified extents and eight independently trained single-mask models."""
from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from .data import IMAGE_KEYS, read_rows, sha256, within, write_json
from .metrics import upsample_probability

THRESHOLDS = (.55,.45,.65,.35,.75,.25,.85,.15)
LOGIT_BIASES = (0.,.4,-.4,.8,-.8,1.2,-1.2,1.6)
MORPHOLOGY_GRID_UNITS = (0.,-1.,1.,-2.,2.,-4.,4.,-6.)


def fixed_banks(probability, output_hw):
    probability=upsample_probability(probability[None],output_hw)[0].clamp(1e-6,1.-1e-6)
    logit=torch.logit(probability)
    base=probability.detach().cpu().numpy()>=.5
    signed=ndimage.distance_transform_edt(base)-ndimage.distance_transform_edt(~base)
    scale=min(output_hw)/28.
    return dict(fixed_threshold_bank=torch.stack([(probability>=v).float() for v in THRESHOLDS]),
                fixed_logit_bias_bank=torch.stack([((logit+v).sigmoid()>=.5).float() for v in LOGIT_BIASES]),
                fixed_morphology_bank=torch.from_numpy(np.stack([(signed>=v*scale).astype(np.float32) for v in MORPHOLOGY_GRID_UNITS])))


def frozen_bank(path):
    path=Path(path);info=json.loads((path/'frozen.json').read_text())
    if sha256(path/'predictions.pt')!=info['predictions_sha256']:
        raise ValueError('predictions changed after freezing')
    return torch.load(path/'predictions.pt',map_location='cpu',weights_only=True),info


def extents(predictions, root, manifest, output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    single,info=frozen_bank(predictions)
    rows=read_rows(manifest)
    if any(set(r)!=IMAGE_KEYS or r['split']!='test' for r in rows) or {r['image_id'] for r in rows}!=set(single):
        raise ValueError('requires matching image-only test rows')
    result={name:{} for name in ('fixed_threshold_bank','fixed_logit_bias_bank','fixed_morphology_bank')}
    for row in rows:
        value=single[row['image_id']]['probability']
        if len(value)!=1:raise ValueError('requires one Single prediction')
        with Image.open(within(root,row['image_path'])) as image:hw=(image.height,image.width)
        for name,masks in fixed_banks(value[0],hw).items():
            result[name][row['image_id']]=dict(probability=masks,scores=torch.ones(8))
    for name,bank in result.items():
        directory=output/name;directory.mkdir(parents=True)
        torch.save(bank,directory/'predictions.pt')
        write_json(directory/'frozen.json',dict(predictions_sha256=sha256(directory/'predictions.pt'),
                   images_sha256=sha256(manifest),source_prediction_sha256=info['predictions_sha256'],
                   binary_proposals=True,method=name))
    return list(result)


def ensemble(predictions, output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    if len(predictions)!=8:raise ValueError('requires eight independent Single prediction directories')
    values=[frozen_bank(p) for p in predictions]
    if any(set(v[0])!=set(values[0][0]) or v[1]['images_sha256']!=values[0][1]['images_sha256'] for v in values):
        raise ValueError('ensemble image identities differ')
    result={}
    for key in values[0][0]:
        if any(v[0][key]['probability'].shape[0]!=1 for v in values):raise ValueError('each member must emit one mask')
        masks=torch.cat([v[0][key]['probability'] for v in values]);scores=torch.cat([v[0][key]['scores'] for v in values])
        order=torch.argsort(scores,descending=True,stable=True)
        result[key]=dict(probability=masks[order],scores=scores[order])
    output.mkdir(parents=True);torch.save(result,output/'predictions.pt')
    write_json(output/'frozen.json',dict(predictions_sha256=sha256(output/'predictions.pt'),
               images_sha256=values[0][1]['images_sha256'],members=[v[1]['predictions_sha256'] for v in values],method='ensemble8'))
    return dict(images=len(result))
