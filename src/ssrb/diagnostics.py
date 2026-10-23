"""Measure changes to the anchor without opening reference masks."""
from pathlib import Path
import json
import numpy as np
from scipy import ndimage
import torch
from torch.nn import functional as F
from .data import IMAGE_KEYS, load_image, read_rows, sha256, within, write_json
from .metrics import encode_binary_rle
from .training import load_model


def boundary(mask):
    mask = mask.astype(bool, copy=False)
    return np.logical_xor(mask, ndimage.binary_erosion(mask, border_value=0)) if mask.any() else np.zeros_like(mask)


def symmetric_boundary_distance(left, right):
    a, b = boundary(left), boundary(right)
    if not a.any() and not b.any():
        return 0.
    if not a.any() or not b.any():
        return float(np.hypot(*left.shape))
    return .5 * (float(ndimage.distance_transform_edt(~b)[a].mean()) +
                 float(ndimage.distance_transform_edt(~a)[b].mean()))


def head_metrics(anchor, residual, anchor_rank=0., residual_rank=0.):
    anchor, residual = np.asarray(anchor, dtype=np.float32), np.asarray(residual, dtype=np.float32)
    a, b = anchor >= 0., (anchor + residual) >= 0.
    band = ndimage.binary_dilation(boundary(a), iterations=3)
    magnitude = np.abs(residual)
    band_mean = float(magnitude[band].mean()) if band.any() else 0.
    outside_mean = float(magnitude[~band].mean()) if (~band).any() else 0.
    rms = np.sqrt(np.mean(residual ** 2))
    return dict(residual_mean_abs=float(magnitude.mean()), residual_rms=float(rms),
                anchor_rms=float(np.sqrt(np.mean(anchor ** 2))),
                relative_residual_rms=float(rms / max(np.sqrt(np.mean(anchor ** 2)), 1e-12)),
                anchor_fused_dice=float((2.*np.logical_and(a,b).sum()+1.)/(a.sum()+b.sum()+1.)),
                anchor_area_fraction=float(a.mean()), fused_area_fraction=float(b.mean()),
                signed_area_change=float(b.mean()-a.mean()), absolute_area_change=float(abs(b.mean()-a.mean())),
                absolute_component_change=float(abs(int(ndimage.label(b)[1])-int(ndimage.label(a)[1]))),
                symmetric_boundary_displacement_px=symmetric_boundary_distance(a,b),
                boundary_band_pixel_fraction=float(band.mean()),
                residual_magnitude_in_boundary_band=float(magnitude[band].sum()/max(float(magnitude.sum()),1e-12)),
                boundary_to_nonboundary_magnitude_ratio=float(band_mean/max(outside_mean,1e-12)),
                anchor_rank_logit=float(anchor_rank), residual_rank_logit=float(residual_rank),
                anchor_rle_128=json.dumps(encode_binary_rle(torch.from_numpy(a))),
                fused_rle_128=json.dumps(encode_binary_rle(torch.from_numpy(b))))


@torch.no_grad()
def run(checkpoint, root, manifest, feature_file, output, *, device='cpu', batch_size=12):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows = sorted(read_rows(manifest), key=lambda r:r['image_id'])
    if any(set(r)!=IMAGE_KEYS or r['split']!='test' for r in rows):
        raise ValueError('diagnostics require image-only test rows')
    payload = torch.load(feature_file, map_location='cpu', weights_only=True)
    if set(payload)!={'features','image_ids','grid_hw'}:
        raise ValueError('unexpected feature metadata')
    index = {str(k):j for j,k in enumerate(payload['image_ids'])}
    if len(index)!=len(payload['image_ids']) or set(index)!={r['image_id'] for r in rows}:
        raise ValueError('feature image identities differ')
    model, ck = load_model(checkpoint, device)
    if ck['stage']!='residual' or any(r['endpoint']!=ck['endpoint'] or r['visible_seed']!=ck['visible_seed'] for r in rows):
        raise ValueError('requires a matching SSRB checkpoint')
    if batch_size<1:
        raise ValueError('batch size must be positive')
    records=[]
    for start in range(0,len(rows),batch_size):
        batch=rows[start:start+batch_size]
        features=torch.stack([payload['features'][index[r['image_id']]].float() for r in batch]).to(device)
        images=torch.stack([load_image(within(root,r['image_path']),ck['config']['image_size']) for r in batch]).to(device)
        anchor,ranks=model.anchor(features)
        anchor=F.interpolate(anchor,size=images.shape[-2:],mode='bilinear',align_corners=False)
        residual=model.raw(images)
        for j,r in enumerate(batch):
            for head in range(ck['config']['heads']):
                values=head_metrics(anchor[j,head].cpu().numpy(),residual['mask_logits'][j,head].cpu().numpy(),
                                    ranks[j,head].cpu(),residual['validity_logits'][j,head].cpu())
                records.append({k:r[k] for k in ('image_id','source_id','endpoint','visible_seed')} |
                               dict(optimizer_seed=ck['optimizer_seed'],head_id=head,**values))
    output.mkdir(parents=True)
    write_json(output/'head_diagnostics.json',records)
    write_json(output/'provenance.json',dict(checkpoint_sha256=sha256(checkpoint),images_sha256=sha256(manifest),
               feature_sha256=sha256(feature_file),rows=len(records),reference_masks_read=False,
               boundary_band_pixels=3,device=device,batch_size=batch_size))
    return dict(images=len(rows),heads=ck['config']['heads'],rows=len(records))
