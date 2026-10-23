"""Visible-validation head matching and the uncalibrated four-plus-four control."""
from __future__ import annotations
import itertools
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .comparisons import ComparisonDataset, configuration, input_snapshot, load as load_comparison
from .data import IMAGE_KEYS, load_image, read_rows, sha256, within, write_json
from .training import load_model


def lexicographic_assignment(cost):
    if tuple(cost.shape) != (8, 8):
        raise ValueError('alignment requires an 8x8 cost matrix')
    array = cost.detach().cpu().double().numpy()
    best, best_value = None, float('inf')
    for permutation in itertools.permutations(range(8)):
        value = float(sum(array[k, permutation[k]] for k in range(8)))
        if value < best_value - 1e-12:
            best_value, best = value, permutation
    return list(best), best_value


def pair_alignment_cost(semantic, raw, target):
    semantic = F.interpolate(semantic, size=raw.shape[-2:], mode='bilinear', align_corners=False)
    cost = torch.zeros(8, 8, dtype=torch.float64)
    for k in range(8):
        for j in range(8):
            fused = semantic[:, k] + raw[:, j]
            bce = F.binary_cross_entropy_with_logits(fused, target, reduction='none').mean((1, 2))
            probability = fused.sigmoid()
            inter = (probability * target).sum((1, 2))
            denominator = probability.sum((1, 2)) + target.sum((1, 2))
            dice_loss = 1. - (2. * inter + 1.) / (denominator + 1.)
            cost[k, j] = float((bce + dice_loss).sum().cpu())
    return cost


def models(semantic_checkpoint, raw_checkpoint, device):
    semantic, sc = load_model(semantic_checkpoint, device)
    raw, rc = load_comparison(raw_checkpoint, device)
    if sc['stage'] != 'semantic' or rc['method'] != 'raw_capacity':
        raise ValueError('requires semantic Capacity-WTA and raw Capacity-WTA')
    for key in ('endpoint', 'visible_seed', 'optimizer_seed'):
        if sc[key] != rc[key]:
            raise ValueError('checkpoint identity differs')
    return semantic, raw, sc, rc


@torch.no_grad()
def align(semantic_checkpoint, raw_checkpoint, root, manifest, output, *, device='cpu', batch_size=12):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    semantic, raw, sc, rc = models(semantic_checkpoint, raw_checkpoint, device)
    rows, seal = input_snapshot(root, manifest, features=True)
    if rc['input_seal']['manifest_sha256'] != seal['manifest_sha256'] or any(seal['files'].get(k) != v for k, v in rc['input_seal']['files'].items()):
        raise ValueError('raw checkpoint inputs differ')
    core_seal = dict(seal['files'], manifest_sha256=seal['manifest_sha256'])
    if sc['input_seal'] != core_seal:
        raise ValueError('semantic checkpoint inputs differ')
    c = configuration('conventional_residual')
    c['image_size'] = rc['config']['image_size']
    data = ComparisonDataset(root, rows, 'val', c, 'conventional_residual')
    cost = torch.zeros(8, 8, dtype=torch.float64)
    for batch in DataLoader(data, batch_size=batch_size, shuffle=False):
        a, _ = semantic(batch['features'].to(device))
        b = raw(batch['image'].to(device))['mask_logits']
        cost += pair_alignment_cost(a, b, batch['target'].to(device))
    permutation, value = lexicographic_assignment(cost / len(data))
    if input_snapshot(root, manifest, features=True)[1] != seal:
        raise ValueError('alignment inputs changed')
    result = dict(schema='ssrb-alignment-v1', permutation=permutation, cost_matrix=(cost / len(data)).tolist(),
                  objective=value, semantic_sha256=sha256(semantic_checkpoint), raw_sha256=sha256(raw_checkpoint),
                  input_seal=seal, examples=len(data))
    write_json(output, result)
    return result


def combine(semantic_masks, semantic_ranks, raw_masks, raw_ranks, *, permutation=None):
    if semantic_masks.shape[1] != 8 or raw_masks.shape[1] != 8:
        raise ValueError('requires two eight-head banks')
    semantic_masks = F.interpolate(semantic_masks, size=raw_masks.shape[-2:], mode='bilinear', align_corners=False)
    if permutation is not None:
        if sorted(permutation) != list(range(8)):
            raise ValueError('not a head permutation')
        masks, ranks = semantic_masks + raw_masks[:, permutation], semantic_ranks + raw_ranks[:, permutation]
        order = torch.argsort(ranks.sigmoid(), dim=1, descending=True, stable=True)
        return masks.sigmoid().gather(1, order[:, :, None, None].expand_as(masks)), ranks.sigmoid().gather(1, order)
    # Preserve the accepted raw-first concatenation; scores do not reorder the banks.
    a = torch.argsort(raw_ranks, dim=1, descending=True, stable=True)[:, :4]
    b = torch.argsort(semantic_ranks, dim=1, descending=True, stable=True)[:, :4]
    probability = torch.cat([raw_masks.sigmoid().gather(1, a[:, :, None, None].expand(-1, -1, *raw_masks.shape[-2:])),
                             semantic_masks.sigmoid().gather(1, b[:, :, None, None].expand(-1, -1, *raw_masks.shape[-2:]))], 1)
    scores = torch.cat([raw_ranks.sigmoid().gather(1, a), semantic_ranks.sigmoid().gather(1, b)], 1)
    return probability, scores


@torch.no_grad()
def infer(semantic_checkpoint, raw_checkpoint, root, manifest, feature_file, output, *, alignment=None,
          device='cpu', batch_size=12):
    import json
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if batch_size < 1:
        raise ValueError('batch size must be positive')
    semantic, raw, sc, rc = models(semantic_checkpoint, raw_checkpoint, device)
    rows = sorted(read_rows(manifest), key=lambda r: r['image_id'])
    if any(set(r) != IMAGE_KEYS or r['split'] != 'test' or r['endpoint'] != sc['endpoint'] or r['visible_seed'] != sc['visible_seed'] for r in rows):
        raise ValueError('requires matching image-only test rows')
    payload = torch.load(feature_file, map_location='cpu', weights_only=True)
    if set(payload) != {'features', 'image_ids', 'grid_hw'} or set(payload['image_ids']) != {r['image_id'] for r in rows}:
        raise ValueError('test feature identities differ')
    index = {name: i for i, name in enumerate(payload['image_ids'])}
    permutation = None
    if alignment:
        info = json.loads(Path(alignment).read_text())
        if info['semantic_sha256'] != sha256(semantic_checkpoint) or info['raw_sha256'] != sha256(raw_checkpoint):
            raise ValueError('alignment belongs to different checkpoints')
        permutation = info['permutation']
    predictions = {}
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        features = torch.stack([payload['features'][index[r['image_id']]].float() for r in batch]).to(device)
        images = torch.stack([load_image(within(root, r['image_path']), rc['config']['image_size']) for r in batch]).to(device)
        a, ar = semantic(features)
        result = raw(images)
        masks, scores = combine(a, ar, result['mask_logits'], result['validity_logits'], permutation=permutation)
        for j, row in enumerate(batch):
            predictions[row['image_id']] = dict(probability=masks[j].cpu(), scores=scores[j].cpu())
    output.mkdir(parents=True)
    torch.save(predictions, output / 'predictions.pt')
    write_json(output / 'frozen.json', dict(predictions_sha256=sha256(output / 'predictions.pt'),
               semantic_sha256=sha256(semantic_checkpoint), raw_sha256=sha256(raw_checkpoint),
               images_sha256=sha256(manifest), alignment_sha256=sha256(alignment) if alignment else None,
               method='posthoc_additive' if alignment else 'native_union_4_4', device=device, batch_size=batch_size))
    return predictions
