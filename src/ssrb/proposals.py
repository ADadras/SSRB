"""Image-only SAM2.1 automatic proposals through a separately installed upstream."""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
import torch
from PIL import Image
from .data import IMAGE_KEYS, read_rows, sha256, within, write_json, pack_binary_prediction
from .execution import record as execution_record

SAM2_COMMIT = '2b90b9f5ceec907a1c18123530e92e794ad901a4'


def rank_candidates(candidates):
    return sorted(candidates, key=lambda r: (-float(r['predicted_iou']), -float(r['stability_score']), int(r['original_index'])))


def native_metrics(predictions, references):
    from .metrics import image_metrics
    if predictions.shape[0] == 0:
        result = {f'{name}@{k}': 0. for name in ('MLQ', 'Worst') for k in (1, 2, 4, 8)}
        result.update({f'Gain@{k}': 0. for k in (2, 4, 8)})
        return result | dict(AUC=0., Containment=float('nan'), Unique95=0., active_head_count=0., effective_k=0.,
                             MLQ_native=0., Worst_native=0., Waste_native=1., PairDice_native=float('nan'))
    result = image_metrics(predictions, references)
    k = int(result['effective_k'])
    return result | {name+'_native': result[f'{name}@{k}'] for name in ('MLQ', 'Worst', 'Waste', 'PairDice')}


def generated_positive_point(probability, output_hw):
    from scipy import ndimage
    from .metrics import upsample_probability
    upsampled = upsample_probability(probability[None], output_hw)[0]
    binary = upsampled >= .5
    if binary.any():
        distance = ndimage.distance_transform_edt(binary.cpu().numpy())
        y_coord, x_coord = np.unravel_index(int(np.argmax(distance)), distance.shape)
        source = 'single_threshold_mask_maximum_interior_distance'
    else:
        y_coord, x_coord = np.unravel_index(int(torch.argmax(upsampled).item()), tuple(upsampled.shape))
        source = 'single_probability_argmax_empty_mask_fallback'
    return np.asarray([[float(x_coord), float(y_coord)]], dtype=np.float32), source


@torch.no_grad()
def infer(root, manifest, output, sam_repo, weights, *, device='cuda', config='configs/sam2.1/sam2.1_hiera_l.yaml',
          single_predictions=None, retain_full_pool=False):
    root, output, sam_repo = Path(root), Path(output), Path(sam_repo).resolve()
    if output.exists():
        raise FileExistsError(output)
    rows = sorted(read_rows(manifest), key=lambda r: r['image_id'])
    if any(set(r) != IMAGE_KEYS or r['split'] != 'test' for r in rows):
        raise ValueError('SAM inference requires image-only test rows')
    if single_predictions and retain_full_pool:
        raise ValueError('full AMG pools are available only in automatic mode')
    single, single_record = None, None
    if single_predictions:
        directory = Path(single_predictions)
        single_record = json.loads((directory/'frozen.json').read_text())
        if (sha256(directory/'predictions.pt') != single_record['predictions_sha256'] or
                single_record.get('images_sha256') != sha256(manifest) or
                single_record.get('method') != 'dino_single'):
            raise ValueError('requires matching frozen Single predictions')
        single = torch.load(directory/'predictions.pt', map_location='cpu', weights_only=True)
        if set(single) != {r['image_id'] for r in rows} or any(v['probability'].shape[0] != 1 for v in single.values()):
            raise ValueError('requires one Single probability map per test image')
    commit = subprocess.check_output(['git', '-C', str(sam_repo), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(sam_repo), 'status', '--porcelain'], text=True).strip()
    if commit != SAM2_COMMIT or dirty:
        raise ValueError('requires the clean, recorded SAM2 checkout')
    sys.path.insert(0, str(sam_repo))
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    model = build_sam2(config_file=config, ckpt_path=str(Path(weights).resolve()), device=device,
                       mode='eval', apply_postprocessing=True)
    if single is None:
        generator = SAM2AutomaticMaskGenerator(model)
    else:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        predictor = SAM2ImagePredictor(model)
    result, pools, prompt_sources = {}, {}, {}
    for position, r in enumerate(rows, start=1):
        with Image.open(within(root, r['image_path'])) as im:
            image = np.asarray(im.convert('RGB'))
        if single is None:
            proposals = generator.generate(image)
            for j, proposal in enumerate(proposals):
                proposal['original_index'] = j
            selected = rank_candidates(proposals)[:8]
            masks = torch.from_numpy(np.stack([p['segmentation'] for p in selected]).astype(np.bool_)) if selected else torch.empty(0, *image.shape[:2], dtype=torch.bool)
            scores = torch.tensor([float(p['predicted_iou']) for p in selected])
            if retain_full_pool:
                all_masks = torch.from_numpy(np.stack([p['segmentation'] for p in proposals]).astype(np.bool_)) if proposals else torch.empty(0, *image.shape[:2], dtype=torch.bool)
                pools[r['image_id']] = pack_binary_prediction(all_masks, torch.tensor([float(p['predicted_iou']) for p in proposals]))
        else:
            point, source = generated_positive_point(single[r['image_id']]['probability'][0], image.shape[:2])
            predictor.set_image(image)
            masks, scores, _ = predictor.predict(point_coords=point, point_labels=np.ones((1,), dtype=np.int32),
                                                 multimask_output=True, return_logits=False)
            order = np.argsort(-np.asarray(scores), kind='stable')
            masks = torch.from_numpy(np.asarray(masks, dtype=np.bool_)[order])
            scores = torch.as_tensor(np.asarray(scores)[order], dtype=torch.float32)
            prompt_sources[source] = prompt_sources.get(source, 0) + 1
        result[r['image_id']] = pack_binary_prediction(masks, scores)
        if position % 25 == 0 or position == len(rows):
            print(f'SAM2 images completed: {position}/{len(rows)}', flush=True)
    output.mkdir(parents=True)
    torch.save(result, output/'predictions.pt')
    record = dict(predictions_sha256=sha256(output/'predictions.pt'),
               images_sha256=sha256(manifest), weights_sha256=sha256(weights), upstream_commit=commit,
               binary_proposals=True, prediction_encoding='binary-rle-v1',
               method='sam2_amg' if single is None else 'sam2_generated_single_point_native_k3', device=device, config=config,
               execution=execution_record())
    if retain_full_pool:
        torch.save(pools, output/'full_pool.pt')
        record['full_pool_sha256'] = sha256(output/'full_pool.pt')
    if single_record:
        record.update(single_predictions_sha256=single_record['predictions_sha256'],
                      single_checkpoint_sha256=single_record['checkpoint_sha256'], prompt_source_counts=prompt_sources)
    write_json(output/'frozen.json', record)
    return result


def evaluate_pool(predictions, root, manifest, output):
    """Evaluate the oracle over every frozen AMG proposal, using hidden references."""
    from .data import prediction_probability
    from .metrics import dice_matrix, load_same_grid_masks, macro_average
    predictions, root, output = Path(predictions), Path(root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    frozen = json.loads((predictions/'frozen.json').read_text())
    if sha256(predictions/'full_pool.pt') != frozen['full_pool_sha256']:
        raise ValueError('the proposal pool changed after freezing')
    pool = torch.load(predictions/'full_pool.pt', map_location='cpu', weights_only=True)
    rows = read_rows(manifest)
    if set(pool) != {r['image_id'] for r in rows} or any(r['split'] != 'test' for r in rows):
        raise ValueError('requires matching test references')
    records=[]
    for row in rows:
        references=load_same_grid_masks([within(root,p) for p in row['heldout_mask_paths']])
        probability=prediction_probability(pool[row['image_id']])
        count=len(probability)
        if count:
            masks=torch.nn.functional.interpolate(probability[:,None].float(),size=references.shape[-2:],mode='nearest')[:,0]>=.5
            best=dice_matrix(masks,references).max(dim=0).values
            mlq,worst=float(best.mean()),float(best.min())
        else:
            mlq,worst=0.,0.
        records.append({k:row[k] for k in ('image_id','source_id','endpoint','visible_seed')} |
                       dict(pool_size=count,OracleMLQ_full_pool=mlq,OracleWorst_full_pool=worst))
        del probability
        if count:
            del masks, best
    metrics=('pool_size','OracleMLQ_full_pool','OracleWorst_full_pool')
    summary={endpoint:macro_average([r for r in records if r['endpoint']==endpoint],metrics)
             for endpoint in sorted({r['endpoint'] for r in records})}
    output.mkdir(parents=True)
    write_json(output/'per_image.json',records)
    write_json(output/'summary.json',summary)
    write_json(output/'evaluation.json',dict(full_pool_sha256=frozen['full_pool_sha256'],
        reference_manifest_sha256=sha256(manifest),scope='Full-pool oracle using hidden references; not deployable.'))
    return summary
