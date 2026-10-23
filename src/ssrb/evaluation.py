"""Evaluation is a separate process that reads saved predictions and references."""
import json
import math
from pathlib import Path
import torch
from . import metrics
from .data import read_rows, sha256, within, write_json, prediction_probability


def evaluate(predictions, root, manifest, output, ged=False):
    predictions, root, output = Path(predictions), Path(root), Path(output)
    frozen = json.loads((predictions/"frozen.json").read_text())
    if sha256(predictions/"predictions.pt") != frozen["predictions_sha256"]:
        raise ValueError("predictions changed after freezing")
    bank = torch.load(predictions/"predictions.pt", map_location="cpu", weights_only=True)
    rows = read_rows(manifest)
    if set(bank) != {r["image_id"] for r in rows}:
        raise ValueError("prediction/reference image sets differ")
    binary_proposals = frozen.get("binary_proposals", False)
    def scores(probability, references):
        if binary_proposals:
            from .proposals import native_metrics
            if probability.shape[0] == 0:
                prediction = torch.empty(0, *references.shape[-2:], dtype=torch.bool)
            else:
                prediction = torch.nn.functional.interpolate(probability[:, None].float(), size=references.shape[-2:], mode="nearest")[:, 0] >= .5
            return native_metrics(prediction, references)
        prediction = metrics.threshold_probability(metrics.upsample_probability(probability, references.shape[-2:]))
        result = metrics.image_metrics(prediction, references)
        if ged:
            result["GED2"] = metrics.ged2(prediction, references)
        return result
    records, reads = [], []
    for r in rows:
        if r["split"] != "test":
            raise ValueError("evaluator accepts test rows only")
        paths = [within(root, p) for p in r["heldout_mask_paths"]]
        reads.extend(r["heldout_mask_paths"])
        references = metrics.load_same_grid_masks(paths)
        probability = prediction_probability(bank[r["image_id"]])
        base = dict(image_id=r["image_id"], source_id=r["source_id"], endpoint=r["endpoint"], visible_seed=r["visible_seed"])
        records.append(base | scores(probability, references))
        if r["endpoint"] == "lidc":
            visible = metrics.load_binary_mask(within(root, r["visible_mask_path"]))[None]
            reads.append(r["visible_mask_path"])
            if visible.shape[-2:] != references.shape[-2:]:
                raise ValueError("LIDC masks must share their native grid")
            keep = metrics.dice_matrix(visible, references)[0] < .80
            if keep.any():
                records.append(base | {"endpoint": "lidc_hard"} | scores(probability, references[keep]))
    keys = sorted(set().union(*(set(r) for r in records)) - {"image_id", "source_id", "endpoint", "visible_seed"})
    summaries = {e: metrics.macro_average([r for r in records if r["endpoint"] == e], keys)
                 for e in sorted({r["endpoint"] for r in records})}
    def finite(obj):
        if isinstance(obj, dict):
            return {k: finite(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [finite(v) for v in obj]
        return None if isinstance(obj, float) and not math.isfinite(obj) else obj
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    write_json(output/"per_image.json", finite(records))
    write_json(output/"summary.json", finite(summaries))
    write_json(output/"evaluation.json", dict(predictions_sha256=frozen["predictions_sha256"],
               reference_manifest_sha256=sha256(manifest), reference_reads=sorted(set(reads))))
    return summaries
