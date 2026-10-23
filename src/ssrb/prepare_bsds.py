from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
from scipy.io import loadmat

def mat_structs(path: Path) -> list[Any]:
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    gt = data["groundTruth"]
    if getattr(gt, "shape", None) == ():
        return [gt.item()]
    return list(np.asarray(gt).ravel())

def mask_dice(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(bool)
    bb = b.astype(bool)
    inter = float(np.logical_and(aa, bb).sum())
    den = float(aa.sum() + bb.sum())
    return (2.0 * inter + 1.0) / (den + 1.0)

def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)

def convert_split(
    *,
    bsds_data: Path,
    out_root: Path,
    split: str,
    min_area_frac: float,
    max_area_frac: float,
    dedup_dice: float,
    max_masks_per_image: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    image_dir = bsds_data / "images" / split
    gt_dir = bsds_data / "groundTruth" / split
    mask_root = out_root / "masks" / split
    for mat_path in sorted(gt_dir.glob("*.mat")):
        image_id = mat_path.stem
        image_path = image_dir / f"{image_id}.jpg"
        if not image_path.exists():
            # BSDS has a few images under jpg names that still match mat stems;
            # if one is missing, skip rather than making a broken manifest row.
            continue
        candidates: list[tuple[np.ndarray, str, float]] = []
        for ann_idx, obj in enumerate(mat_structs(mat_path)):
            seg = np.asarray(obj.Segmentation)
            total = float(seg.size)
            for label in np.unique(seg):
                region = seg == label
                area_frac = float(region.sum()) / max(1.0, total)
                if area_frac < float(min_area_frac) or area_frac > float(max_area_frac):
                    continue
                candidates.append((region, f"A{ann_idx:02d}_R{int(label):03d}", area_frac))
        candidates.sort(key=lambda item: item[2], reverse=True)
        kept: list[tuple[np.ndarray, str, float]] = []
        for region, ann_id, area_frac in candidates:
            if any(mask_dice(region, old_region) >= float(dedup_dice) for old_region, _, _ in kept):
                continue
            kept.append((region, ann_id, area_frac))
            if int(max_masks_per_image) > 0 and len(kept) >= int(max_masks_per_image):
                break
        if len(kept) < 2:
            continue
        mask_paths = []
        annotator_ids = []
        areas = []
        for idx, (region, ann_id, area_frac) in enumerate(kept):
            mask_path = mask_root / image_id / f"{image_id}_{ann_id}_{idx:03d}.png"
            save_mask(region, mask_path)
            mask_paths.append(str(mask_path))
            annotator_ids.append(ann_id)
            areas.append(area_frac)
        rows.append(
            {
                "image_id": f"BSDS_{image_id}",
                "image_path": str(image_path),
                "split": split,
                "task_id": f"BSDS_{image_id}",
                "class_or_object_id": "partition_region",
                "annotator_ids": json.dumps(annotator_ids),
                "mask_paths": json.dumps(mask_paths),
                "num_masks": len(mask_paths),
                "metadata": json.dumps(
                    {
                        "dataset": "BSDS500",
                        "conversion": "partition_regions",
                        "min_area_frac": float(min_area_frac),
                        "max_area_frac": float(max_area_frac),
                        "dedup_dice": float(dedup_dice),
                        "area_frac_mean": float(np.mean(areas)),
                    }
                ),
            }
        )
    return rows

def prepare_bsds(data_root, output):
    from .data import write_json
    data_root, output = Path(data_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    # Write images and derived masks to the requested output directory.
    import shutil
    rows = []
    for split in ("train", "val", "test"):
        source = convert_split(bsds_data=data_root, out_root=output, split=split,
                               min_area_frac=.015, max_area_frac=.85, dedup_dice=.95, max_masks_per_image=16)
        for r in source:
            name = "images/" + split + "/" + Path(r["image_path"]).name
            (output / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(r["image_path"], output / name)
            rows.append(dict(image_id=r["image_id"], source_id=r["image_id"], split=split, endpoint="bsds",
                             image_path=name, mask_paths=[str(Path(p).relative_to(output)) for p in json.loads(r["mask_paths"])]))
    write_json(output / "all_annotations.json", rows)
    return len(rows)
