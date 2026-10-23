from __future__ import annotations
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
NODULE_RE = re.compile(r"Segmentation of Nodule\s+(\d+)\s+-\s+Annotation\s+(.+)")

def log(event: str, **kwargs: Any) -> None:
    print(json.dumps({"event": event, **kwargs}, sort_keys=True), flush=True)

def stable_split(patient_id: str, train_frac: float, val_frac: float) -> str:
    value = int(hashlib.sha1(str(patient_id).encode("utf-8")).hexdigest()[:8], 16) / float(16**8)
    if value < float(train_frac):
        return "train"
    if value < float(train_frac) + float(val_frac):
        return "val"
    return "test"

def parse_nodule(description: str) -> tuple[str, str] | None:
    match = NODULE_RE.match(str(description))
    if not match:
        return None
    return match.group(1), match.group(2).strip().replace(" ", "_")

def source_sop_for_frame(fg: Any) -> str | None:
    for item in getattr(fg, "DerivationImageSequence", []):
        for src in getattr(item, "SourceImageSequence", []):
            return str(src.ReferencedSOPInstanceUID)
    return None

def ct_series_uid(ds: Any) -> str | None:
    for item in getattr(ds, "ReferencedSeriesSequence", []):
        uid = getattr(item, "SeriesInstanceUID", None)
        if uid:
            return str(uid)
    return None

@dataclass
class SegVolume:
    path: Path
    patient_id: str
    study_uid: str
    series_uid: str
    nodule_id: str
    annotator_id: str
    ct_series_uid: str
    frames: dict[str, np.ndarray]
    z_by_sop: dict[str, float]

def load_seg(path: Path) -> SegVolume | None:
    ds = pydicom.dcmread(str(path))
    parsed = parse_nodule(getattr(ds, "SeriesDescription", ""))
    if parsed is None:
        return None
    nodule_id, annotator_id = parsed
    arr = ds.pixel_array.astype(bool)
    if arr.ndim == 2:
        arr = arr[None]
    frames: dict[str, np.ndarray] = {}
    z_by_sop: dict[str, float] = {}
    for idx, fg in enumerate(getattr(ds, "PerFrameFunctionalGroupsSequence", [])):
        sop = source_sop_for_frame(fg)
        if sop is None:
            continue
        mask = arr[idx].astype(bool)
        if not mask.any():
            continue
        frames[sop] = mask
        if hasattr(fg, "PlanePositionSequence"):
            z_by_sop[sop] = float(fg.PlanePositionSequence[0].ImagePositionPatient[2])
    if not frames:
        return None
    series = ct_series_uid(ds)
    if series is None:
        return None
    return SegVolume(
        path=path,
        patient_id=str(ds.PatientID),
        study_uid=str(ds.StudyInstanceUID),
        series_uid=str(ds.SeriesInstanceUID),
        nodule_id=str(nodule_id),
        annotator_id=str(annotator_id),
        ct_series_uid=series,
        frames=frames,
        z_by_sop=z_by_sop,
    )

def ct_to_uint8(path: Path) -> np.ndarray:
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = arr * slope + intercept
    lo, hi = -1000.0, 500.0
    return ((np.clip(hu, lo, hi) - lo) / (hi - lo) * 255.0).astype(np.uint8)

def bbox_from_masks(masks: list[np.ndarray], pad: int, min_size: int) -> tuple[int, int, int, int]:
    union = np.logical_or.reduce(masks)
    ys, xs = np.where(union)
    h, w = union.shape
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0 = max(0, y0 - int(pad))
    x0 = max(0, x0 - int(pad))
    y1 = min(h, y1 + int(pad))
    x1 = min(w, x1 + int(pad))
    cy = (y0 + y1) // 2
    cx = (x0 + x1) // 2
    size = max(int(min_size), y1 - y0, x1 - x0)
    half = (size + 1) // 2
    y0 = max(0, min(h - size, cy - half))
    x0 = max(0, min(w - size, cx - half))
    y1 = min(h, y0 + size)
    x1 = min(w, x0 + size)
    return y0, y1, x0, x1

def save_png(arr: np.ndarray, path: Path, *, mode: str = "L") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode=mode).save(path)

def build_rows(
    root: Path,
    *,
    min_annotations: int,
    pad: int,
    min_crop: int,
    max_tasks: int,
    max_z_mm: float,
    output_root: Path,
) -> pd.DataFrame:
    seg_paths = sorted((root / "raw" / "seg_qiicr").glob("*/*.dcm"))
    groups: dict[tuple[str, str, str], list[SegVolume]] = {}
    skipped = 0
    for idx, path in enumerate(seg_paths, 1):
        try:
            seg = load_seg(path)
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        if seg is None:
            skipped += 1
            continue
        key = (seg.patient_id, seg.study_uid, seg.nodule_id)
        groups.setdefault(key, []).append(seg)
        if idx % 500 == 0:
            log("e2201_decode_progress", done=idx, total=len(seg_paths), groups=len(groups), skipped=skipped)
    rows: list[dict[str, Any]] = []
    out_root = output_root
    exact_slice_masks = 0
    nearest_slice_masks = 0
    discarded_annotations = 0
    usable_groups = 0
    for key, segs in sorted(groups.items()):
        if len(segs) < int(min_annotations):
            continue
        sop_scores: dict[str, int] = {}
        z_lookup: dict[str, float] = {}
        for seg in segs:
            z_lookup.update(seg.z_by_sop)
            for sop, mask in seg.frames.items():
                sop_scores[sop] = sop_scores.get(sop, 0) + int(mask.sum())
        if not sop_scores:
            continue
        center_sop = max(sop_scores, key=sop_scores.get)
        center_z = z_lookup.get(center_sop)
        masks = []
        annotators = []
        source_sops = []
        z_offsets = []
        ct_series = None
        for seg in segs:
            mask = seg.frames.get(center_sop)
            source_sop = center_sop if mask is not None else None
            z_offset = 0.0 if mask is not None else None
            if mask is None:
                # Fall back to the nearest annotated slice only when the SEG
                # gives physical z coordinates close to the canonical plane.
                if center_z is not None and seg.z_by_sop:
                    nearest = min(seg.z_by_sop, key=lambda s: abs(seg.z_by_sop[s] - center_z))
                    z_offset = float(seg.z_by_sop[nearest] - center_z)
                    if abs(z_offset) <= float(max_z_mm):
                        mask = seg.frames.get(nearest)
                        source_sop = nearest
            if mask is not None and mask.any():
                masks.append(mask.astype(bool))
                annotators.append(seg.annotator_id)
                source_sops.append(str(source_sop))
                z_offsets.append(float(z_offset or 0.0))
                ct_series = seg.ct_series_uid
                if source_sop == center_sop:
                    exact_slice_masks += 1
                else:
                    nearest_slice_masks += 1
            else:
                discarded_annotations += 1
        if len(masks) < int(min_annotations) or ct_series is None:
            continue
        usable_groups += 1
        ct_path = root / "raw/ct_slices" / f"{center_sop}.dcm"
        if not ct_path.is_file():
            raise FileNotFoundError(f"obtain the authorized CT slice first: {ct_path.name}")
        image = ct_to_uint8(ct_path)
        if image.shape != masks[0].shape:
            continue
        y0, y1, x0, x1 = bbox_from_masks(masks, pad=int(pad), min_size=int(min_crop))
        patient_id, study_uid, nodule_id = key
        task_id = f"{patient_id}_N{nodule_id}_{center_sop[-12:]}"
        image_path = out_root / "images" / f"{task_id}.png"
        save_png(image[y0:y1, x0:x1], image_path)
        mask_paths = []
        areas = []
        for j, mask in enumerate(masks):
            crop = mask[y0:y1, x0:x1]
            areas.append(float(crop.mean()))
            mask_path = out_root / "masks" / task_id / f"{task_id}_ann{j:02d}_{annotators[j]}.png"
            save_png((crop.astype(np.uint8) * 255), mask_path)
            mask_paths.append(str(mask_path))
        split = stable_split(patient_id, 0.70, 0.15)
        rows.append(
            {
                "image_id": task_id,
                "image_path": str(image_path),
                "split": split,
                "task_id": task_id,
                "class_or_object_id": "lung_nodule",
                "annotator_ids": json.dumps(annotators),
                "mask_paths": json.dumps(mask_paths),
                "num_masks": len(mask_paths),
                "metadata": json.dumps(
                    {
                        "dataset": "LIDC-IDRI QIICR DICOM-SEG",
                        "patient_id": patient_id,
                        "study_uid": study_uid,
                        "nodule_id": nodule_id,
                        "center_sop_uid": center_sop,
                        "center_z_mm": None if center_z is None else float(center_z),
                        "ct_series_uid": ct_series,
                        "crop_y0y1x0x1": [int(y0), int(y1), int(x0), int(x1)],
                        "area_frac_mean": float(np.mean(areas)),
                        "group_annotation_count": int(len(segs)),
                        "kept_annotation_count": int(len(mask_paths)),
                        "discarded_annotation_count": int(len(segs) - len(mask_paths)),
                        "source_sop_uids": source_sops,
                        "source_z_offsets_mm": z_offsets,
                    }
                ),
            }
        )
        if int(max_tasks) > 0 and len(rows) >= int(max_tasks):
            break
        if len(rows) % 200 == 0:
            log("e2201_manifest_progress", rows=len(rows), groups=len(groups))
    frame = pd.DataFrame(rows).sort_values(["split", "image_id"])
    out_root.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_root / "manifest.parquet", index=False)
    frame.to_csv(out_root / "manifest.csv", index=False)
    if len(frame):
        patient_split = (
            frame.assign(patient_id=frame["metadata"].map(lambda s: json.loads(str(s))["patient_id"]))
            .drop_duplicates(["split", "patient_id"])
            .groupby("split")["patient_id"]
            .nunique()
            .to_dict()
        )
        mask_dist = {str(k): int(v) for k, v in frame["num_masks"].value_counts().sort_index().to_dict().items()}
    else:
        patient_split = {}
        mask_dist = {}
    stats = {
        "rows": int(len(frame)),
        "splits": {str(k): int(v) for k, v in frame["split"].value_counts().to_dict().items()} if len(frame) else {},
        "patient_splits": {str(k): int(v) for k, v in patient_split.items()},
        "num_masks_min": int(frame["num_masks"].min()) if len(frame) else 0,
        "num_masks_mean": float(frame["num_masks"].mean()) if len(frame) else 0.0,
        "num_masks_max": int(frame["num_masks"].max()) if len(frame) else 0,
        "num_masks_distribution": mask_dist,
        "tasks_ge2_masks": int((frame["num_masks"] >= 2).sum()) if len(frame) else 0,
        "tasks_ge3_masks": int((frame["num_masks"] >= 3).sum()) if len(frame) else 0,
        "tasks_ge4_masks": int((frame["num_masks"] >= 4).sum()) if len(frame) else 0,
        "decoded_seg": int(len(seg_paths)),
        "groups": int(len(groups)),
        "usable_groups": int(usable_groups),
        "skipped_seg": int(skipped),
        "exact_slice_masks": int(exact_slice_masks),
        "nearest_slice_masks": int(nearest_slice_masks),
        "discarded_annotations": int(discarded_annotations),
        "max_z_mm": float(max_z_mm),
    }
    (out_root / "manifest_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True))
    log("e2201_lidc_manifest_done", manifest=str(out_root / "manifest.parquet"), **stats)
    return frame

def prepare_lidc(data_root, output):
    from .data import write_json
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    frame = build_rows(Path(data_root), output_root=output, min_annotations=2, pad=24,
                       min_crop=96, max_tasks=0, max_z_mm=1.5)
    rows = []
    for r in frame.to_dict("records"):
        metadata = json.loads(r["metadata"])
        rows.append(dict(image_id=r["image_id"], source_id=metadata["patient_id"], split=r["split"], endpoint="lidc",
                         image_path=str(Path(r["image_path"]).relative_to(output)),
                         mask_paths=[str(Path(p).relative_to(output)) for p in json.loads(r["mask_paths"])]))
    write_json(output/"all_annotations.json", rows)
    return len(rows)
