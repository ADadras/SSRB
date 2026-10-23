from __future__ import annotations
import json
import shutil
from pathlib import Path
from typing import Any
import pandas as pd

def log(event: str, **kwargs: Any) -> None:
    print(json.dumps({"event": event, **kwargs}, sort_keys=True), flush=True)

def _normalize_isic_id(value: str) -> str:
    stem = Path(str(value).strip()).stem
    return stem

def _split_image_ids(split_csv: Path) -> list[str]:
    df = pd.read_csv(split_csv)
    for col in ("ISIC_id", "isic_id", "image_id", "image", "id"):
        if col in df.columns:
            return [_normalize_isic_id(x) for x in df[col].astype(str).tolist()]
    if len(df.columns) == 1:
        return [_normalize_isic_id(x) for x in df.iloc[:, 0].astype(str).tolist()]
    raise ValueError(f"cannot infer image id column in {split_csv}: {list(df.columns)}")

def build_ima_manifest(dataset_dir: Path, output: Path, limit_images: int = 0) -> tuple[Path, dict[str, Any]]:
    raw_dir = dataset_dir / "raw"
    seg_root = dataset_dir / "segs"
    image_dir = dataset_dir / "images"
    out_dir = output
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_meta = pd.read_csv(raw_dir / "seg_metadata_multiannotator_subset.csv")
    rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}

    for split in ("train", "val", "test"):
        ids = _split_image_ids(raw_dir / f"{split}.csv")
        if limit_images > 0:
            ids = ids[:limit_images]
        split_counts[split] = len(ids)
        split_meta = seg_meta[seg_meta["ISIC_id"].astype(str).isin(set(ids))].copy()
        if "annotator" in split_meta.columns:
            split_meta = split_meta[split_meta["annotator"].astype(str).str.startswith("A")]
        for isic_id, g in split_meta.groupby("ISIC_id", sort=True):
            image_path = image_dir / f"{isic_id}.jpg"
            mask_paths: list[str] = []
            annotator_ids: list[str] = []
            for _, r in g.sort_values("seg_filename").iterrows():
                mask_path = seg_root / str(r["seg_filename"])
                if mask_path.exists():
                    mask_paths.append(str(mask_path))
                    annotator_ids.append(str(r.get("annotator", "")))
            if len(mask_paths) < 2:
                continue
            rows.append(
                {
                    "image_id": str(isic_id),
                    "image_path": str(image_path),
                    "split": split,
                    "task_id": str(isic_id),
                    "class_or_object_id": "lesion",
                    "annotator_ids": json.dumps(annotator_ids),
                    "mask_paths": json.dumps(mask_paths),
                    "num_masks": len(mask_paths),
                    "metadata": json.dumps({"dataset": "IMA++", "source": "ISIC Archive collection 482"}),
                }
            )
    manifest = pd.DataFrame(rows)
    out_path = out_dir / "manifest.parquet"
    manifest.to_parquet(out_path, index=False)
    csv_path = out_dir / "manifest.csv"
    manifest.to_csv(csv_path, index=False)
    stats = {
        "rows": int(len(manifest)),
        "split_counts_from_split_files": split_counts,
        "manifest_split_counts": manifest["split"].value_counts().to_dict() if len(manifest) else {},
        "min_masks": int(manifest["num_masks"].min()) if len(manifest) else 0,
        "max_masks": int(manifest["num_masks"].max()) if len(manifest) else 0,
        "mean_masks": float(manifest["num_masks"].mean()) if len(manifest) else 0.0,
        "missing_images": int(sum(not Path(p).exists() for p in manifest["image_path"])) if len(manifest) else 0,
    }
    (out_dir / "manifest_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    log("ima_manifest_done", path=str(out_path), **stats)
    return out_path, stats

def prepare_ima(data_root, output):
    from .data import write_json
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    file, stats = build_ima_manifest(Path(data_root).resolve(), output)
    frame = pd.read_parquet(file)
    rows = []
    for r in frame.to_dict("records"):
        image_name = "images/" + Path(r["image_path"]).name
        (output/"images").mkdir(exist_ok=True)
        (output/"masks").mkdir(exist_ok=True)
        shutil.copyfile(r["image_path"], output/image_name)
        masks = []
        for j, p in enumerate(json.loads(r["mask_paths"])):
            name = f"masks/{r['image_id']}_{j}.png"
            shutil.copyfile(p, output/name)
            masks.append(name)
        rows.append(dict(image_id=r["image_id"], source_id=r["image_id"], split=r["split"], endpoint="ima",
                         image_path=image_name, mask_paths=masks))
    write_json(output/"all_annotations.json", rows)
    return stats
