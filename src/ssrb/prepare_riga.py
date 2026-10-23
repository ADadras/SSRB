from __future__ import annotations
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from PIL import Image

def stable_float(value: str) -> float:
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / float(16**8)

def safe_id(path: str, task: str) -> str:
    stem = path.replace("RIGA/", "").replace("/", "_").replace(".", "_")
    return f"RIGA_{stem}_{task}"

def read_csv_from_zip(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    with zf.open(name) as f:
        return pd.read_csv(f).assign(csv_name=name)

def save_image_from_zip(zf: zipfile.ZipFile, rel: str, out: Path) -> None:
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(rel) as f:
        image = Image.open(f).convert("RGB")
        image.save(out)

def save_mask(mask: np.ndarray, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(out)

def split_for(csv_name: str, image_rel: str, val_frac: float) -> str:
    if csv_name.endswith("_test.csv"):
        return "test"
    if csv_name.endswith("_train.csv"):
        return "val" if stable_float(image_rel) < float(val_frac) else "train"
    raise ValueError(f"unexpected labeled csv: {csv_name}")

def build_manifest(zip_path: Path, out_dir: Path, *, val_frac: float) -> pd.DataFrame:
    csv_names = [
        "BinRushed_train.csv",
        "BinRushed_test.csv",
        "Magrabia_train.csv",
        "Magrabia_test.csv",
        "MESSIDOR_Base1_train.csv",
        "MESSIDOR_Base1_test.csv",
        "MESSIDOR_Base2_train.csv",
        "MESSIDOR_Base2_test.csv",
        "MESSIDOR_Base3_train.csv",
        "MESSIDOR_Base3_test.csv",
    ]
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for csv_name in csv_names:
            frame = read_csv_from_zip(zf, csv_name)
            for rec in frame.to_dict("records"):
                image_rel = str(rec["image"])
                mask_base_rel = str(rec["mask"])
                split = split_for(csv_name, image_rel, float(val_frac))
                image_out = out_dir / "images" / image_rel.replace("RIGA/", "")
                save_image_from_zip(zf, image_rel, image_out)
                mask_paths: dict[str, list[str]] = {"optic_disc": [], "optic_cup": []}
                annotator_ids: list[str] = []
                base_path = Path(mask_base_rel)
                for grader in range(1, 7):
                    grader_rel = str(base_path.with_name(f"{base_path.stem}-{grader}{base_path.suffix}"))
                    if grader_rel not in zf.namelist():
                        continue
                    with zf.open(grader_rel) as f:
                        arr = np.asarray(Image.open(f).convert("L"))
                    # RIGA/RIGA+ encodes background=0, neuroretinal rim/disc
                    # region=128, and optic cup=255.
                    disc = arr >= 128
                    cup = arr == 255
                    ann_id = f"G{grader}"
                    annotator_ids.append(ann_id)
                    for task, binary in [("optic_disc", disc), ("optic_cup", cup)]:
                        mask_out = out_dir / "masks" / task / image_rel.replace("RIGA/", "").replace(".tif", f"_{ann_id}.png")
                        save_mask(binary, mask_out)
                        mask_paths[task].append(str(mask_out))
                if len(annotator_ids) < 2:
                    continue
                for task in ["optic_disc", "optic_cup"]:
                    if len(mask_paths[task]) < 2:
                        continue
                    rows.append(
                        {
                            "image_id": safe_id(image_rel, task),
                            "image_path": str(image_out),
                            "split": split,
                            "task_id": safe_id(image_rel, task),
                            "class_or_object_id": task,
                            "annotator_ids": json.dumps(annotator_ids),
                            "mask_paths": json.dumps(mask_paths[task]),
                            "num_masks": len(mask_paths[task]),
                            "metadata": json.dumps(
                                {
                                    "dataset": "RIGA+",
                                    "source_csv": csv_name,
                                    "image_rel": image_rel,
                                    "mask_encoding": "disc=(value>=128), cup=(value==255), rim_only=(value==128)",
                                }
                            ),
                        }
                    )
    return pd.DataFrame(rows).sort_values(["split", "image_id", "class_or_object_id"])

def prepare_riga(zip_path, output):
    from .data import write_json
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    frame = build_manifest(Path(zip_path), output, val_frac=.20)
    rows = []
    for r in frame.to_dict("records"):
        meta = json.loads(r["metadata"])
        rows.append(dict(image_id=r["image_id"], source_id=meta["image_rel"], split=r["split"],
                         endpoint="riga_disc" if r["class_or_object_id"]=="optic_disc" else "riga_cup",
                         image_path=str(Path(r["image_path"]).relative_to(output)),
                         mask_paths=[str(Path(p).relative_to(output)) for p in json.loads(r["mask_paths"])]))
    for endpoint in ("riga_disc", "riga_cup"):
        write_json(output/f"{endpoint}.json", [r for r in rows if r["endpoint"]==endpoint])
    # The original feature cache batches both tasks together. Keep those batch
    # boundaries while exposing only image metadata to the feature extractor.
    write_json(output/"feature_images.json", feature_image_rows(rows))
    return len(rows)


def feature_image_rows(rows):
    return [{k:r[k] for k in ("image_id", "source_id", "split", "endpoint", "image_path")} | {"visible_seed":0}
            for r in rows]
