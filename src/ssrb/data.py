"""Portable manifests with one fixed training target per image."""
from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = np.asarray([.485, .456, .406], dtype=np.float32)
IMAGENET_STD = np.asarray([.229, .224, .225], dtype=np.float32)
VISIBLE_KEYS = {"image_id", "image_path", "visible_mask_path", "source_id", "split", "endpoint", "visible_seed"}
IMAGE_KEYS = {"image_id", "image_path", "source_id", "split", "endpoint", "visible_seed"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_rows(path):
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest must be a nonempty JSON list")
    return rows


def within(root, value):
    root = Path(root).resolve()
    if Path(value).is_absolute():
        raise ValueError("manifest paths must be relative to the data root")
    p = (root / value).resolve()
    if not p.is_relative_to(root):
        raise ValueError("manifest path escapes the data root")
    return p


def validate_visible(rows):
    for row in rows:
        if set(row) != VISIBLE_KEYS:
            raise ValueError("visible manifest must contain exactly the permitted columns")
        if row["split"] not in {"train", "val"}:
            raise ValueError("trainer accepts only train and val")
    if {r["split"] for r in rows} != {"train", "val"}:
        raise ValueError("both train and val are required")
    if len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate image IDs")
    if len({r["endpoint"] for r in rows}) != 1 or len({r["visible_seed"] for r in rows}) != 1:
        raise ValueError("one run requires one endpoint and one visible seed")
    groups = {}
    for r in rows:
        groups.setdefault(r["source_id"], set()).add(r["split"])
    if any(len(s) != 1 for s in groups.values()):
        raise ValueError("a source occurs in multiple splits")


def seal_inputs(root, manifest):
    root = Path(root).resolve()
    rows = read_rows(manifest)
    validate_visible(rows)
    names = {r[k] for r in rows for k in ("image_path", "visible_mask_path")}
    names.update({"train_features.pt", "val_features.pt"})
    seal = {name: sha256(within(root, name)) for name in sorted(names)}
    seal["manifest_sha256"] = sha256(manifest)
    write_json(root / "inputs.sha256.json", seal)
    return seal


def verify_inputs(root, manifest):
    root = Path(root)
    rows = read_rows(manifest)
    validate_visible(rows)
    seal = json.loads((root / "inputs.sha256.json").read_text())
    expected_names = {r[k] for r in rows for k in ("image_path", "visible_mask_path")}
    expected_names.update({"train_features.pt", "val_features.pt", "manifest_sha256"})
    if set(seal) != expected_names or seal["manifest_sha256"] != sha256(manifest):
        raise ValueError("manifest seal mismatch")
    for name, expected in seal.items():
        if name != "manifest_sha256" and sha256(within(root, name)) != expected:
            raise ValueError(f"input hash changed: {name}")
    return rows, seal


def load_image(path, image_size):
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    return torch.from_numpy((array - IMAGENET_MEAN) / IMAGENET_STD).permute(2, 0, 1).float()


def load_visible_mask(path, size_hw):
    with Image.open(path) as image:
        array = np.asarray(image.convert("L").resize((size_hw[1], size_hw[0]), Image.Resampling.NEAREST), dtype=np.float32)
    if array.max(initial=0.0) > 1.0:
        array = array / 255.0
    return torch.from_numpy((array > .5).astype(np.float32))


class VisibleDataset(Dataset):
    def __init__(self, root, rows, payload, split, *, residual=False, image_size=128):
        self.root = Path(root)
        self.rows = sorted([r for r in rows if r["split"] == split], key=lambda r: r["image_id"])
        self.features = payload["features"]
        self.index = {str(i): j for j, i in enumerate(payload["image_ids"])}
        self.grid = tuple(payload["grid_hw"])
        self.residual = residual
        self.image_size = image_size
        self._input_cache = None
        if set(payload) != {"features", "image_ids", "grid_hw"}:
            raise ValueError("feature payload contains unexpected metadata")
        if any(r["image_id"] not in self.index for r in self.rows):
            raise ValueError("missing features")
        if self.features.ndim != 4 or self.features.shape[2] != self.grid[0] * self.grid[1]:
            raise ValueError("invalid feature grid")

    def __len__(self):
        return len(self.rows)

    def _inputs(self, row):
        result = {"target": load_visible_mask(within(self.root, row["visible_mask_path"]),
                                              (self.image_size, self.image_size) if self.residual else self.grid)}
        if self.residual:
            result["image"] = load_image(within(self.root, row["image_path"]), self.image_size)
        return result

    def preload_inputs(self):
        if self._input_cache is None:
            self._input_cache = [self._inputs(row) for row in self.rows]
        return sum(t.numel()*t.element_size() for record in self._input_cache for t in record.values())

    def __getitem__(self, index):
        row = self.rows[index]
        inputs = self._inputs(row) if self._input_cache is None else self._input_cache[index]
        result = {"features": self.features[self.index[row["image_id"]]].float(),
                  "target": inputs["target"], "image_id": row["image_id"]}
        if self.residual:
            result["image"] = inputs["image"]
        return result


def freeze_manifests(source, data_root, output, seed):
    """Preprocessing reads authorized annotations; the trainer gets separate copies."""
    rows = read_rows(source)
    data_root, output = Path(data_root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    visible, test_images, hidden = [], [], []
    groups = {}
    for r in rows:
        groups.setdefault((r["endpoint"], r["source_id"]), set()).add(r["split"])
    if any(len(v) > 1 for v in groups.values()):
        raise ValueError("a source occurs in multiple splits")
    if len({(r["endpoint"], r["image_id"]) for r in rows}) != len(rows):
        raise ValueError("duplicate task/image IDs")
    for n, r in enumerate(sorted(rows, key=lambda x: x["image_id"])):
        masks = r["mask_paths"]
        if len(masks) < 2:
            raise ValueError("at least two masks per task are required")
        index = (int(hashlib.sha1(r["image_id"].encode()).hexdigest()[:8], 16) + seed) % len(masks)
        split = r["split"]
        if split not in {"train", "val", "test"}:
            raise ValueError("unknown split")
        dest = output / ("trainer" if split != "test" else "inference")
        image_name = f"images/{n:06d}{Path(r['image_path']).suffix.lower()}"
        (dest / "images").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(within(data_root, r["image_path"]), dest / image_name)
        base = {"image_id": r["image_id"], "source_id": r["source_id"], "image_path": image_name,
                "split": split, "endpoint": r["endpoint"], "visible_seed": seed}
        if split != "test":
            name = f"masks/{n:06d}.png"
            (dest / "masks").mkdir(exist_ok=True)
            shutil.copyfile(within(data_root, masks[index]), dest / name)
            visible.append(base | {"visible_mask_path": name})
        else:
            test_images.append(base)
            references = []
            visible_name = None
            (output / "evaluator/masks").mkdir(parents=True, exist_ok=True)
            for j, m in enumerate(masks):
                name = f"masks/{n:06d}_{j}.png"
                shutil.copyfile(within(data_root, m), output / "evaluator" / name)
                if j != index:
                    references.append(name)
                else:
                    visible_name = name
            hidden.append({k: v for k, v in base.items() if k != "image_path"} |
                          {"heldout_mask_paths": references, "visible_mask_path": visible_name})
    validate_visible(visible)
    write_json(output / "trainer/visible.json", visible)
    write_json(output / "inference/images.json", test_images)
    write_json(output / "evaluator/hidden.json", hidden)
    return {"train": sum(r["split"] == "train" for r in visible),
            "val": sum(r["split"] == "val" for r in visible), "test": len(hidden)}


def pack_binary_prediction(masks, scores):
    """Lossless encoding for native binary proposals, which can be very large."""
    from .metrics import encode_binary_rle
    if masks.dtype != torch.bool or masks.ndim != 3 or len(scores) != len(masks):
        raise ValueError('packing requires binary K,H,W masks and K scores')
    return dict(binary_rle=[encode_binary_rle(m) for m in masks],
                image_hw=list(masks.shape[-2:]), scores=scores.detach().cpu())


def prediction_probability(record):
    """Read ordinary probabilities or losslessly encoded binary proposals."""
    if 'probability' in record:
        return record['probability']
    from .metrics import decode_binary_rle
    if set(record) != {'binary_rle','image_hw','scores'}:
        raise ValueError('unsupported prediction record')
    shape=tuple(record['image_hw'])
    if len(shape)!=2 or min(shape)<1 or len(record['scores'])!=len(record['binary_rle']):
        raise ValueError('invalid binary proposal dimensions')
    masks=[decode_binary_rle(r) for r in record['binary_rle']]
    if any(tuple(m.shape)!=shape for m in masks):
        raise ValueError('binary proposal grids differ')
    return torch.stack(masks) if masks else torch.empty(0,*shape,dtype=torch.bool)
