"""BSDS construction example using the original six-panel layout."""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageOps
from .data import read_rows,within


def _fit_image_to_canvas(path: str, canvas_size: int = 220) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = ImageOps.fit(image, (canvas_size, canvas_size), method=Image.Resampling.LANCZOS)
    return np.asarray(image)

def _load_mask(path: str, *, canvas_size: int) -> np.ndarray:
    img = Image.open(path).convert("L")
    img = ImageOps.fit(img, (canvas_size, canvas_size), method=Image.Resampling.NEAREST)
    return (np.asarray(img) > 127).astype(np.float32)

def _overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = image.astype(np.float32) / 255.0
    color = np.array([0.0, 0.85, 1.0], dtype=np.float32)
    out = np.where(mask[..., None] > 0, 0.45 * out + 0.55 * color, out)
    edge = mask - np.minimum.reduce(
        [
            np.roll(mask, 1, 0),
            np.roll(mask, -1, 0),
            np.roll(mask, 1, 1),
            np.roll(mask, -1, 1),
        ]
    )
    out[edge > 0] = np.array([1.0, 1.0, 1.0])
    return np.clip(out, 0, 1)

def main(source,output):
    source,output=Path(source),Path(output)
    payload=json.loads(source.read_text())
    def resolve(name):
        path=Path(name)
        return path if path.is_absolute() else source.resolve().parent/path
    root=resolve(payload['root'])
    rows=[r for r in read_rows(resolve(payload['manifest'])) if r['image_id']==payload['image_id']]
    if len(rows)!=1 or rows[0]['endpoint']!='bsds' or rows[0]['split']!='test':
        raise ValueError('requires one BSDS test image from the native converter')
    row=rows[0]
    output.parent.mkdir(parents=True,exist_ok=True)
    image_arr = _fit_image_to_canvas(str(within(root,row["image_path"])), canvas_size=300)
    paths = [str(within(root,name)) for name in row["mask_paths"][:5]]
    panels = [image_arr.astype(np.float32) / 255.0]
    for path in paths:
        mask = _load_mask(path, canvas_size=300)
        panels.append(_overlay(image_arr, mask))
    gutter = 8
    composite = np.ones((2 * 300 + gutter, 3 * 300 + 2 * gutter, 3), dtype=np.float32)
    for idx, panel in enumerate(panels):
        row_idx = idx // 3
        col_idx = idx % 3
        y0 = row_idx * (300 + gutter)
        x0 = col_idx * (300 + gutter)
        composite[y0 : y0 + 300, x0 : x0 + 300] = panel
    fig_path = output
    Image.fromarray((np.clip(composite, 0, 1) * 255).astype(np.uint8)).save(fig_path, "PDF", resolution=150.0)
