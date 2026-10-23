"""Publication panels selected from residual diagnostics before reading references."""
from pathlib import Path
from dataclasses import dataclass
from typing import Any,Iterable
import json
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image,ImageFilter,ImageOps
from .data import read_rows,within,sha256,write_json,prediction_probability
from .metrics import upsample_probability


@dataclass(frozen=True)
class EndpointSpec:
    endpoint: str
    label: str
    visible_crop: bool = False

SPECS = (
    EndpointSpec("lidc", "LIDC extent", visible_crop=True),
    EndpointSpec("ima", "IMA++ boundary"),
    EndpointSpec("riga_disc", "RIGA high agreement"),
    EndpointSpec("bsds", "BSDS structure"),
)

VISIBLE_COLOR = np.asarray((230, 159, 0), dtype=np.float32) / 255.0

PREDICTION_COLOR = np.asarray((0, 114, 178), dtype=np.float32) / 255.0

PREDICTION_EDGE = np.asarray((255, 255, 255), dtype=np.float32) / 255.0

HIDDEN_COLORS = np.asarray(
    (
        (204, 121, 167),
        (0, 158, 115),
        (213, 94, 0),
        (86, 180, 233),
        (240, 228, 66),
        (0, 114, 178),
        (230, 159, 0),
        (0, 0, 0),
    ),
    dtype=np.float32,
) / 255.0

def _median_selection(diagnostics: pd.DataFrame) -> tuple[str, int, float, float]:
    image_values = (
        diagnostics.groupby("image_id", sort=True)["anchor_fused_dice"]
        .mean()
        .sort_index()
    )
    median = float(image_values.median())
    image_id = sorted(
        image_values.index,
        key=lambda value: (abs(float(image_values.loc[value]) - median), str(value)),
    )[0]
    selected = diagnostics[diagnostics["image_id"].astype(str) == str(image_id)]
    head_values = selected.set_index("head_id")["anchor_fused_dice"]
    head_median = float(head_values.median())
    head_id = int(
        sorted(
            head_values.index,
            key=lambda value: (
                abs(float(head_values.loc[value]) - head_median),
                int(value),
            ),
        )[0]
    )
    return str(image_id), head_id, float(image_values.loc[image_id]), median

def _load_image(path: Path, height: int, width: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as handle:
        image = ImageOps.exif_transpose(handle).convert("RGB")
        image = image.resize((int(width), int(height)), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)

def _load_mask(path: Path, height: int, width: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as handle:
        mask = handle.convert("L").resize(
            (int(width), int(height)), Image.Resampling.NEAREST
        )
        return np.asarray(mask) > 0

def _visible_crop(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return a square LIDC crop fixed solely from the visible annotation."""

    height, width = mask.shape
    yy, xx = np.where(mask)
    if not len(xx):
        return 0, height, 0, width
    object_size = max(int(xx.max() - xx.min() + 1), int(yy.max() - yy.min() + 1))
    crop_size = min(min(height, width), max(48, int(np.ceil(3.2 * object_size))))
    cy = (float(yy.min()) + float(yy.max())) / 2.0
    cx = (float(xx.min()) + float(xx.max())) / 2.0
    top = int(round(cy - crop_size / 2))
    left = int(round(cx - crop_size / 2))
    top = min(max(top, 0), height - crop_size)
    left = min(max(left, 0), width - crop_size)
    return top, top + crop_size, left, left + crop_size

def _boundary(mask: np.ndarray) -> np.ndarray:
    inner = mask.copy()
    inner[1:, :] &= mask[:-1, :]
    inner[:-1, :] &= mask[1:, :]
    inner[:, 1:] &= mask[:, :-1]
    inner[:, :-1] &= mask[:, 1:]
    return mask & ~inner

def _thicken(mask: np.ndarray) -> np.ndarray:
    diameter = max(1, int(round(min(mask.shape) / 220)))
    diameter = 2 * diameter + 1
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(image.filter(ImageFilter.MaxFilter(diameter))) > 0

def _overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: np.ndarray,
    *,
    alpha: float,
    edge_color: np.ndarray | None = None,
) -> np.ndarray:
    output = image.astype(np.float32) / 255.0
    output[mask] = output[mask] * (1.0 - alpha) + color * alpha
    edge = _thicken(_boundary(mask))
    output[edge] = color if edge_color is None else edge_color
    return np.clip(output, 0.0, 1.0)

def _heldout_overlay(image: np.ndarray, masks: np.ndarray) -> np.ndarray:
    output = image.astype(np.float32) / 255.0
    fill_alpha = 0.24 if len(masks) == 1 else (0.055 if len(masks) <= 6 else 0.0)
    for index, mask in enumerate(masks):
        color = HIDDEN_COLORS[index % len(HIDDEN_COLORS)]
        if fill_alpha:
            output[mask] = output[mask] * (1.0 - fill_alpha) + color * fill_alpha
        output[_thicken(_boundary(mask))] = color
    return np.clip(output, 0.0, 1.0)

def _show(axis: Any, image: np.ndarray) -> None:
    axis.imshow(image)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)

def render(rows: Iterable[dict[str, Any]], output_path: Path) -> None:
    rows = list(rows)
    if len(rows) != len(SPECS):
        raise ValueError("qualitative figure row count differs from the fixed spec")
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )
    figure = plt.figure(figsize=(7.05, 3.72), constrained_layout=False)
    outer = figure.add_gridspec(
        len(rows),
        7,
        width_ratios=(1.05, 1.05, 1.05, 1.0, 1.0, 1.0, 1.0),
        wspace=0.028,
        hspace=0.025,
    )
    for row_index, row in enumerate(rows):
        image_axis = figure.add_subplot(outer[row_index, 0])
        visible_axis = figure.add_subplot(outer[row_index, 1])
        hidden_axis = figure.add_subplot(outer[row_index, 2])
        _show(image_axis, row["image"])
        _show(
            visible_axis,
            _overlay(
                row["image"],
                row["visible"],
                VISIBLE_COLOR,
                alpha=0.44,
                edge_color=np.ones(3, dtype=np.float32),
            ),
        )
        _show(hidden_axis, _heldout_overlay(row["image"], row["hidden"]))
        label = row["label"] + (
            " (zoom)" if row["crop_basis"] != "full reference grid" else ""
        )
        image_axis.set_ylabel(label, fontsize=6.6, labelpad=4.5)
        if row_index == 0:
            image_axis.set_title("Image", fontsize=7.5, pad=2.5)
            visible_axis.set_title("Visible", fontsize=7.5, pad=2.5)
            hidden_axis.set_title("Held-out", fontsize=7.5, pad=2.5)

        for head_index, mask in enumerate(row["bank"][:4]):
            axis = figure.add_subplot(outer[row_index, head_index + 3])
            _show(
                axis,
                _overlay(
                    row["image"],
                    mask,
                    PREDICTION_COLOR,
                    alpha=0.42,
                    edge_color=PREDICTION_EDGE,
                ),
            )
            axis.text(
                0.035,
                0.955,
                str(head_index + 1),
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=5.7,
                color="white",
                bbox={
                    "facecolor": "black",
                    "edgecolor": "none",
                    "alpha": 0.72,
                    "pad": 0.65,
                },
            )
            if row_index == 0:
                axis.set_title(f"SSRB {head_index + 1}", fontsize=7.5, pad=2.5)
    figure.subplots_adjust(left=0.108, right=0.998, bottom=0.012, top=0.972)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight", pad_inches=0.10)
    plt.close(figure)

def main(source,output):
    source,output=Path(source),Path(output)
    def resolve(name):
        path=Path(name)
        return path if path.is_absolute() else source.resolve().parent/path
    payload=json.loads(source.read_text())
    cases={r['endpoint']:r for r in payload}
    if len(payload)!=len(SPECS) or set(cases)!={s.endpoint for s in SPECS}:
        raise ValueError('requires LIDC, IMA++, RIGA disc and BSDS cases')
    rows=[];records=[]
    for spec in SPECS:
        case=cases[spec.endpoint]
        directory=resolve(case['diagnostics'])
        provenance=json.loads((directory/'provenance.json').read_text())
        if provenance.get('reference_masks_read') is not False:
            raise ValueError('selection diagnostics must precede reference reads')
        diagnostics=pd.DataFrame(json.loads((directory/'head_diagnostics.json').read_text()))
        if set(diagnostics.endpoint)!={spec.endpoint} or set(diagnostics.visible_seed)!={0} or set(diagnostics.optimizer_seed)!={0}:
            raise ValueError('figure selection requires matching visible/optimizer zero diagnostics')
        image_id,head,agreement,median=_median_selection(diagnostics)
        # The selected identity is fixed before the reference manifest is opened.
        selected_id=str(image_id)
        predictions=resolve(case['predictions'])
        frozen=json.loads((predictions/'frozen.json').read_text())
        if sha256(predictions/'predictions.pt')!=frozen['predictions_sha256']:
            raise ValueError('prediction bank changed')
        if not provenance.get('checkpoint_sha256') or provenance['checkpoint_sha256']!=frozen.get('checkpoint_sha256'):
            raise ValueError('diagnostic/prediction checkpoints differ')
        bank=torch.load(predictions/'predictions.pt',map_location='cpu',weights_only=True)
        manifest=resolve(case['manifest']);identities=read_rows(manifest)
        match=[r for r in identities if str(r['image_id'])==selected_id]
        if len(match)!=1 or match[0]['endpoint']!=spec.endpoint or match[0]['visible_seed']!=0 or match[0]['split']!='test':
            raise ValueError('selected image is not one matching test case')
        identity=match[0];reference_root=resolve(case['reference_root'])
        hidden_paths=[within(reference_root,name) for name in identity['heldout_mask_paths']]
        if not hidden_paths:
            raise ValueError('qualitative reference set is empty')
        with Image.open(hidden_paths[0]) as handle:width,height=handle.size
        probability=prediction_probability(bank[selected_id])
        masks=(upsample_probability(probability,(height,width))>=.5).cpu().numpy()
        if masks.shape!=(8,height,width):
            raise ValueError('requires the preserved ranked eight-mask bank')
        image_manifest=resolve(case['image_manifest']) if 'image_manifest' in case else resolve(case['root'])/'images.json'
        image_rows=[r for r in read_rows(image_manifest) if r['image_id']==selected_id]
        if len(image_rows)!=1 or any(image_rows[0][k]!=identity[k] for k in ('endpoint','visible_seed','source_id','split')):
            raise ValueError('image and reference identities differ')
        image_path=within(resolve(case['root']),image_rows[0]['image_path'])
        image=_load_image(image_path,height,width)
        visible=_load_mask(within(reference_root,identity['visible_mask_path']),height,width)
        hidden=np.stack([_load_mask(path,height,width) for path in hidden_paths])
        crop=(0,height,0,width);basis='full reference grid'
        if spec.visible_crop:
            candidate=_visible_crop(visible)
            if candidate!=crop:crop=candidate;basis='square crop determined only from visible annotation'
        top,bottom,left,right=crop
        rows.append(dict(label=spec.label,crop_basis=basis,image=image[top:bottom,left:right],
            visible=visible[top:bottom,left:right],hidden=hidden[:,top:bottom,left:right],bank=masks[:,top:bottom,left:right]))
        records.append(dict(endpoint=spec.endpoint,image_id=selected_id,representative_head=head,
            selection_anchor_fused_dice_mean=agreement,selection_endpoint_median=median,
            selection_fixed_before_reference_manifest_read=True,predictions_sha256=frozen['predictions_sha256'],
            checkpoint_sha256=frozen['checkpoint_sha256'],crop_reference_grid_tblr=list(crop)))
    render(rows,output)
    write_json(output.with_suffix('.record.json'),dict(profile='qualitative',source_sha256=sha256(source),
        output_sha256=sha256(output),reference_masks_used_for_rendering=True,reference_masks_used_for_selection=False,cases=records))
