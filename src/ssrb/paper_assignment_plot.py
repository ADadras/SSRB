"""Assignment examples: select from evaluated effects, then render."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import json
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from .analysis import REGIMES
from .data import read_rows, within, sha256, write_json, prediction_probability
from .metrics import upsample_probability


def portable_frame(evaluation,method):
    evaluation=Path(evaluation)
    frame=pd.DataFrame(json.loads((evaluation/'per_image.json').read_text()))
    frame['regime']=frame.endpoint.map(REGIMES)
    if frame.regime.isna().any():
        raise ValueError('unknown qualitative endpoint')
    frame['subset']=np.where(frame.endpoint=='lidc_hard','visible_dice_lt_0.80','all_hidden')
    frame['endpoint']=frame.endpoint.replace({'lidc_hard':'lidc'})
    frame['method']=method;frame['evaluation_file']=str(evaluation.resolve())
    return frame


def main(source,output):
    source,output=Path(source),Path(output)
    def resolve(value):
        p=Path(value)
        return p if p.is_absolute() else source.resolve().parent/p
    specs=json.loads(source.read_text())
    frames=[];by_evaluation={};inputs=[]
    for spec in specs:
        evaluation=resolve(spec['evaluation'])
        key=str(evaluation.resolve())
        if key in by_evaluation or spec['method'] not in ('dmb_rn','dmb_wta') or int(spec['optimizer_seed'])!=0:
            raise ValueError('assignment panels require unique RN/WTA evaluations at optimizer seed zero')
        frame=portable_frame(evaluation,spec['method'])
        frames.append(frame);by_evaluation[key]=spec
        inputs.append(dict(evaluation_sha256=sha256(evaluation/'per_image.json'),method=spec['method']))
    frame=pd.concat(frames,ignore_index=True)
    if frame.duplicated(IDENTITY+['method']).any():
        raise ValueError('duplicate qualitative comparison units')
    _,panels=choose_panels(frame)
    rows=[]
    for panel in PANEL_ORDER:
        record=panels[panel]
        for selected in record['images']:
            image_id=selected['image_id'];banks=[];reference=None;reference_spec=None
            for method in ('wta','rn'):
                spec=by_evaluation[selected[method+'_evaluation_file']]
                pred=resolve(spec['predictions']);frozen=json.loads((pred/'frozen.json').read_text())
                if sha256(pred/'predictions.pt')!=frozen['predictions_sha256']:
                    raise ValueError('prediction bank changed')
                manifest=resolve(spec['manifest'])
                match=[r for r in read_rows(manifest) if r['image_id']==image_id and r['visible_seed']==selected['visible_seed']]
                if len(match)!=1:
                    raise ValueError('selected reference identity is ambiguous or absent')
                if reference is not None and (reference!=match[0] or resolve(reference_spec['reference_root'])!=resolve(spec['reference_root'])):
                    raise ValueError('RN and WTA references differ')
                reference=match[0];reference_spec=spec
                bank=torch.load(pred/'predictions.pt',map_location='cpu',weights_only=True)
                banks.append(prediction_probability(bank[image_id]))
            root=resolve(reference_spec['reference_root'])
            def mask(name):
                with Image.open(within(root,name)) as im:
                    return np.asarray(im.convert('L'))>0
            visible=mask(reference['visible_mask_path'])
            hidden=np.stack([mask(name) for name in reference['heldout_mask_paths']])
            height,width=visible.shape
            banks=[(upsample_probability(p,(height,width))>=.5).cpu() for p in banks]
            if any(tuple(p.shape)!=(8,height,width) for p in banks) or tuple(hidden.shape[1:])!=(height,width):
                raise ValueError('assignment banks and annotations have incompatible dimensions')
            image_manifest=resolve(reference_spec['image_manifest']) if 'image_manifest' in reference_spec else resolve(reference_spec['root'])/'images.json'
            image_rows=[r for r in read_rows(image_manifest) if r['image_id']==image_id]
            if len(image_rows)!=1 or any(image_rows[0][k]!=reference[k] for k in ('endpoint','visible_seed','source_id','split')):
                raise ValueError('image and reference identities differ')
            image=_rgb_on_grid(within(resolve(reference_spec['root']),image_rows[0]['image_path']),height,width)
            rows.append(dict(panel=panel,regime=selected['regime'],source_id=record['source_id'],image_id=image_id,visible_seed=selected['visible_seed'],source_effect=record['source_effect'],image_effect=selected['image_effect'],image=image,visible=visible,hidden=hidden,wta_bank=banks[0],rn_bank=banks[1]))
    output.parent.mkdir(parents=True,exist_ok=True)
    render(rows,output,output.with_suffix('.png'))
    write_json(output.with_suffix('.record.json'),dict(profile='assignment-examples',selection_rule=SELECTION_RULE,reference_evaluation_used_for_selection=True,reference_masks_used_for_rendering=True,panels=panels,inputs=inputs))


IDENTITY = [
    "endpoint",
    "subset",
    "visible_seed",
    "image_id",
    "source_id",
    "regime",
]

SELECTION_RULE = (
    "median_source_RN_minus_WTA_MLQ_stable_source_then_image_tie_break"
)

def paired_image_rows(frame: pd.DataFrame) -> pd.DataFrame:
    required = set(IDENTITY + ["method", "MLQ@8", "evaluation_file"])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"evaluation rows lack qualitative-selection fields: {missing}")
    left = frame[frame["method"].astype(str) == "dmb_rn"][
        IDENTITY + ["MLQ@8", "evaluation_file"]
    ]
    right = frame[frame["method"].astype(str) == "dmb_wta"][
        IDENTITY + ["MLQ@8", "evaluation_file"]
    ]
    paired = left.merge(
        right,
        on=IDENTITY,
        how="inner",
        suffixes=("_rn", "_wta"),
        validate="one_to_one",
    )
    paired["difference"] = paired["MLQ@8_rn"].astype(float) - paired[
        "MLQ@8_wta"
    ].astype(float)
    if not np.isfinite(paired["difference"]).all():
        raise ValueError("qualitative-selection differences are not finite")
    return paired

def representative_source(
    rows: pd.DataFrame,
    *,
    panel: str,
    required_regimes: tuple[str, ...],
    required_seeds: tuple[int, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    subset = rows[
        rows["regime"].astype(str).isin(required_regimes)
        & rows["visible_seed"].astype(int).isin(required_seeds)
    ].copy()
    if subset.empty:
        raise ValueError(f"no paired rows for qualitative panel {panel}")
    subset["regime_seed_cell"] = list(
        zip(subset["regime"].astype(str), subset["visible_seed"].astype(int))
    )
    summary = (
        subset.groupby("source_id", as_index=False)
        .agg(
            effect=("difference", "mean"),
            image_rows=("image_id", "size"),
            visible_seeds=("visible_seed", "nunique"),
            regimes=("regime", "nunique"),
            regime_seed_cells=("regime_seed_cell", "nunique"),
        )
        .sort_values("source_id", kind="stable")
    )
    summary = summary[
        (summary["visible_seeds"] == len(required_seeds))
        & (summary["regimes"] == len(required_regimes))
        & (
            summary["regime_seed_cells"]
            == len(required_seeds) * len(required_regimes)
        )
    ].copy()
    if summary.empty:
        raise ValueError(f"no complete sources for qualitative panel {panel}")
    median = float(summary["effect"].median())
    summary["panel"] = panel
    summary["median_effect"] = median
    summary["distance_to_median"] = (summary["effect"] - median).abs()
    summary = summary.sort_values(
        ["distance_to_median", "source_id"], kind="stable"
    ).reset_index(drop=True)
    summary["chosen"] = False
    summary.loc[0, "chosen"] = True
    selected_source = str(summary.loc[0, "source_id"])
    selected_effect = float(summary.loc[0, "effect"])
    source_rows = subset[subset["source_id"].astype(str) == selected_source].copy()
    return summary, {
        "panel": panel,
        "source_id": selected_source,
        "source_effect": selected_effect,
        "median_effect": median,
        "distance_to_median": float(abs(selected_effect - median)),
        "candidate_sources": int(len(summary)),
        "required_regimes": list(required_regimes),
        "required_visible_seeds": list(required_seeds),
        "source_rows": source_rows,
    }

def representative_image(
    rows: pd.DataFrame,
    *,
    source_effect: float,
    visible_seed: int,
    regime: str,
) -> dict[str, Any]:
    candidates = rows[
        (rows["visible_seed"].astype(int) == int(visible_seed))
        & (rows["regime"].astype(str) == str(regime))
    ].copy()
    if candidates.empty:
        raise ValueError(
            f"selected source lacks seed {visible_seed} and regime {regime}"
        )
    candidates["distance_to_source_effect"] = (
        candidates["difference"].astype(float) - float(source_effect)
    ).abs()
    chosen = candidates.sort_values(
        ["distance_to_source_effect", "image_id"], kind="stable"
    ).iloc[0]
    return {
        "regime": str(regime),
        "visible_seed": int(visible_seed),
        "image_id": str(chosen["image_id"]),
        "image_effect": float(chosen["difference"]),
        "rn_evaluation_file": str(chosen["evaluation_file_rn"]),
        "wta_evaluation_file": str(chosen["evaluation_file_wta"]),
    }

def choose_panels(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    paired = paired_image_rows(frame)
    candidate_tables: list[pd.DataFrame] = []
    selections: dict[str, Any] = {}

    lidc_seed = None
    for seed in (0, 1, 2):
        available = paired[
            (paired["regime"].astype(str) == "LIDC-hard")
            & (paired["visible_seed"].astype(int) == seed)
        ]
        if not available.empty:
            lidc_seed = seed
            break
    if lidc_seed is None:
        raise ValueError("no LIDC-hard rows are available for qualitative selection")
    candidates, selected = representative_source(
        paired,
        panel="lidc_hard",
        required_regimes=("LIDC-hard",),
        required_seeds=(lidc_seed,),
    )
    candidate_tables.append(candidates)
    source_rows = selected.pop("source_rows")
    selected["images"] = [
        representative_image(
            source_rows,
            source_effect=float(selected["source_effect"]),
            visible_seed=lidc_seed,
            regime="LIDC-hard",
        )
    ]
    selections["lidc_hard"] = selected

    candidates, selected = representative_source(
        paired,
        panel="riga",
        required_regimes=("RIGA-disc", "RIGA-cup"),
        required_seeds=(0, 1, 2),
    )
    candidate_tables.append(candidates)
    source_rows = selected.pop("source_rows")
    selected["images"] = [
        representative_image(
            source_rows,
            source_effect=float(selected["source_effect"]),
            visible_seed=0,
            regime=regime,
        )
        for regime in ("RIGA-disc", "RIGA-cup")
    ]
    selections["riga"] = selected

    candidates, selected = representative_source(
        paired,
        panel="bsds",
        required_regimes=("BSDS",),
        required_seeds=(0, 1, 2),
    )
    candidate_tables.append(candidates)
    source_rows = selected.pop("source_rows")
    selected["images"] = [
        representative_image(
            source_rows,
            source_effect=float(selected["source_effect"]),
            visible_seed=0,
            regime="BSDS",
        )
    ]
    selections["bsds"] = selected
    return pd.concat(candidate_tables, ignore_index=True), selections

PANEL_ORDER = ("lidc_hard", "riga", "bsds")

ROW_LABELS = {
    "LIDC-hard": "LIDC-hard",
    "RIGA-disc": "RIGA disc",
    "RIGA-cup": "RIGA cup",
    "BSDS": "BSDS",
}

def _rgb_on_grid(path: Path, height: int, width: int) -> np.ndarray:
    with Image.open(path) as image:
        resized = image.convert("RGB").resize(
            (int(width), int(height)), Image.Resampling.BILINEAR
        )
        return np.asarray(resized)

def _mask_overlay(image: np.ndarray, mask: np.ndarray, color: tuple[float, ...]) -> np.ndarray:
    output = image.astype(np.float32) / 255.0
    alpha = 0.48 * mask.astype(np.float32)[..., None]
    tint = np.asarray(color[:3], dtype=np.float32)[None, None, :]
    return output * (1.0 - alpha) + tint * alpha

def _show_base(axis: Any, image: np.ndarray) -> None:
    axis.imshow(image)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)

def _show_bank(
    cell: Any,
    image: np.ndarray,
    bank: torch.Tensor,
    *,
    color: tuple[float, ...],
) -> None:
    grid = cell.subgridspec(2, 4, wspace=0.02, hspace=0.12)
    for index, mask in enumerate(bank.numpy()):
        axis = plt.subplot(grid[index // 4, index % 4])
        _show_base(axis, _mask_overlay(image, mask, color))
        axis.text(
            0.03,
            0.95,
            str(index + 1),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=5,
            color="white",
            bbox={"facecolor": "black", "edgecolor": "none", "pad": 0.4, "alpha": 0.65},
        )

def render(rows: list[dict[str, Any]], pdf_path: Path, png_path: Path) -> None:
    if len(rows) != 4:
        raise ValueError("qualitative figure requires LIDC, two RIGA, and BSDS rows")
    figure = plt.figure(figsize=(7.0, 5.7), constrained_layout=False)
    outer = figure.add_gridspec(
        len(rows), 5, width_ratios=(1.0, 1.0, 1.0, 2.05, 2.05), wspace=0.04, hspace=0.12
    )
    headers = ("Image", "Visible", "Held-out", "WTA heads 1–8", "RN heads 1–8")
    visible_color = matplotlib.colormaps["tab10"](2)
    wta_color = matplotlib.colormaps["tab10"](1)
    rn_color = matplotlib.colormaps["tab10"](0)
    contour_colors = [matplotlib.colormaps["tab20"](index) for index in range(20)]
    for row_index, row in enumerate(rows):
        image = row["image"]
        image_axis = figure.add_subplot(outer[row_index, 0])
        _show_base(image_axis, image)
        image_axis.set_ylabel(
            f"{ROW_LABELS[row['regime']]}\nseed {row['visible_seed']}\n"
            f"Δ={row['source_effect']:+.3f}",
            fontsize=6,
        )
        visible_axis = figure.add_subplot(outer[row_index, 1])
        _show_base(visible_axis, _mask_overlay(image, row["visible"], visible_color))
        hidden_axis = figure.add_subplot(outer[row_index, 2])
        _show_base(hidden_axis, image)
        for index, mask in enumerate(row["hidden"]):
            if mask.any() and not mask.all():
                hidden_axis.contour(
                    mask.astype(float),
                    levels=[0.5],
                    colors=[contour_colors[index % len(contour_colors)]],
                    linewidths=0.55,
                )
        _show_bank(outer[row_index, 3], image, row["wta_bank"], color=wta_color)
        _show_bank(outer[row_index, 4], image, row["rn_bank"], color=rn_color)
        if row_index == 0:
            for column, title in enumerate(headers[:3]):
                figure.axes[column].set_title(title, fontsize=7, pad=2)
            # Bank columns do not have a parent axis, so place figure-level headers.
            figure.text(0.642, 0.988, headers[3], ha="center", va="top", fontsize=7)
            figure.text(0.882, 0.988, headers[4], ha="center", va="top", fontsize=7)
    figure.subplots_adjust(left=0.085, right=0.997, bottom=0.012, top=0.965)
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.015)
    figure.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.015)
    plt.close(figure)
