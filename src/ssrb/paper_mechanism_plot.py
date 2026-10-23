"""Render the paper's residual panels from external images and portable weights."""
from pathlib import Path
from decimal import Decimal,ROUND_HALF_UP
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn import functional as F
from .data import IMAGE_KEYS,load_image,read_rows,sha256,within,write_json
from .training import load_model


def display_image(path: str) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    lo, hi = np.quantile(array, [0.01, 0.99])
    return np.clip((array - lo) / max(float(hi - lo), 1e-6), 0, 1)

def residual_summary(path: Path, endpoint: str) -> dict[str, float]:
    """Read the three figure annotations from the full diagnostic summary."""
    frame = pd.read_csv(path)
    metrics = {
        "dice": "anchor_fused_dice",
        "displacement": "symmetric_boundary_displacement_px",
        "boundary_ratio": "boundary_to_nonboundary_magnitude_ratio",
    }
    result: dict[str, float] = {}
    for label, metric in metrics.items():
        rows = frame[(frame["endpoint"] == endpoint) & (frame["metric"] == metric)]
        if len(rows) != 1:
            raise ValueError(f"expected one {endpoint}/{metric} diagnostic row, found {len(rows)}")
        result[label] = float(rows.iloc[0]["mean"])
    return result

def manuscript_two_decimals(value: float) -> str:
    """Match the plan's two-decimal rendering of the three-decimal summary."""
    rounded_summary = Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    return f"{rounded_summary.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):.2f}"

def main(source,output):
    source,output=Path(source),Path(output)
    payload=json.loads(source.read_text())
    def resolve(name):
        path=Path(name)
        return path if path.is_absolute() else source.resolve().parent/path
    cases={r['endpoint']:r for r in payload['cases']}
    if len(payload['cases'])!=2 or set(cases)!={'lidc','bsds'}:
        raise ValueError('requires one LIDC and one BSDS case')
    diagnostic_summary=resolve(payload['summary'])
    output.parent.mkdir(parents=True,exist_ok=True)
    device=torch.device('cpu')
    figure, axes = plt.subplots(2, 5, figsize=(7.6, 3.2))
    figure.subplots_adjust(left=.055, right=.805, top=.865, bottom=.055, wspace=.055, hspace=.09)
    summaries: dict[str, dict[str, float]] = {}
    displayed_examples: dict[str, dict[str, object]] = {}
    for row_index, endpoint in enumerate(("lidc", "bsds")):
        summaries[endpoint] = residual_summary(diagnostic_summary, endpoint)
        case=cases[endpoint]
        image_id=str(case['image_id']);head=int(case['head_id'])
        if not 0<=head<8:
            raise ValueError('head index must be between zero and seven')
        image_root=resolve(case['root']);manifest=resolve(case['manifest'])
        rows=read_rows(manifest)
        if any(set(r)!=IMAGE_KEYS or r['split']!='test' for r in rows):
            raise ValueError('mechanism panels require image-only test rows')
        identities=[r for r in rows if r['image_id']==image_id]
        if len(identities)!=1 or identities[0]['endpoint']!=endpoint:
            raise ValueError('case does not identify one matching test image')
        image_path=str(within(image_root,identities[0]['image_path']))
        checkpoint=resolve(case['checkpoint']);model,ck=load_model(checkpoint,device)
        if ck['stage']!='residual' or ck['endpoint']!=endpoint or ck['visible_seed']!=identities[0]['visible_seed']:
            raise ValueError('case and SSRB checkpoint differ')
        features=torch.load(resolve(case['feature_file']),map_location='cpu',weights_only=True)
        if set(features)!={'features','image_ids','grid_hw'}:
            raise ValueError('unexpected feature metadata')
        feature_index={str(value):index for index,value in enumerate(features['image_ids'])}
        feature=features['features'][feature_index[image_id]].float()[None].to(device)
        override=case.get('selection_basis')=='visual_clarity'
        image_tensor = load_image(image_path, 128)[None].to(device)
        with torch.no_grad():
            anchor_logits, _ = model.anchor(feature)
            anchor_logits = F.interpolate(anchor_logits, size=(128, 128), mode="bilinear", align_corners=False)
            residual_logits = model.raw(image_tensor)["mask_logits"]
            fused_logits = anchor_logits + residual_logits
        image = display_image(image_path)
        display_hw = tuple(int(value) for value in image.shape[:2])

        # Map every model field to the displayed source-image grid before overlaying contours.
        anchor_field = F.interpolate(
            anchor_logits[:, head:head + 1], size=display_hw,
            mode="bilinear", align_corners=False,
        )[0, 0]
        residual_field = F.interpolate(
            residual_logits[:, head:head + 1], size=display_hw,
            mode="bilinear", align_corners=False,
        )[0, 0]
        fused_field = F.interpolate(
            fused_logits[:, head:head + 1], size=display_hw,
            mode="bilinear", align_corners=False,
        )[0, 0]
        anchor = anchor_field.sigmoid().cpu().numpy()
        residual = residual_field.cpu().numpy()
        fused = fused_field.sigmoid().cpu().numpy()
        anchor_mask = anchor >= .5
        fused_mask = fused >= .5
        denominator = int(anchor_mask.sum() + fused_mask.sum())
        example_dice = (
            2.0 * float(np.logical_and(anchor_mask, fused_mask).sum()) / denominator
            if denominator else 1.0
        )
        displayed_examples[endpoint] = {
            "image_id": image_id,
            "head_id": head,
            "anchor_fused_dice": example_dice,
            "selection_basis": (
                "visual clarity from source image and model outputs only"
                if override else
                "fixed-seed example nearest endpoint-median anchor-fused Dice"
            ),
        }
        panels = (image, anchor, residual, fused, image)
        titles = ("Image", "Semantic anchor", "Raw residual", "SSRB", "Anchor/SSRB contours")
        limit = float(max(abs(np.quantile(residual, .01)), abs(np.quantile(residual, .99)), 1e-6))
        for column, (axis, panel, title) in enumerate(zip(axes[row_index], panels, titles, strict=True)):
            if column in (0, 4):
                axis.imshow(panel)
            elif column == 2:
                axis.imshow(panel, cmap="coolwarm", vmin=-limit, vmax=limit)
            else:
                axis.imshow(panel, cmap="viridis", vmin=0, vmax=1)
            if column == 4:
                # Okabe--Ito blue/orange, with line style as a second cue.
                axis.contour(anchor_mask, levels=[.5], colors=["#E69F00"], linewidths=1.25,
                             linestyles="dashed")
                axis.contour(fused_mask, levels=[.5], colors=["#0072B2"], linewidths=1.35)
            axis.set_title(title if row_index == 0 else "", fontsize=7.2, pad=3)
            axis.set_xticks([])
            axis.set_yticks([])
        axes[row_index, 0].set_ylabel("LIDC" if endpoint == "lidc" else "BSDS",
                                      fontsize=8, fontweight="bold", labelpad=4)

        summary = summaries[endpoint]
        metric_text = (
            rf"$\bf{{Anchor\!\! -\!\! SSRB\ Dice}}$" "\n"
            f"{summary['dice']:.3f}\n\n"
            rf"$\bf{{Boundary\ displacement}}$" "\n"
            f"{manuscript_two_decimals(summary['displacement'])} px\n\n"
            rf"$\bf{{Boundary\ ratio}}$" "\n"
            f"{summary['boundary_ratio']:.2f}"
        )
        row_y = .635 if row_index == 0 else .195
        figure.text(.825, row_y, metric_text, ha="left", va="center", fontsize=7.0,
                    linespacing=1.02,
                    bbox={"boxstyle": "round,pad=.35", "facecolor": "#F7F7F7",
                          "edgecolor": "#B8B8B8", "linewidth": .6})

    figure.legend(
        handles=(
            Line2D([0], [0], color="#E69F00", lw=1.25, ls="--", label="Anchor"),
            Line2D([0], [0], color="#0072B2", lw=1.35, label="SSRB"),
        ),
        loc="upper right", bbox_to_anchor=(.995, .985), frameon=False, fontsize=6.8,
        handlelength=1.6, handletextpad=.35, borderaxespad=0,
    )
    figure.savefig(output, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)

    write_json(output.with_suffix('.record.json'),dict(profile='mechanism',
        reference_masks_read=False,cases=displayed_examples,side_metrics=summaries,
        source_manifest_sha256=sha256(source),summary_sha256=sha256(diagnostic_summary),
        output_pdf_sha256=sha256(output),output_png_sha256=sha256(output.with_suffix('.png'))))
