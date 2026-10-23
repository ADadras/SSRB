"""Small synthetic fixture for the entire pipeline; no external data required."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .data import freeze_manifests, read_rows, write_json, seal_inputs
from .training import train
from .inference import infer


def prepare(root):
    root = Path(root)
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    rows = []
    for i in range(16):
        split = "train" if i < 8 else "val" if i < 12 else "test"
        yy, xx = np.mgrid[:32, :32]
        mask = (xx-14-i%3)**2 + (yy-16)**2 < 60
        image = np.stack([mask*150+20, (xx*5+i)%256, (yy*6)%256], axis=-1).astype(np.uint8)
        Image.fromarray(image).save(root/f"image_{i}.png")
        masks = []
        for j in range(2):
            name = f"mask_{i}_{j}.png"
            value = mask if j == 0 else np.roll(mask, 3, axis=1)
            Image.fromarray(value.astype(np.uint8)*255).save(root/name)
            masks.append(name)
        rows.append(dict(image_id=f"synthetic_{i:02d}", source_id=f"source_{i:02d}", image_path=f"image_{i}.png",
                         mask_paths=masks, split=split, endpoint="synthetic"))
    write_json(root/"all_annotations.json", rows)
    return root/"all_annotations.json"


def prepare_demo(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = prepare(output/"source")
    freeze_manifests(source, source.parent, output/"prepared", 0)
    generator = torch.Generator().manual_seed(41)
    for part, manifest in (("trainer", "visible.json"), ("inference", "images.json")):
        root = output/"prepared"/part
        rows = sorted(read_rows(root/manifest), key=lambda r: r["image_id"])
        for split in sorted({r["split"] for r in rows}):
            subset = [r for r in rows if r["split"] == split]
            torch.save(dict(features=torch.randn(len(subset),3,4,8,generator=generator),
                            image_ids=[r["image_id"] for r in subset], grid_hw=(2,2)), root/f"{split}_features.pt")
    root = output/"prepared/trainer"
    seal_inputs(root, root/"visible.json")
    write_json(output/"tiny.json", dict(hidden=16,base=2,image_size=32,semantic_epochs=6,residual_epochs=2,
               semantic_patience=1,residual_patience=1,semantic_batch=8,residual_batch=4,workers=0))
    return output


def smoke(output, device="cpu"):
    output = prepare_demo(output)
    root = output/"prepared/trainer"
    train("semantic",root,root/"visible.json",output/"semantic",config=output/"tiny.json",device=device)
    train("residual",root,root/"visible.json",output/"residual",config=output/"tiny.json",device=device,anchor=output/"semantic/checkpoint.pt")
    root = output/"prepared/inference"
    infer(output/"residual/checkpoint.pt",root,root/"images.json",root/"test_features.pt",output/"predictions",device)
    # Evaluation is invoked only after predictions are saved.
    from .evaluation import evaluate
    return evaluate(output/"predictions", output/"prepared/evaluator", output/"prepared/evaluator/hidden.json", output/"evaluation")
