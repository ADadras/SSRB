"""Two-stage SSRB training with the accepted loss and selection rules."""
from __future__ import annotations
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import VisibleDataset, sha256, verify_inputs, write_json
from .execution import configure as configure_execution, record as execution_record
from .losses import objective
from .models import DirectMaskBank, FrozenDinoRawResidual

DEFAULTS = dict(heads=8, hidden=256, base=24, image_size=128, semantic_epochs=40, semantic_patience=8,
                semantic_batch=32, semantic_lr=.0003, weight_decay=.0001, eligible_after=4,
                residual_epochs=80, residual_patience=12, residual_batch=12, residual_lr=.001, workers=4)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configuration(path=None):
    config = dict(DEFAULTS)
    if path:
        overrides = json.loads(Path(path).read_text())
        if set(overrides) - set(config):
            raise ValueError("unknown configuration fields")
        config.update(overrides)
    return config


def load_model(checkpoint, device="cpu"):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if ck.get("schema") != "ssrb-release-v1":
        raise ValueError("unsupported checkpoint schema")
    anchor = DirectMaskBank(**ck["model_args"])
    if ck["stage"] == "semantic":
        anchor.load_state_dict(ck["state_dict"], strict=True)
        return anchor.to(device).eval(), ck
    model = FrozenDinoRawResidual(anchor, heads=ck["config"]["heads"], base=ck["config"]["base"])
    model.load_state_dict(ck["state_dict"], strict=True)
    return model.to(device).eval(), ck


def epoch_run(model, loader, device, epoch, optimizer=None, residual=False):
    training = optimizer is not None
    model.train(training)
    total, count = 0., 0
    with torch.enable_grad() if training else torch.no_grad():
        for batch_index, batch in enumerate(loader):
            features, target = batch["features"].to(device), batch["target"].to(device)
            masks, ranks = model(features, batch["image"].to(device)) if residual else model(features)
            loss, _ = objective(masks, ranks, target, training=training,
                                capacity_offset=(epoch-1)*len(loader)+batch_index)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            size = features.shape[0]
            total += float(loss.detach()) * size
            count += size
    return total / count


def train(stage, root, manifest, output, *, config=None, seed=0, device="cpu", anchor=None, deterministic=False, cache_inputs=False, _reserved_output=False):
    configure_execution(deterministic)
    start = time.perf_counter()
    root, manifest, output = Path(root), Path(manifest), Path(output)
    if output.exists() and not _reserved_output:
        raise FileExistsError(output)
    c = configuration(config)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available")
    rows, seal = verify_inputs(root, manifest)
    seed_all(seed)
    residual = stage == "residual"
    if stage not in {"semantic", "residual"}:
        raise ValueError(stage)
    payloads = {s: torch.load(root/f"{s}_features.pt", map_location="cpu", weights_only=True) for s in ("train", "val")}
    datasets = {s: VisibleDataset(root, rows, payloads[s], s, residual=residual, image_size=c["image_size"]) for s in payloads}
    cached_bytes = {s: data.preload_inputs() for s, data in datasets.items()} if cache_inputs else {}
    loader_args = dict(batch_size=c[f"{stage}_batch"], num_workers=c["workers"], pin_memory=device.startswith("cuda"))
    # Semantic runs use the global generator, with a seed set before model initialization.
    # Residual runs use the explicitly seeded train generator from that stage.
    extra = {"generator": torch.Generator().manual_seed(seed)} if residual else {}
    train_loader = DataLoader(datasets["train"], shuffle=True, **loader_args, **extra)
    val_loader = DataLoader(datasets["val"], shuffle=False, **loader_args)
    shape = payloads["train"]["features"].shape
    args = dict(layers=shape[1], dim=shape[3], queries=c["heads"], hidden=c["hidden"], dropout=0.)
    if residual:
        if anchor is None:
            raise ValueError("residual training requires a semantic checkpoint")
        bank, anchor_ck = load_model(anchor, device)
        if (anchor_ck["stage"] != "semantic" or anchor_ck["input_seal"] != seal
                or anchor_ck["optimizer_seed"] != seed):
            raise ValueError("anchor was trained on different visible inputs or optimizer seed")
        args = anchor_ck["model_args"]
        model = FrozenDinoRawResidual(bank, heads=c["heads"], base=c["base"]).to(device)
        # Preserve the original validation pre-forward and its RNG/BN behavior.
        with torch.no_grad():
            sample = next(iter(val_loader))
            base_masks, base_ranks = model.anchor(sample["features"].to(device))
            masks, ranks = model(sample["features"].to(device), sample["image"].to(device))
            expected = torch.nn.functional.interpolate(base_masks, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            if not torch.equal(masks, expected) or not torch.equal(ranks, base_ranks):
                raise AssertionError("zero residual changed the anchor")
        optimizer = torch.optim.Adam(model.raw.parameters(), lr=c["residual_lr"])
    else:
        model = DirectMaskBank(**args).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=c["semantic_lr"], weight_decay=c["weight_decay"])
    output.mkdir(parents=True, exist_ok=_reserved_output)
    history, best, best_epoch, best_state, bad = [], math.inf, None, None, 0
    for epoch in range(1, c[f"{stage}_epochs"]+1):
        if sha256(manifest) != seal["manifest_sha256"]:
            raise ValueError("visible choices changed during training")
        training = epoch_run(model, train_loader, device, epoch, optimizer, residual)
        validation = epoch_run(model, val_loader, device, epoch, residual=residual)
        eligible = residual or epoch > c["eligible_after"]
        row = dict(epoch=epoch, train_objective=training, validation_objective=validation, eligible=eligible)
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        improved = validation < best if residual else validation < best - 1e-12
        if eligible and improved:
            best, best_epoch, bad = validation, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif eligible:
            bad += 1
            if bad >= c[f"{stage}_patience"]:
                break
    if best_state is None:
        raise RuntimeError("no eligible checkpoint")
    # Recheck input hashes at the end, before publishing the checkpoint locally.
    verify_inputs(root, manifest)
    ck = dict(schema="ssrb-release-v1", stage=stage, state_dict=best_state, model_args=args, config=c,
              input_seal=seal, optimizer_seed=seed, visible_seed=rows[0]["visible_seed"], endpoint=rows[0]["endpoint"],
              best_epoch=best_epoch, best_objective=best)
    if residual:
        ck["anchor_sha256"] = sha256(anchor)
    torch.save(ck, output/"checkpoint.pt")
    write_json(output/"history.json", history)
    write_json(output/"run.json", {k: v for k, v in ck.items() if k not in {"state_dict"}} |
               dict(checkpoint_sha256=sha256(output/"checkpoint.pt"), seconds=time.perf_counter()-start,
                    torch_version=str(torch.__version__), device=str(device), execution=execution_record(),
                    cache_inputs=bool(cache_inputs), cached_input_bytes=cached_bytes))
    return ck, history
