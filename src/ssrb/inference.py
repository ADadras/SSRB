"""Save predictions from images and features before any reference masks are read."""
from pathlib import Path
import torch
from .data import IMAGE_KEYS, load_image, read_rows, sha256, within, write_json
from .training import load_model
from .execution import configure as configure_execution, record as execution_record


@torch.no_grad()
def infer(checkpoint, root, manifest, feature_file, output, device="cpu", batch_size=12, deterministic=False):
    configure_execution(deterministic)
    output, root = Path(output), Path(root)
    if output.exists():
        raise FileExistsError(output)
    rows = sorted(read_rows(manifest), key=lambda r: r["image_id"])
    if any(set(r) != IMAGE_KEYS or r["split"] != "test" for r in rows):
        raise ValueError("inference accepts image-only test manifests")
    payload = torch.load(feature_file, map_location="cpu", weights_only=True)
    if set(payload) != {"features", "image_ids", "grid_hw"}:
        raise ValueError("unexpected feature metadata")
    index = {str(i): j for j, i in enumerate(payload["image_ids"])}
    model, ck = load_model(checkpoint, device)
    if any(r["endpoint"] != ck["endpoint"] or r["visible_seed"] != ck["visible_seed"] for r in rows):
        raise ValueError("checkpoint does not match image manifest")
    if batch_size < 1:
        raise ValueError("inference batch size must be positive")
    result = {}
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start+batch_size]
        features = torch.stack([payload["features"][index[r["image_id"]]].float() for r in batch]).to(device)
        if ck["stage"] == "residual":
            image = torch.stack([load_image(within(root, r["image_path"]), ck["config"]["image_size"]) for r in batch]).to(device)
            masks, ranks = model(features, image)
        else:
            masks, ranks = model(features)
        probabilities = masks.sigmoid().cpu()
        scores = ranks.sigmoid().cpu()
        for j, r in enumerate(batch):
            order = torch.argsort(scores[j], descending=True, stable=True)
            result[r["image_id"]] = {"probability": probabilities[j, order], "scores": scores[j, order]}
    output.mkdir(parents=True)
    torch.save(result, output/"predictions.pt")
    write_json(output/"frozen.json", dict(predictions_sha256=sha256(output/"predictions.pt"),
               checkpoint_sha256=sha256(checkpoint), images_sha256=sha256(manifest), feature_sha256=sha256(feature_file), device=str(device), batch_size=batch_size, execution=execution_record()))
    return result
