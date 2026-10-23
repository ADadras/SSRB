"""Frozen DINOv3 ViT-B/16 and DINOv2 ViT-S/14 feature extraction."""
from pathlib import Path
import hashlib
import json
import shutil
import tempfile
import torch
import PIL
from PIL import Image
import torchvision
from torchvision import transforms
from .data import read_rows, within, sha256, write_json
from .execution import record as execution_record


def load_local_dinov3(repo, weights):
    """Load the supplied weights without consulting a user's checkpoint cache."""
    weights = Path(weights).resolve(strict=True)
    previous = torch.hub.get_dir()
    with tempfile.TemporaryDirectory(prefix='ssrb-weights-') as directory:
        cached = Path(directory)/'checkpoints'/weights.name
        cached.parent.mkdir()
        try:
            cached.symlink_to(weights)
        except OSError:
            shutil.copyfile(weights, cached)
        torch.hub.set_dir(directory)
        try:
            return torch.hub.load(str(Path(repo).resolve()), 'dinov3_vitb16', source='local',
                                  pretrained=True, weights=str(weights), check_hash=False)
        finally:
            torch.hub.set_dir(previous)


@torch.no_grad()
def extract(root, manifest, output, dino_repo, weights, device="cuda", batch_size=16, backbone="dinov3"):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows = sorted(read_rows(manifest), key=lambda r: r["image_id"])
    if backbone not in {"dinov3", "dinov2"} or batch_size < 1:
        raise ValueError("invalid backbone or batch size")
    feature_layers = [5, 8, 11] if backbone == "dinov3" else [3, 6, 11]
    shape = (3, 784, 768) if backbone == "dinov3" else (3, 1024, 384)
    if backbone == "dinov3":
        model = load_local_dinov3(dino_repo, weights)
    else:
        model = torch.hub.load(str(Path(dino_repo).resolve()), "dinov2_vits14", source="local", pretrained=False)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    model.to(device).eval().requires_grad_(False)
    transform = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor(),
                 transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    dtype = torch.float16 if backbone == "dinov3" else torch.float32
    output.mkdir(parents=True)
    splits = sorted({r["split"] for r in rows})
    for split in splits:
        # The accepted extractors batch each split separately. Combining splits
        # changes batch boundaries and can change CUDA floating-point results.
        selected = [r for r in rows if r["split"] == split]
        features, ids = [], []
        for start in range(0, len(selected), batch_size):
            items = selected[start:start+batch_size]
            images = []
            for r in items:
                with Image.open(within(root, r["image_path"])) as im:
                    images.append(transform(im.convert("RGB")))
            outs = model.get_intermediate_layers(torch.stack(images).to(device), n=feature_layers,
                                                reshape=False, return_class_token=True)
            layers = [(out[0] if isinstance(out, tuple) else out).detach().to(dtype).cpu() for out in outs]
            value = torch.stack(layers, dim=1)
            if tuple(value.shape[1:]) != shape:
                raise ValueError("unexpected backbone token dimensions")
            features.append(value)
            ids.extend(r["image_id"] for r in items)
        torch.save(dict(features=torch.cat(features), image_ids=ids, grid_hw=(28, 28) if backbone == "dinov3" else (32, 32)), output/f"{split}_features.pt")
    write_json(output/"feature_provenance.json", dict(weights_sha256=sha256(weights), image_manifest_sha256=sha256(manifest),
               backbone=backbone, layers=feature_layers, image_size=448, dtype=str(dtype).removeprefix("torch."),
               batch_size=batch_size, batches_separated_by_split=True, device=str(device), execution=execution_record(),
               torchvision=str(torchvision.__version__), pillow=PIL.__version__,
               preprocessing="torchvision Resize/ToTensor/ImageNet normalization"))


def fingerprint(source):
    """Hash ordered portable feature tensors without exporting their values."""
    payload = torch.load(source, map_location='cpu', weights_only=True)
    if set(payload) != {'features', 'image_ids', 'grid_hw'}:
        raise ValueError('requires a portable cache with no extra metadata')
    value = payload['features'].contiguous()
    if value.ndim != 4 or value.shape[0] != len(payload['image_ids']):
        raise ValueError('feature tensor and image identities do not agree')
    return dict(shape=list(value.shape), dtype=str(value.dtype).removeprefix('torch.'),
                grid_hw=list(payload['grid_hw']),
                image_ids_sha256=hashlib.sha256(json.dumps(payload['image_ids'],separators=(',',':')).encode()).hexdigest(),
                tensor_sha256=hashlib.sha256(memoryview(value.view(torch.uint8).numpy()).cast('B')).hexdigest())


def verify(source, reference, endpoint, split, backbone):
    entries = json.loads(Path(reference).read_text())['entries']
    choices = [r for r in entries if (r['endpoint'],r['split'],r['backbone']) == (endpoint,split,backbone)]
    if len(choices) != 1:
        raise ValueError('requires one verified reference for the task, split and backbone')
    actual = fingerprint(source)
    different = [k for k,v in actual.items() if choices[0][k] != v]
    if different:
        raise ValueError('feature cache differs from the verified reference: '+', '.join(different))
    return dict(status='verified-reference-feature-cache',endpoint=endpoint,split=split,backbone=backbone,**actual)


def import_cache(source, output, manifest=None, split=None):
    """Import trusted cached tensors, optionally selecting one endpoint and split."""
    from .data import IMAGE_KEYS, validate_visible
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    value = torch.load(source, map_location='cpu', weights_only=True)
    payload = {k:value[k] for k in ('features','image_ids','grid_hw')}
    index = {str(k):j for j,k in enumerate(payload['image_ids'])}
    if len(index)!=len(payload['image_ids']) or len(index)!=len(payload['features']):
        raise ValueError('feature identities are duplicated or incomplete')
    if manifest is not None:
        rows = read_rows(manifest)
        if not all(set(r)==IMAGE_KEYS for r in rows):
            validate_visible(rows)
        split = split or source.name.removesuffix('_features.pt')
        if split not in {'train','val','test'}:
            raise ValueError('specify train, val or test when filtering a cache')
        selected = sorted(str(r['image_id']) for r in rows if r['split']==split)
        if not selected or len(set(selected))!=len(selected) or not set(selected).issubset(index):
            raise ValueError('manifest feature identities are missing, duplicated or empty')
        payload['features'] = payload['features'][[index[k] for k in selected]]
        payload['image_ids'] = selected
    elif split is not None:
        raise ValueError('split filtering requires a manifest')
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return str(output)
