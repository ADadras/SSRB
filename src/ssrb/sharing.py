"""The four support-sharing controls retained in the paper's negative results."""
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
from typing import Any
import argparse
import hashlib
import json
import math
import time
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from . import losses
from .comparisons import ComparisonDataset,configuration,input_snapshot
from .data import sha256,within,write_json
from .execution import configure,record
from .models import DirectMaskBank
from .training import seed_all

METHODS=('sa_dmb','capacity_sa_dmb','sa_dmb_shuffled','capacity_sa_dmb_shuffled')


def scheduled_temperature(
    *, start: float, end: float, epoch: int, epochs: int, schedule: str
) -> float:
    fraction = 0.0 if int(epochs) <= 1 else (int(epoch) - 1) / (int(epochs) - 1)
    if schedule == "linear":
        return float(start) + (float(end) - float(start)) * fraction
    if schedule == "geometric":
        if float(start) <= 0 or float(end) <= 0:
            raise ValueError("geometric temperatures must be positive")
        return float(start) * (float(end) / float(start)) ** fraction
    raise ValueError(f"unknown temperature schedule: {schedule}")

def normalized_entropy(frequencies: torch.Tensor) -> float:
    probability = frequencies / frequencies.sum().clamp_min(1e-12)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum()
    return float(entropy / math.log(max(2, int(probability.numel()))))

def _capacity_assignment(costs, *, method, temperature, capacity_offset):
    if method!='capacity_wta' or temperature!=1.:
        raise ValueError('the sharing adapter requires the fixed capacity anchor')
    return losses.assignment_weights(costs,method=method,capacity_offset=capacity_offset)


base=SimpleNamespace(pair_cost=losses.pair_cost,assignment_weights=_capacity_assignment,pair_diversity=losses.pair_diversity,scheduled_temperature=scheduled_temperature,normalized_entropy=normalized_entropy)

SUPPORT_METHODS = {
    "sa_dmb",
    "capacity_sa_dmb",
    "sa_dmb_shuffled",
    "capacity_sa_dmb_shuffled",
}

CAPACITY_METHODS = {
    "capacity_wta",
    "capacity_sa_dmb",
    "capacity_sa_dmb_shuffled",
}

SHUFFLED_METHODS = {"sa_dmb_shuffled", "capacity_sa_dmb_shuffled"}

SOFT_METHODS = {
    "dmb_rn",
    "sa_dmb",
    "capacity_sa_dmb",
    "sa_dmb_shuffled",
    "capacity_sa_dmb_shuffled",
}

def stable_ranks(costs: torch.Tensor, anchor: torch.Tensor | None = None) -> torch.Tensor:
    """Return stable ordinal ranks, optionally forcing a supplied anchor to rank zero."""
    batch, queries = costs.shape
    order = torch.argsort(costs.detach(), dim=1, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    if anchor is None:
        values = torch.arange(queries, device=costs.device, dtype=torch.float32)
        ranks.scatter_(1, order, values[None, :].expand(batch, -1))
        return ranks
    if tuple(anchor.shape) != (batch,):
        raise ValueError("anchor shape does not match costs")
    for row in range(batch):
        winner = int(anchor[row])
        remaining = order[row][order[row] != winner]
        ranks[row, winner] = 0.0
        ranks[row, remaining] = torch.arange(
            1, queries, device=costs.device, dtype=torch.float32
        )
    return ranks

def support_embeddings(mask_logits: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Stop-gradient probability-weighted DINO support embedding."""
    probabilities = mask_logits.sigmoid().detach()
    batch, queries, height, width = probabilities.shape
    feature_grid = features.detach().float().mean(dim=1)
    tokens = int(feature_grid.shape[1])
    if height * width != tokens:
        side = int(round(math.sqrt(tokens)))
        if side * side != tokens:
            raise ValueError(f"support expects a square feature grid, got {tokens} tokens")
        probabilities = F.interpolate(
            probabilities, size=(side, side), mode="bilinear", align_corners=False
        )
    masks = probabilities.flatten(2)
    areas = masks.mean(dim=2)
    normalized_features = F.normalize(feature_grid, dim=2)
    denominator = masks.sum(dim=2, keepdim=True).clamp_min(1e-6)
    support = torch.einsum("bkt,btc->bkc", masks, normalized_features) / denominator
    support = F.normalize(support, dim=2)
    if support.shape[:2] != (batch, queries):
        raise AssertionError("support shape does not match mask bank")
    return support.detach(), areas.detach()

def support_log_affinity(support: torch.Tensor) -> torch.Tensor:
    """Global-scale=1 paper-specified RBF affinity, returned in stable log form."""
    batch, queries, _ = support.shape
    if queries <= 1:
        return support.new_zeros((batch, queries, queries))
    distance2 = torch.cdist(support, support, p=2).pow(2)
    eye = torch.eye(queries, device=support.device, dtype=torch.bool)[None]
    scale = distance2.masked_fill(eye, 0.0).sum(dim=(1, 2))
    scale = (scale / float(queries * (queries - 1))).view(batch, 1, 1)
    scale = scale.clamp_min(1e-6)
    output = -distance2 / scale
    return output.masked_fill(eye, 0.0).detach()

def deterministic_deranged_rows(
    rows: torch.Tensor,
    anchors: torch.Tensor,
    image_ids: list[str],
    *,
    endpoint: str,
    visible_seed: int,
    salt: str,
) -> tuple[torch.Tensor, list[list[int]]]:
    """Derange non-anchor affinity values without touching RNG state or marginals."""
    batch, queries = rows.shape
    if queries < 3:
        raise ValueError("the locked support derangement requires K>=3")
    if len(image_ids) != batch or tuple(anchors.shape) != (batch,):
        raise ValueError("shuffle identifiers do not match affinity rows")
    shuffled = rows.clone()
    mappings: list[list[int]] = []
    for row_index, image_id in enumerate(image_ids):
        anchor = int(anchors[row_index])
        eligible = [head for head in range(queries) if head != anchor]
        eligible.sort(
            key=lambda head: hashlib.sha256(
                f"{salt}|{endpoint}|{visible_seed}|{image_id}|{head}".encode()
            ).hexdigest()
        )
        donors = eligible[1:] + eligible[:1]
        mapping = list(range(queries))
        for receiver, donor in zip(eligible, donors, strict=True):
            shuffled[row_index, receiver] = rows[row_index, donor]
            mapping[receiver] = donor
        mapping[anchor] = anchor
        mappings.append(mapping)
    return shuffled, mappings

def capacity_anchors(costs: torch.Tensor, *, capacity_offset: int) -> torch.Tensor:
    _, winners = base.assignment_weights(
        costs,
        method="capacity_wta",
        temperature=1.0,
        capacity_offset=int(capacity_offset),
    )
    return winners

def active_method(method: str, epoch: int, warmup_epochs: int) -> str:
    if int(epoch) <= int(warmup_epochs):
        if method in {"dmb_rn", "sa_dmb", "sa_dmb_shuffled"}:
            return "hard_wta"
        if method in {"capacity_sa_dmb", "capacity_sa_dmb_shuffled"}:
            return "capacity_wta"
    return str(method)

def factorial_assignment_weights(
    costs: torch.Tensor,
    mask_logits: torch.Tensor,
    features: torch.Tensor,
    image_ids: list[str],
    *,
    method: str,
    temperature: float,
    capacity_offset: int,
    endpoint: str,
    visible_seed: int,
    shuffle_salt: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    batch, queries = costs.shape
    raw_winners = costs.detach().argmin(dim=1)
    anchors = (
        capacity_anchors(costs, capacity_offset=int(capacity_offset))
        if method in CAPACITY_METHODS
        else raw_winners
    )
    if method in {"hard_wta", "capacity_wta"}:
        weights = F.one_hot(anchors, num_classes=queries).float()
        return weights, anchors, {
            "support_neff": 1.0,
            "assignment_anchor_weight": 1.0,
            "support_cross_separation": 0.0,
            "shuffle_applied": 0.0,
        }
    ranks = stable_ranks(costs, anchor=anchors if method in CAPACITY_METHODS else None)
    logits = -ranks / max(1e-6, float(temperature))
    diagnostics = {
        "assignment_anchor_weight": 1.0,
        "support_cross_separation": 0.0,
        "shuffle_applied": 0.0,
    }
    if method in SUPPORT_METHODS:
        support, _ = support_embeddings(mask_logits, features)
        log_affinity = support_log_affinity(support)
        batch_indices = torch.arange(batch, device=costs.device)
        anchor_rows = log_affinity[batch_indices, anchors]
        if method in SHUFFLED_METHODS:
            anchor_rows, _ = deterministic_deranged_rows(
                anchor_rows,
                anchors,
                image_ids,
                endpoint=str(endpoint),
                visible_seed=int(visible_seed),
                salt=str(shuffle_salt),
            )
            diagnostics["shuffle_applied"] = 1.0
        logits = logits + anchor_rows
        affinity = log_affinity.exp()
        off_diagonal = ~torch.eye(queries, device=costs.device, dtype=torch.bool)
        diagnostics["support_cross_separation"] = float(
            (1.0 - affinity[:, off_diagonal].mean(dim=1)).mean().cpu()
        )
    weights = torch.softmax(logits, dim=1)
    diagnostics["support_neff"] = float(
        (1.0 / weights.detach().pow(2).sum(dim=1).clamp_min(1e-12)).mean().cpu()
    )
    diagnostics["assignment_anchor_weight"] = float(
        weights.detach().gather(1, anchors[:, None]).mean().cpu()
    )
    return weights, anchors, diagnostics

def factorial_objective(
    mask_logits: torch.Tensor,
    rank_logits: torch.Tensor,
    target: torch.Tensor,
    features: torch.Tensor,
    image_ids: list[str],
    *,
    method: str,
    temperature: float,
    q_loss_weight: float,
    pair_diversity_weight: float,
    capacity_offset: int,
    endpoint: str,
    visible_seed: int,
    shuffle_salt: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    costs = base.pair_cost(mask_logits, target)
    weights, anchors, support_diagnostics = factorial_assignment_weights(
        costs,
        mask_logits,
        features,
        image_ids,
        method=str(method),
        temperature=float(temperature),
        capacity_offset=int(capacity_offset),
        endpoint=str(endpoint),
        visible_seed=int(visible_seed),
        shuffle_salt=str(shuffle_salt),
    )
    detached = weights.detach()
    mask_loss = (detached * costs).sum(dim=1).mean()
    rank_loss = F.binary_cross_entropy_with_logits(rank_logits, detached)
    diversity = base.pair_diversity(mask_logits)
    loss = (
        mask_loss
        + float(q_loss_weight) * rank_loss
        + float(pair_diversity_weight) * diversity
    )
    best = costs.detach().min(dim=1).values
    return loss, {
        "mask_loss": float(mask_loss.detach().cpu()),
        "rank_loss": float(rank_loss.detach().cpu()),
        "pair_diversity_penalty": float(diversity.detach().cpu()),
        "winners": anchors.detach().cpu(),
        "mean_pair_cost": float(costs.detach().mean().cpu()),
        "mean_assignment_cost": float(
            (detached * costs).sum(dim=1).mean().detach().cpu()
        ),
        "assignment_regret": float(
            ((detached * costs).sum(dim=1) - best).mean().detach().cpu()
        ),
        **support_diagnostics,
    }

def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    optimizer: torch.optim.Optimizer | None,
    validation: bool,
) -> tuple[dict[str, float], set[str]]:
    training = optimizer is not None
    model.train(training)
    method = "hard_wta" if validation else active_method(
        str(args.method), int(epoch), int(args.wta_warmup_epochs)
    )
    temperature = base.scheduled_temperature(
        start=float(args.temperature_start),
        end=float(args.temperature_end),
        epoch=int(epoch),
        epochs=int(args.epochs),
        schedule=str(args.temperature_schedule),
    )
    total = 0
    sums: Counter[str] = Counter()
    counts = torch.zeros(int(args.queries), dtype=torch.float64)
    accessed: set[str] = set()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            features = batch["features"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            image_ids = [str(value) for value in batch["image_id"]]
            mask_logits, rank_logits = model(features)
            loss, parts = factorial_objective(
                mask_logits,
                rank_logits,
                target,
                features,
                image_ids,
                method=method,
                temperature=temperature,
                q_loss_weight=float(args.q_loss_weight),
                pair_diversity_weight=float(args.pair_diversity_weight),
                capacity_offset=(int(epoch) - 1) * len(loader) + int(batch_index),
                endpoint=str(args.endpoint),
                visible_seed=int(args.visible_seed),
                shuffle_salt=str(args.shuffle_salt),
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            winners = parts.pop("winners")
            count = int(features.shape[0])
            counts += torch.bincount(winners, minlength=int(args.queries)).double()
            sums["objective"] += float(loss.detach().cpu()) * count
            for key, value in parts.items():
                sums[key] += float(value) * count
            total += count
            accessed.update(str(path) for path in batch["mask_path"])
    frequencies = counts / counts.sum().clamp_min(1.0)
    metrics = {key: value / max(1, total) for key, value in sums.items()}
    metrics.update(
        {
            "examples": float(total),
            "temperature": float(temperature),
            "winner_entropy": base.normalized_entropy(frequencies),
            "winner_min_frequency": float(frequencies.min()),
            "winner_active": float((frequencies > 0).sum()),
        }
    )
    for head, frequency in enumerate(frequencies):
        metrics[f"winner_frequency_head_{head}"] = float(frequency)
    return metrics, accessed

DEFAULTS=dict(queries=8,hidden=256,dropout=0.,epochs=40,patience=8,batch_size=32,
    workers=4,lr=.0003,weight_decay=.0001,q_loss_weight=.30,pair_diversity_weight=0.,
    temperature_start=1.,temperature_end=.1,temperature_schedule='linear',wta_warmup_epochs=4,
    shuffle_salt='wacv-r2-sa-capacity-20260814')

class SharingDataset(ComparisonDataset):
    def __getitem__(self,index):
        value=super().__getitem__(index)
        value['mask_path']=str(within(self.root,self.rows[index]['visible_mask_path']))
        return value


def train(method,root,manifest,output,*,seed=0,device='cpu',deterministic=False,_reserved_output=False):
    if method not in METHODS or seed not in (0,1,2):
        raise ValueError('unsupported sharing control or optimizer seed')
    configure(deterministic);root,output=Path(root),Path(output)
    if output.exists() and not _reserved_output:
        raise FileExistsError(output)
    rows,seal=input_snapshot(root,manifest,features=True)
    if rows[0]['endpoint'] not in ('lidc','bsds'):
        raise ValueError('the reported sharing experiment covers LIDC and BSDS')
    args=SimpleNamespace(**DEFAULTS,method=method,endpoint=rows[0]['endpoint'],visible_seed=rows[0]['visible_seed'])
    seed_all(seed);c=configuration('dino_wta')
    data={s:SharingDataset(root,rows,s,c,'dino_wta') for s in ('train','val')}
    loaders={s:DataLoader(data[s],batch_size=args.batch_size,shuffle=s=='train',num_workers=args.workers,pin_memory=device.startswith('cuda')) for s in data}
    shape=data['train'].payload['features'].shape
    model_args=dict(layers=int(shape[1]),dim=int(shape[3]),queries=args.queries,hidden=args.hidden,dropout=args.dropout)
    model=DirectMaskBank(**model_args).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    output.mkdir(parents=True,exist_ok=_reserved_output)
    history=[];best=math.inf;state=None;bad=0;started=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        training,_=run_epoch(model,loaders['train'],device=torch.device(device),args=args,epoch=epoch,optimizer=optimizer,validation=False)
        validation,_=run_epoch(model,loaders['val'],device=torch.device(device),args=args,epoch=epoch,optimizer=None,validation=True)
        eligible=epoch>args.wta_warmup_epochs
        row=dict(epoch=epoch,active_method=active_method(method,epoch,args.wta_warmup_epochs),checkpoint_eligible=eligible,train=training,validation=validation)
        history.append(row);print(json.dumps(row),flush=True)
        candidate=float(validation['objective'])
        if eligible and candidate<best-1e-12:
            best,best_epoch,bad=candidate,epoch,0
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        elif eligible:
            bad+=1
            if bad>=args.patience:break
    if state is None:
        raise RuntimeError('no eligible checkpoint')
    if input_snapshot(root,manifest,features=True)[1]!=seal:
        raise ValueError('training input bytes changed')
    ck=dict(schema='ssrb-comparison-v1',method='dino_'+method,config=c,sharing_config=DEFAULTS,
        model_args=model_args,state_dict=state,input_seal=seal,endpoint=args.endpoint,
        visible_seed=args.visible_seed,optimizer_seed=seed,best_epoch=best_epoch,best_objective=best)
    torch.save(ck,output/'checkpoint.pt');write_json(output/'history.json',history)
    write_json(output/'run.json',{k:v for k,v in ck.items() if k!='state_dict'}|dict(checkpoint_sha256=sha256(output/'checkpoint.pt'),seconds=time.perf_counter()-started,execution=record()))
    return ck,history
