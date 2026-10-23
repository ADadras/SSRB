"""Train and infer the paper's probabilistic, raw-bank and initialization controls."""
from __future__ import annotations
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from .data import (IMAGE_KEYS, load_image, load_visible_mask, read_rows, sha256,
                   validate_visible, within, write_json)
from .execution import configure as configure_execution, record as execution_record
from .losses import assignment_weights, pair_cost, pair_diversity
from .models import DirectMaskBank, FrozenDinoRawResidual, KHeadValidityUNet
from .probabilistic import ProbUNet, dice_loss, kl_normal
from .training import load_model as load_core, seed_all

METHODS = ('raw_capacity', 'wide_raw', 'conventional_residual', 'probunet', 'cimd',
           'dino_wta', 'dino_soft_wta', 'dino_rank_neighborhood', 'dino_single', 'dino_mc_dropout')

class ConventionalResidual(FrozenDinoRawResidual):
    def zero_initialize_residual(self):
        # Retain exactly the ordinary output-layer initialization from PyTorch.
        pass


def configuration(method, path=None):
    if method not in METHODS:
        raise ValueError('unsupported comparison method')
    c = dict(heads=8, base=29 if method == 'wide_raw' else 24, hidden=256,
             image_size=128, epochs=80, patience=12, batch_size=12, lr=.001,
             weight_decay=0., workers=4, latent_dim=8, beta=.01,
             dropout=0., temperature_start=1., temperature_end=.1,
             warmup_epochs=4, diffusion_steps=1000, validation_noise_seed=3480,
             num_channels=128, num_res_blocks=2, num_heads=1,
             attention_resolutions='16', cimd_latent_dim=6)
    if method.startswith('dino_'):
        c.update(epochs=40, patience=8, batch_size=32, lr=.0003, weight_decay=.0001)
    if method in {'dino_single', 'dino_mc_dropout'}:
        c.update(heads=1, dropout=.2 if method=='dino_mc_dropout' else 0.)
    if method == 'cimd':
        c.update(batch_size=20, lr=.0001)
    if path:
        update = json.loads(Path(path).read_text())
        if set(update) - set(c):
            raise ValueError('unknown comparison configuration fields')
        c.update(update)
    if any(c[k] < 1 for k in ('heads','base','epochs','patience','batch_size','image_size')):
        raise ValueError('network/training sizes must be positive')
    return c


def fixed_validation_noise(image_id, *, image_size, diffusion_steps, seed=3480):
    digest = hashlib.sha256(f'{image_id}|{seed}|CIMD-visible-val-v1'.encode()).digest()
    generator = np.random.default_rng(int.from_bytes(digest[:8], 'little', signed=False))
    timestep = int(generator.integers(0, diffusion_steps))
    return timestep, torch.from_numpy(generator.standard_normal((1,image_size,image_size)).astype(np.float32))


def luminance(path, size):
    with Image.open(path) as im:
        array = np.asarray(im.convert('L').resize((size,size),Image.Resampling.BILINEAR), dtype=np.float32)/255.
    return torch.from_numpy(array)[None].repeat(4,1,1)


def input_snapshot(root, manifest, *, features=False):
    rows = read_rows(manifest)
    validate_visible(rows)
    names = {r[k] for r in rows for k in ('image_path','visible_mask_path')}
    if features:
        names |= {'train_features.pt','val_features.pt'}
    return rows, {'manifest_sha256':sha256(manifest),
                  'files':{n:sha256(within(root,n)) for n in sorted(names)}}


class ComparisonDataset(Dataset):
    def __init__(self, root, rows, split, c, method):
        self.root, self.c, self.method = Path(root), c, method
        self.rows = sorted([r for r in rows if r['split']==split],key=lambda r:r['image_id'])
        if not self.rows:
            raise ValueError('empty training/validation split')
        self.payload = None
        if method.startswith('dino_') or method=='conventional_residual':
            self.payload=torch.load(self.root/f'{split}_features.pt',map_location='cpu',weights_only=True)
            if set(self.payload)!={'features','image_ids','grid_hw'}:
                raise ValueError('unexpected feature columns')
            self.index={str(k):i for i,k in enumerate(self.payload['image_ids'])}
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        row=self.rows[index]; c=self.c
        size=tuple(self.payload['grid_hw']) if self.method.startswith('dino_') else (c['image_size'],)*2
        result={'image_id':row['image_id'],
                'target':load_visible_mask(within(self.root,row['visible_mask_path']),size)}
        if self.method=='cimd':
            result['image']=luminance(within(self.root,row['image_path']),c['image_size'])
            if row['split']=='val':
                result['fixed_timestep'],result['fixed_noise']=fixed_validation_noise(row['image_id'],image_size=c['image_size'],diffusion_steps=c['diffusion_steps'],seed=c['validation_noise_seed'])
        elif not self.method.startswith('dino_'):
            result['image']=load_image(within(self.root,row['image_path']),c['image_size'])
        if self.payload is not None:
            result['features']=self.payload['features'][self.index[row['image_id']]].float()
        return result


def bank_objective(masks, ranks, target, method, c, epoch, offset, training):
    cost=pair_cost(masks,target)
    if method=='dino_soft_wta':
        weights=torch.softmax(-cost/.05,dim=1)
    elif method=='dino_rank_neighborhood' and epoch>c['warmup_epochs']:
        order=torch.argsort(cost.detach(),dim=1,stable=True)
        rank=torch.empty_like(order,dtype=torch.float32)
        rank.scatter_(1,order,torch.arange(cost.shape[1],device=cost.device,dtype=torch.float32)[None].expand_as(rank))
        temperature=c['temperature_start']+(c['temperature_end']-c['temperature_start'])*(epoch-1)/max(1,c['epochs']-1)
        weights=torch.exp(-rank/max(1e-6,temperature));weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-12)
    else:
        assignment='capacity_wta' if training and method in {'raw_capacity','wide_raw','conventional_residual'} else 'hard_mcl'
        weights,_=assignment_weights(cost,method=assignment,capacity_offset=offset)
    loss=(weights.detach()*cost).sum(1).mean()+.30*F.binary_cross_entropy_with_logits(ranks,weights.detach())
    if method in {'raw_capacity','wide_raw'}:
        return loss
    return loss+0.*pair_diversity(masks)


def make_model(method, c, *, payload=None, anchor=None, cimd_root=None):
    if method=='probunet':
        args=dict(base=c['base'],latent_dim=c['latent_dim']);return ProbUNet(**args),args
    if method=='cimd':
        from .cimd import CIMDComponents
        if cimd_root is None:
            raise ValueError('CIMD requires a separate, pinned upstream checkout')
        args=dict(image_size=c['image_size'],num_channels=c['num_channels'],num_res_blocks=c['num_res_blocks'],num_heads=c['num_heads'],attention_resolutions=c['attention_resolutions'],latent_dim=c['cimd_latent_dim'])
        return CIMDComponents(**args,cimd_root=Path(cimd_root)),args
    if method=='conventional_residual':
        bank,ck=load_core(anchor)
        if ck['stage']!='semantic':raise ValueError('requires a semantic bank')
        return ConventionalResidual(bank,heads=c['heads'],base=c['base']),ck['model_args']
    if method.startswith('dino_'):
        shape=payload['features'].shape
        args=dict(layers=shape[1],dim=shape[-1],queries=c['heads'],hidden=c['hidden'],dropout=c['dropout'])
        return DirectMaskBank(**args),args
    args=dict(heads=c['heads'],base=c['base']);return KHeadValidityUNet(**args),args


def epoch_run(model,loader,device,method,c,epoch,optimizer=None,diffusion=None):
    training=optimizer is not None;model.train(training);total=0.;count=0
    with torch.enable_grad() if training else torch.no_grad():
        for i,batch in enumerate(loader):
            target=batch['target'].to(device)
            if method=='probunet':
                image=batch['image'].to(device);pm,pl,z,skips=model.encode_prior(image)
                qm,ql=model.encode_posterior(image,target[:,None]);latent=model.sample(qm,ql) if training else qm
                logits=model.decode(z,skips,latent)
                loss=F.binary_cross_entropy_with_logits(logits,target[:,None])+dice_loss(logits,target[:,None])+c['beta']*kl_normal(qm,ql,pm,pl)
            elif method=='cimd':
                from .cimd import training_objective
                image=batch['image'].to(device)
                timestep=torch.randint(0,c['diffusion_steps'],(image.shape[0],),device=device) if training else batch['fixed_timestep'].to(device)
                noise=None if training else batch['fixed_noise'].to(device)
                loss,_=training_objective(model,diffusion,image,target[:,None],timestep,noise=noise)
            else:
                if method.startswith('dino_'):masks,ranks=model(batch['features'].to(device))
                elif method=='conventional_residual':masks,ranks=model(batch['features'].to(device),batch['image'].to(device))
                else:
                    result=model(batch['image'].to(device));masks,ranks=result['mask_logits'],result['validity_logits']
                loss=bank_objective(masks,ranks,target,method,c,epoch,(epoch-1)*len(loader)+i,training)
            if not torch.isfinite(loss):raise ValueError('nonfinite objective')
            if training:
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            n=target.shape[0];total+=float(loss.detach())*n;count+=n
    return total/count


def train(method,root,manifest,output,*,config=None,anchor=None,cimd_root=None,seed=0,device='cpu',deterministic=False,_reserved_output=False):
    configure_execution(deterministic)
    root,output=Path(root),Path(output)
    if output.exists() and not _reserved_output:raise FileExistsError(output)
    c=configuration(method,config);needs=method.startswith('dino_') or method=='conventional_residual'
    rows,seal=input_snapshot(root,manifest,features=needs);seed_all(seed)
    data={s:ComparisonDataset(root,rows,s,c,method) for s in ('train','val')}
    if method=='conventional_residual':
        ack=torch.load(anchor,map_location='cpu',weights_only=True)
        if ack['endpoint']!=rows[0]['endpoint'] or ack['visible_seed']!=rows[0]['visible_seed'] or ack['optimizer_seed']!=seed or ack['input_seal']['manifest_sha256']!=seal['manifest_sha256']:
            raise ValueError('anchor identity does not match inputs')
        # The core anchor seal includes feature hashes; check all shared files.
        for name,value in seal['files'].items():
            if ack['input_seal'].get(name)!=value:raise ValueError('anchor input bytes differ')
    # The original raw-capacity trainer uses the global RNG. Only the two novelty controls use a separate generator.
    extra={'generator':torch.Generator().manual_seed(seed)} if method in {'wide_raw','conventional_residual'} else {}
    loaders={s:DataLoader(data[s],batch_size=c['batch_size'],shuffle=s=='train',num_workers=c['workers'],pin_memory=device.startswith('cuda'),**(extra if s=='train' else {})) for s in data}
    model,args=make_model(method,c,payload=data['train'].payload,anchor=anchor,cimd_root=cimd_root);model=model.to(device)
    if method=='conventional_residual':parameters=model.raw.parameters()
    else:parameters=model.parameters()
    opt=(torch.optim.AdamW if method.startswith('dino_') or method=='cimd' else torch.optim.Adam)(parameters,lr=c['lr'],**({'weight_decay':c['weight_decay']} if method.startswith('dino_') or method=='cimd' else {}))
    diffusion=None
    if method=='cimd':
        from .cimd import create_diffusion
        diffusion=create_diffusion(steps=c['diffusion_steps'],cimd_root=Path(cimd_root))
    output.mkdir(parents=True,exist_ok=_reserved_output);history=[];best=math.inf;state=None;bad=0;started=time.perf_counter()
    for epoch in range(1,c['epochs']+1):
        if sha256(manifest)!=seal['manifest_sha256']:raise ValueError('visible choices changed')
        tr=epoch_run(model,loaders['train'],device,method,c,epoch,opt,diffusion)
        val=epoch_run(model,loaders['val'],device,method,c,epoch,diffusion=diffusion)
        eligible=method!='dino_rank_neighborhood' or epoch>c['warmup_epochs']
        history.append(dict(epoch=epoch,train_objective=tr,validation_objective=val,eligible=eligible));print(json.dumps(history[-1]),flush=True)
        improved = val < best-1e-12 if method.startswith('dino_') else val < best
        if eligible and improved:
            best,best_epoch,bad=val,epoch,0;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        elif eligible:
            bad+=1
            if bad>=c['patience']:break
    if state is None:raise RuntimeError('no eligible checkpoint')
    if input_snapshot(root,manifest,features=needs)[1]!=seal:raise ValueError('training input bytes changed')
    ck=dict(schema='ssrb-comparison-v1',method=method,config=c,model_args=args,state_dict=state,input_seal=seal,endpoint=rows[0]['endpoint'],visible_seed=rows[0]['visible_seed'],optimizer_seed=seed,best_epoch=best_epoch,best_objective=best)
    if method=='conventional_residual':ck['anchor_sha256']=sha256(anchor)
    torch.save(ck,output/'checkpoint.pt');write_json(output/'history.json',history)
    write_json(output/'run.json',{k:v for k,v in ck.items() if k!='state_dict'}|dict(checkpoint_sha256=sha256(output/'checkpoint.pt'),seconds=time.perf_counter()-started,device=device,torch_version=str(torch.__version__),execution=execution_record()))
    return ck,history


def load(checkpoint,device='cpu',cimd_root=None):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if ck.get('schema')!='ssrb-comparison-v1':raise ValueError('not a comparison checkpoint')
    method,c,args=ck['method'],ck['config'],ck['model_args']
    if method=='conventional_residual':model=ConventionalResidual(DirectMaskBank(**args),heads=c['heads'],base=c['base'])
    elif method=='probunet':model=ProbUNet(**args)
    elif method=='cimd':
        from .cimd import CIMDComponents
        if not cimd_root:raise ValueError('CIMD checkout is required')
        model=CIMDComponents(**args,cimd_root=Path(cimd_root))
    elif method.startswith('dino_'):model=DirectMaskBank(**args)
    else:model=KHeadValidityUNet(**args)
    model.load_state_dict(ck['state_dict'],strict=True);return model.to(device).eval(),ck


@torch.no_grad()
def infer(checkpoint,root,manifest,output,*,feature_file=None,cimd_root=None,device='cpu',batch_size=None,sampling_seed=None,deterministic=False):
    configure_execution(deterministic)
    root,output=Path(root),Path(output)
    if output.exists():raise FileExistsError(output)
    rows=sorted(read_rows(manifest),key=lambda r:r['image_id'])
    if any(set(r)!=IMAGE_KEYS or r['split']!='test' for r in rows):raise ValueError('inference requires image-only test rows')
    info=torch.load(checkpoint,map_location='cpu',weights_only=True)
    method,c=info['method'],info['config']
    sampling_seed=(3490 if method=='cimd' else 3470) if sampling_seed is None else sampling_seed
    batch_size=(1 if method=='cimd' else 32 if method.startswith('dino_') else 12) if batch_size is None else batch_size
    seed_all(sampling_seed)
    model,ck=load(checkpoint,device,cimd_root)
    if batch_size<1:raise ValueError('batch size must be positive')
    if any(r['endpoint']!=ck['endpoint'] or r['visible_seed']!=ck['visible_seed'] for r in rows):raise ValueError('checkpoint/manifest identity differs')
    payload=None
    if method.startswith('dino_') or method=='conventional_residual':
        payload=torch.load(feature_file,map_location='cpu',weights_only=True)
        if set(payload)!={'features','image_ids','grid_hw'}:
            raise ValueError('unexpected feature columns')
        index={str(k):i for i,k in enumerate(payload['image_ids'])}
        if len(index)!=len(payload['image_ids']) or set(index)!={r['image_id'] for r in rows}:
            raise ValueError('test feature identities differ')
    diffusion=None
    if method=='cimd':
        from .cimd import create_diffusion,sample_mask
        diffusion=create_diffusion(steps=c['diffusion_steps'],cimd_root=Path(cimd_root))
    result={}
    # DataLoader consumes the same parent RNG seed as the accepted stochastic evaluator.
    for indices in DataLoader(range(len(rows)),batch_size=batch_size,shuffle=False,num_workers=0):
        batch=[rows[int(i)] for i in indices]
        if not method.startswith('dino_'):
            image=torch.stack([(luminance if method=='cimd' else load_image)(within(root,r['image_path']),c['image_size']) for r in batch]).to(device)
        if method=='probunet':
            pm,pl,z,skips=model.encode_prior(image)
            probability=torch.cat([model.decode(z,skips,model.sample(pm,pl)).sigmoid() for _ in range(8)],1)
            scores=torch.ones(probability.shape[:2],device=device)
        elif method=='cimd':
            expanded=image[:,None].expand(-1,8,-1,-1,-1).reshape(-1,*image.shape[1:]).contiguous()
            signed=sample_mask(model.model,diffusion,expanded)
            probability=((signed+1.)/2.).clamp(0,1).reshape(len(batch),8,*signed.shape[-2:]);scores=torch.ones(probability.shape[:2],device=device)
        else:
            if payload is not None:features=torch.stack([payload['features'][index[r['image_id']]].float() for r in batch]).to(device)
            if method=='dino_mc_dropout':
                model.train()
                draws=[model(features) for _ in range(8)]
                masks=torch.cat([d[0] for d in draws],1);ranks=torch.cat([d[1] for d in draws],1)
            elif method.startswith('dino_'):masks,ranks=model(features)
            elif method=='conventional_residual':masks,ranks=model(features,image)
            else:
                raw=model(image);masks,ranks=raw['mask_logits'],raw['validity_logits']
            probability,scores=masks.sigmoid(),ranks.sigmoid()
        for j,row in enumerate(batch):
            order=torch.argsort(scores[j],descending=True,stable=True)
            result[row['image_id']]={'probability':probability[j,order].cpu(),'scores':scores[j,order].cpu()}
    output.mkdir(parents=True);torch.save(result,output/'predictions.pt')
    write_json(output/'frozen.json',dict(predictions_sha256=sha256(output/'predictions.pt'),checkpoint_sha256=sha256(checkpoint),images_sha256=sha256(manifest),sampling_seed=sampling_seed,device=device,batch_size=batch_size,method=method,
            optimizer_seed=ck['optimizer_seed'],visible_seed=ck['visible_seed'],endpoint=ck['endpoint'],execution=execution_record()))
    return result
