"""Train the last DINOv3 block with the decoder, preserving the global batch."""
from pathlib import Path
import math
import time
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from .data import load_visible_mask, read_rows, sha256, validate_visible, within, write_json, IMAGE_KEYS
from .losses import objective
from .models import DirectMaskBank
from .execution import configure as configure_execution, record as execution_record
from .features import load_local_dinov3
from .training import seed_all


def configure_last_block(encoder):
    encoder.requires_grad_(False)
    if len(encoder.blocks) != 12:
        raise ValueError('requires a twelve-block DINOv3 encoder')
    encoder.blocks[11].requires_grad_(True)
    encoder.eval()
    return [name for name, p in encoder.named_parameters() if p.requires_grad]


def quantized_features(encoder, images, microbatch_size=16):
    if microbatch_size < 1:
        raise ValueError('microbatch size must be positive')
    chunks=[]
    for image in images.split(microbatch_size):
        outs=encoder.get_intermediate_layers(image.float(),n=[5,8,11],reshape=False,return_class_token=True)
        chunks.append(torch.stack([out[0] if isinstance(out,tuple) else out for out in outs],1).half().float())
    return torch.cat(chunks,0)


def load_encoder(repo, weights, device):
    return load_local_dinov3(repo, weights).float().to(device).eval()


class Images(Dataset):
    def __init__(self, root, rows, split):
        self.root=Path(root)
        self.rows=sorted([r for r in rows if r['split']==split],key=lambda r:r['image_id'])
        self.transform=transforms.Compose([transforms.Resize((448,448)),transforms.ToTensor(),
                       transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self):
        return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        with Image.open(within(self.root,row['image_path'])) as image:
            result=dict(image=self.transform(image.convert('RGB')),image_id=row['image_id'])
        if 'visible_mask_path' in row:
            result['target']=load_visible_mask(within(self.root,row['visible_mask_path']),(28,28))
        return result


def train(root,manifest,output,dino_repo,weights,*,seed=0,device='cuda',deterministic=False):
    configure_execution(deterministic)
    output,root=Path(output),Path(root)
    if output.exists():raise FileExistsError(output)
    if not device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('the accepted partial-DINO training configuration requires a CUDA GPU')
    if (torch.cuda.get_device_name() != 'NVIDIA A40' or torch.cuda.get_device_capability() != (8,6)
            or str(torch.__version__) != '2.9.1+cu128' or torch.version.cuda != '12.8'):
        raise RuntimeError('the accepted partial-DINO profile requires NVIDIA A40 with PyTorch 2.9.1+cu128')
    rows=read_rows(manifest);validate_visible(rows)
    names={r[k] for r in rows for k in ('image_path','visible_mask_path')}
    seal={n:sha256(within(root,n)) for n in sorted(names)}
    manifest_hash=sha256(manifest)
    encoder=load_encoder(dino_repo,weights,device);trainable=configure_last_block(encoder)
    seed_all(seed)
    args=dict(layers=3,dim=768,queries=8,hidden=256,dropout=0.)
    decoder=DirectMaskBank(**args).to(device)
    loaders={s:DataLoader(Images(root,rows,s),batch_size=32,shuffle=s=='train',num_workers=4,
                          pin_memory=True) for s in ('train','val')}
    optimizer=torch.optim.AdamW([dict(params=decoder.parameters(),lr=.0003),
                 dict(params=[p for p in encoder.parameters() if p.requires_grad],lr=.000003)],weight_decay=.0001)
    # Require the accepted cache quantization profile before training.
    reference=torch.load(root/'train_features.pt',map_location='cpu',weights_only=True)
    first=next(iter(DataLoader(Images(root,rows,'train'),batch_size=32,shuffle=False,num_workers=4)))
    cache_index={k:i for i,k in enumerate(reference['image_ids'])}
    if len(first['image_id']) != 32:
        raise RuntimeError('preflight requires the first 32 training images in sorted order')
    observed=quantized_features(encoder,first['image'].to(device))
    expected=torch.stack([reference['features'][cache_index[k]].float() for k in first['image_id']]).to(device)
    difference=(observed.detach()-expected).abs()
    preflight=dict(split='train',images=32,encoder_microbatch=16,
                   maximum_absolute_error=float(difference.max()),mean_absolute_error=float(difference.mean()),
                   cache_exact=torch.equal(observed.detach(),expected),execution=execution_record())
    output.mkdir(parents=True)
    write_json(output/'preflight.json',preflight)
    if not preflight['cache_exact']:
        raise RuntimeError('initial online features differ from the supplied frozen cache; exact preflight failed')
    masks,ranks=decoder(observed)
    smoke,parts=objective(masks,ranks,first['target'].to(device),training=True,capacity_offset=0)
    smoke.backward()
    def gradient_norm(parameters):
        return sum(float(p.grad.detach().double().square().sum()) for p in parameters if p.grad is not None)**.5
    preflight.update(decoder_gradient_norm=gradient_norm(decoder.parameters()),
                     encoder_gradient_norm=gradient_norm(p for p in encoder.parameters() if p.requires_grad),
                     capacity_counts=torch.bincount(parts['winners'],minlength=8).tolist())
    preflight['smoke_passed']=all(math.isfinite(preflight[k]) and preflight[k]>0 for k in
                                  ('decoder_gradient_norm','encoder_gradient_norm')) and preflight['capacity_counts']==[4]*8
    write_json(output/'preflight.json',preflight)
    if not preflight['smoke_passed']:
        raise RuntimeError('partial-DINO gradient or global capacity preflight failed')
    encoder.zero_grad(set_to_none=True)
    del observed,expected,difference,masks,ranks,smoke,parts,reference
    # Reset after preflight so it does not consume training-loader RNG.
    seed_all(seed);decoder=DirectMaskBank(**args).to(device)
    optimizer=torch.optim.AdamW([dict(params=decoder.parameters(),lr=.0003),
                 dict(params=[p for p in encoder.parameters() if p.requires_grad],lr=.000003)],weight_decay=.0001)
    best=math.inf;bad=0;history=[];state=None;started=time.perf_counter()
    for epoch in range(1,41):
        losses={}
        for split,loader in loaders.items():
            training=split=='train';decoder.train(training);encoder.eval();total=0.;count=0
            with torch.enable_grad() if training else torch.no_grad():
                for i,batch in enumerate(loader):
                    features=quantized_features(encoder,batch['image'].to(device),16)
                    masks,ranks=decoder(features)
                    loss,_=objective(masks,ranks,batch['target'].to(device),training=training,
                                     capacity_offset=(epoch-1)*len(loader)+i)
                    if not torch.isfinite(loss):raise ValueError('nonfinite objective')
                    if training:
                        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                    n=len(batch['image_id']);total+=float(loss.detach())*n;count+=n
            losses[split]=total/count
        history.append(dict(epoch=epoch,train_objective=losses['train'],validation_objective=losses['val'],eligible=epoch>4))
        print(history[-1],flush=True)
        if epoch>4 and losses['val']<best-1e-12:
            best,best_epoch,bad=losses['val'],epoch,0
            state=dict(decoder={k:v.detach().cpu().clone() for k,v in decoder.state_dict().items()},
                       block={k:v.detach().cpu().clone() for k,v in encoder.blocks[11].state_dict().items()})
        elif epoch>4:
            bad+=1
            if bad>=8:break
    if sha256(manifest)!=manifest_hash or any(sha256(within(root,n))!=v for n,v in seal.items()):
        raise ValueError('training inputs changed')
    checkpoint=dict(schema='ssrb-partial-v1',state=state,model_args=args,weights_sha256=sha256(weights),
                    endpoint=rows[0]['endpoint'],visible_seed=rows[0]['visible_seed'],optimizer_seed=seed,
                    best_epoch=best_epoch,best_objective=best,trainable_names=trainable,
                    manifest_sha256=manifest_hash,input_seal=seal)
    torch.save(checkpoint,output/'checkpoint.pt');write_json(output/'history.json',history)
    write_json(output/'run.json',{k:v for k,v in checkpoint.items() if k!='state'}|dict(seconds=time.perf_counter()-started,
               checkpoint_sha256=sha256(output/'checkpoint.pt'),encoder_microbatch=16,global_batch=32,device=device,execution=execution_record()))
    return dict(best_epoch=best_epoch,epochs=len(history))


@torch.no_grad()
def infer(checkpoint,root,manifest,output,dino_repo,weights,*,device='cuda',batch_size=12,deterministic=False):
    configure_execution(deterministic)
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if ck['schema']!='ssrb-partial-v1' or sha256(weights)!=ck['weights_sha256']:
        raise ValueError('partial checkpoint/initial weights differ')
    rows=read_rows(manifest)
    if any(set(r)!=IMAGE_KEYS or r['split']!='test' or r['endpoint']!=ck['endpoint'] or r['visible_seed']!=ck['visible_seed'] for r in rows):
        raise ValueError('requires matching image-only test rows')
    encoder=load_encoder(dino_repo,weights,device);encoder.blocks[11].load_state_dict(ck['state']['block'])
    decoder=DirectMaskBank(**ck['model_args']).to(device).eval();decoder.load_state_dict(ck['state']['decoder'])
    result={}
    for batch in DataLoader(Images(root,rows,'test'),batch_size=batch_size,shuffle=False):
        masks,ranks=decoder(quantized_features(encoder,batch['image'].to(device),16))
        for j,key in enumerate(batch['image_id']):
            order=torch.argsort(ranks[j].sigmoid(),descending=True,stable=True)
            result[key]=dict(probability=masks[j,order].sigmoid().cpu(),scores=ranks[j,order].sigmoid().cpu())
    output.mkdir(parents=True);torch.save(result,output/'predictions.pt')
    write_json(output/'frozen.json',dict(predictions_sha256=sha256(output/'predictions.pt'),
               checkpoint_sha256=sha256(checkpoint),images_sha256=sha256(manifest),device=device,batch_size=batch_size,
               execution=execution_record()))
    return dict(images=len(result))
