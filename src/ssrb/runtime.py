"""File-to-bank timing with the paper's A40 protocol and precision settings."""
from pathlib import Path
import math
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision import transforms

from .data import IMAGE_KEYS, load_image, read_rows, sha256, within, write_json
from .execution import record as execution_record
from .training import seed_all

METHODS=('semantic','ssrb','raw_capacity','wide_raw','probunet','cimd','sam2_amg','sam2_point')
PAPER_PARAMETERS=dict(semantic=86136339,ssrb=87230851,raw_capacity=1094512,
                      wide_raw=1597162,probunet=1790121,cimd=88340058,sam2_amg=224446642,
                      sam2_point=310579383)


def online_features(encoder,images):
    outputs=encoder.get_intermediate_layers(images,n=[5,8,11],reshape=False,return_class_token=True)
    patches=[out[0] if isinstance(out,tuple) else out for out in outputs]
    tokens={int(value.shape[1]) for value in patches}
    if len(tokens)!=1:
        raise ValueError('DINO layers have different token grids')
    side=int(round(math.sqrt(tokens.pop())))
    if side*side!=patches[0].shape[1]:
        raise ValueError('DINO token grid is not square')
    return torch.stack(patches,dim=1)


def make_runner(method,paths,*,checkpoint=None,dino_repo=None,dino_weights=None,
                cimd_root=None,sam_repo=None,sam_weights=None,device='cuda'):
    if method not in METHODS:
        raise ValueError('unsupported runtime method')
    device=torch.device(device)
    encoder=None;model=None;ck=None;diffusion=None;generator=None
    if method in {'semantic','ssrb'}:
        from .training import load_model
        from .features import load_local_dinov3
        model,ck=load_model(checkpoint,str(device))
        if ck['stage']!=('semantic' if method=='semantic' else 'residual'):
            raise ValueError('runtime method and core checkpoint differ')
        if dino_repo is None or dino_weights is None:
            raise ValueError('DINO repository and weights are required')
        repo=Path(dino_repo).resolve()
        head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        dirty=subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
        if head!='54694f7627fd815f62a5dcc82944ffa6153bbb76' or dirty:
            raise ValueError('requires the clean recorded DINOv3 checkout')
        encoder=load_local_dinov3(dino_repo,dino_weights).to(device).eval().requires_grad_(False)
        image_size=128
    elif method in {'sam2_amg','sam2_point'}:
        from .proposals import SAM2_COMMIT
        if sam_repo is None or sam_weights is None:
            raise ValueError('SAM2 repository and weights are required')
        repo=Path(sam_repo).resolve()
        head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        dirty=subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
        if head!=SAM2_COMMIT or dirty:
            raise ValueError('requires the clean recorded SAM2 checkout')
        sys.path.insert(0,str(repo))
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        model=build_sam2(config_file='configs/sam2.1/sam2.1_hiera_l.yaml',
                        ckpt_path=str(Path(sam_weights).resolve()),device=str(device),
                        mode='eval',apply_postprocessing=True)
        if method=='sam2_amg':
            generator=SAM2AutomaticMaskGenerator(model)
        else:
            from .comparisons import load
            from .features import load_local_dinov3
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            predictor=SAM2ImagePredictor(model)
            sam_parameters=sum(p.numel() for p in model.parameters())
            model,ck=load(checkpoint,str(device))
            if ck['method']!='dino_single':
                raise ValueError('point timing requires the trained Single predictor')
            if dino_repo is None or dino_weights is None:
                raise ValueError('point timing requires DINO repository and weights')
            repo=Path(dino_repo).resolve()
            head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
            dirty=subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
            if head!='54694f7627fd815f62a5dcc82944ffa6153bbb76' or dirty:
                raise ValueError('requires the clean recorded DINOv3 checkout')
            encoder=load_local_dinov3(dino_repo,dino_weights).to(device).eval().requires_grad_(False)
        image_size=1024
    else:
        from .comparisons import load
        model,ck=load(checkpoint,str(device),cimd_root=cimd_root)
        if ck['method']!=method:
            raise ValueError('runtime method and comparison checkpoint differ')
        image_size=ck['config']['image_size']
        if method=='cimd':
            from .cimd import create_diffusion
            diffusion=create_diffusion(steps=1000,cimd_root=Path(cimd_root))
    transform=transforms.Compose([transforms.Resize((448 if encoder is not None else image_size,)*2),
        transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    amp=method in {'semantic','ssrb','raw_capacity','wide_raw'}

    @torch.no_grad()
    def step(index):
        path=paths[index]
        if method=='sam2_point':
            from .proposals import generated_positive_point
            with Image.open(path) as opened:
                rgb=opened.convert('RGB');image=np.asarray(rgb)
                dino_input=transform(rgb)[None].to(device)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                logits,_=model(online_features(encoder,dino_input))
            probability=logits[0,0].sigmoid().float().cpu()
            point,_=generated_positive_point(probability,image.shape[:2])
            predictor.set_image(image)
            masks,_,_=predictor.predict(point_coords=point,point_labels=np.ones((1,),dtype=np.int32),
                                        multimask_output=True,return_logits=False)
            return int(len(masks))
        if method=='sam2_amg':
            with Image.open(path) as opened:image=np.asarray(opened.convert('RGB'))
            return int(len(generator.generate(image)))
        if method in {'raw_capacity','wide_raw'}:
            with Image.open(path) as opened:output_hw=(opened.height,opened.width)
            raw=load_image(path,image_size)[None].to(device)
        else:
            with Image.open(path) as opened:
                if method=='cimd':
                    gray=opened.convert('L');output_hw=(gray.height,gray.width)
                    gray=gray.resize((image_size,image_size),Image.Resampling.BILINEAR)
                    image=torch.from_numpy(np.asarray(gray,dtype=np.float32)/255.)[None,None].repeat(1,4,1,1)
                else:
                    rgb=opened.convert('RGB');output_hw=(rgb.height,rgb.width)
                    image=transform(rgb)[None]
                if method=='ssrb':
                    array=np.asarray(rgb.resize((128,128),Image.Resampling.BILINEAR),dtype=np.float32)/255.
                    array=(array-np.asarray([.485,.456,.406],dtype=np.float32))/np.asarray([.229,.224,.225],dtype=np.float32)
                    raw=torch.from_numpy(array).permute(2,0,1).float()[None]
            image=image.to(device)
            if method=='ssrb':raw=raw.to(device)
        if method=='probunet':
            mu,logvar,features,skips=model.encode_prior(image)
            probabilities=[model.decode(features,skips,model.sample(mu,logvar)).sigmoid() for _ in range(8)]
            return F.interpolate(torch.cat(probabilities,dim=1),size=output_hw,mode='bilinear',align_corners=False)>=.5
        if method=='cimd':
            from .cimd import sample_mask
            expanded=image[:,None].expand(-1,8,-1,-1,-1).reshape(8,*image.shape[1:]).contiguous()
            signed=sample_mask(model.model,diffusion,expanded)
            probability=((signed+1.)/2.).clamp(0.,1.)
            return F.interpolate(probability,size=output_hw,mode='bilinear',align_corners=False)[:,0]>=.5
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda' and amp):
            if encoder is not None:
                features=online_features(encoder,image)
                logits,ranks=model(features,raw) if method=='ssrb' else model(features)
            else:
                bank=model(raw);logits,ranks=bank['mask_logits'],bank['validity_logits']
            probability=F.interpolate(logits.sigmoid(),size=output_hw,mode='bilinear',align_corners=False)
            order=torch.argsort(ranks,dim=1,descending=True,stable=True)
            return probability.gather(1,order[:,:,None,None].expand_as(probability))>=.5

    parameters=sum(p.numel() for p in model.parameters())
    external=sum(p.numel() for p in encoder.parameters()) if encoder is not None else 0
    if method=='sam2_point':external+=sam_parameters
    if method=='sam2_amg':external=parameters
    task=0 if method=='sam2_amg' else parameters
    complexity=dict(total_system_parameters=parameters+(0 if method=='sam2_amg' else external),
        task_trainable_parameters=task,externally_pretrained_parameters=external,
        native_k='up_to_8_after_native_ranking' if method=='sam2_amg' else 8,
        precision='bf16_autocast' if amp and device.type=='cuda' else 'float32')
    if method=='cimd':
        complexity.update(inference_active_parameters=sum(p.numel() for p in model.model.parameters()),
                          reverse_model_calls=1000,sample_step_evaluations=8000)
    if method=='sam2_amg':
        grid_points=sum(len(grid) for grid in generator.point_grids)
        complexity.update(automatic_grid_points_per_image=grid_points,
                          mask_decoder_batches_per_image=math.ceil(grid_points/generator.points_per_batch))
    if method=='sam2_point':
        complexity.update(native_k=3,precision='dino_bf16_sam_fp32' if device.type=='cuda' else 'float32',
                          image_encoder_passes_per_image=2,mask_decoder_passes_per_image=2)
    return step,complexity


def benchmark(root,manifest,output,*,method,checkpoint=None,dino_repo=None,dino_weights=None,
              cimd_root=None,sam_repo=None,sam_weights=None,device='cuda',seed=None,
              subset_size=8,warmup_runs=100,timed_runs=500,smoke=False):
    root,output=Path(root),Path(output)
    if output.exists():raise FileExistsError(output)
    if min(subset_size,timed_runs)<1 or warmup_runs<0:raise ValueError('invalid timing counts')
    rows=sorted(read_rows(manifest),key=lambda r:r['image_id'])
    if any(set(r)!=IMAGE_KEYS or r['split']!='test' for r in rows):
        raise ValueError('runtime requires image-only test rows')
    if len({r['image_id'] for r in rows})!=len(rows):raise ValueError('duplicate runtime images')
    required_seed=3946 if method=='ssrb' else 3967 if method in {'raw_capacity','wide_raw'} else 3670
    seed=required_seed if seed is None else seed
    if not smoke:
        counts=(8,1,24,required_seed) if method=='sam2_point' else (8,100,500,required_seed)
        if (subset_size,warmup_runs,timed_runs,seed)!=counts:
            raise ValueError('paper timing requires its recorded sources, warm-ups, timed runs and seed')
        if not device.startswith('cuda') or not torch.cuda.is_available() or torch.cuda.get_device_name()!='NVIDIA A40':
            raise RuntimeError('paper timing requires one NVIDIA A40')
        if torch.cuda.device_count()!=1 or str(torch.__version__)!='2.9.1+cu128' or torch.version.cuda!='12.8':
            raise RuntimeError('paper timing requires one visible GPU and PyTorch 2.9.1+cu128')
        if len(rows)<8 or any(r['endpoint']!='lidc' or r['visible_seed']!=0 for r in rows):
            raise ValueError('paper timing requires the LIDC visible-choice-zero test manifest')
    selected=rows[:subset_size]
    paths=[within(root,r['image_path']) for r in selected]
    inputs=dict(manifest_sha256=sha256(manifest),image_sha256=[sha256(p) for p in paths])
    artifacts={name:sha256(path) for name,path in [('checkpoint',checkpoint),
        ('dino_weights',dino_weights),('sam_weights',sam_weights)] if path}
    if not smoke:
        shapes=[]
        for path in paths:
            with Image.open(path) as opened:shapes.append(opened.size)
        if set(shapes)!={(96,96)}:raise ValueError('paper runtime sources must retain their native 96x96 ROI grid')
    seed_all(seed)
    runner,complexity=make_runner(method,paths,checkpoint=checkpoint,dino_repo=dino_repo,dino_weights=dino_weights,
        cimd_root=cimd_root,sam_repo=sam_repo,sam_weights=sam_weights,device=device)
    if not smoke and complexity['total_system_parameters']!=PAPER_PARAMETERS[method]:
        raise ValueError('model size differs from the reported runtime model')
    output.mkdir(parents=True)
    gpu=torch.device(device).type=='cuda'
    for index in range(warmup_runs):
        runner(index%len(paths))
        write_json(output/'progress.json',dict(phase='warmup',completed=index+1,total=warmup_runs))
    if gpu:
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    times=[]
    for index in range(timed_runs):
        if gpu:torch.cuda.synchronize()
        start=time.perf_counter();runner(index%len(paths))
        if gpu:torch.cuda.synchronize()
        times.append(time.perf_counter()-start)
        write_json(output/'progress.json',dict(phase='timed',completed=index+1,total=timed_runs))
    if inputs!=dict(manifest_sha256=sha256(manifest),image_sha256=[sha256(p) for p in paths]):
        raise ValueError('runtime inputs changed')
    if artifacts!={name:sha256(path) for name,path in [('checkpoint',checkpoint),
        ('dino_weights',dino_weights),('sam_weights',sam_weights)] if path}:
        raise ValueError('runtime checkpoint or pretrained weights changed')
    array=np.asarray(times,dtype=np.float64)
    record=dict(status='smoke-complete' if smoke else 'paper-protocol-complete',method=method,
        protocol=('custom_smoke' if smoke else 'single_point_one_warmup_three_passes'
                  if method=='sam2_point' else '100_warmups_500_timed_runs'),
        runtime_seed=seed,subset_rule='cycle_first_sorted_unique_test_images',subset_size=len(paths),
        input_seal=inputs,complexity=complexity,execution=execution_record()|dict(
            cpu_threads=torch.get_num_threads(),omp_threads=os.environ.get('OMP_NUM_THREADS'),
            mkl_threads=os.environ.get('MKL_NUM_THREADS')),
        timing_scope='file I/O, preprocessing, transfer, complete native bank and original-grid readout',
        timing=dict(batch_size=1,warmup_runs=warmup_runs,timed_runs=timed_runs,
            seconds_total=float(array.sum()),seconds_per_run_mean=float(array.mean()),
            seconds_per_run_std=float(array.std(ddof=1)) if len(array)>1 else 0.,
            seconds_per_run_median=float(np.median(array)),seconds_per_run_p95=float(np.quantile(array,.95)),
            cuda_peak_memory_bytes=int(torch.cuda.max_memory_allocated()) if gpu else 0))
    record.update({name+'_sha256':value for name,value in artifacts.items()})
    write_json(output/'samples.json',times)
    write_json(output/'runtime.json',record)
    return record
