import io
import json
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import savemat
from ssrb.prepare_bsds import prepare_bsds
from ssrb.prepare_riga import prepare_riga
from ssrb.prepare_ima import prepare_ima
from ssrb.prepare_lidc import bbox_from_masks, stable_split
from ssrb.data import freeze_manifests, read_rows


def test_bsds_conversion(tmp_path):
    source=tmp_path/'source'
    for split in ('train','val','test'):
        (source/'images'/split).mkdir(parents=True)
        (source/'groundTruth'/split).mkdir(parents=True)
        Image.fromarray(np.zeros((10,12,3),dtype=np.uint8)).save(source/'images'/split/'1.jpg')
        seg=np.ones((10,12),dtype=np.uint16)
        seg[:,6:]=2
        gt=np.empty((1,1),dtype=object)
        gt[0,0]={'Segmentation':seg}
        savemat(source/'groundTruth'/split/'1.mat',{'groundTruth':gt})
    # Separate BSDS source IDs are required between splits in the real data.
    assert prepare_bsds(source,tmp_path/'out')==3
    rows=read_rows(tmp_path/'out/all_annotations.json')
    assert all(len(r['mask_paths'])==2 for r in rows)


def test_riga_disc_cup_encoding(tmp_path):
    archive=tmp_path/'riga.zip'
    names=['BinRushed','Magrabia','MESSIDOR_Base1','MESSIDOR_Base2','MESSIDOR_Base3']
    with zipfile.ZipFile(archive,'w') as z:
        for name in names:
            for split in ('train','test'):
                rows=[{'image':'RIGA/fundus.tif','mask':'RIGA/mask.tif'}] if name=='BinRushed' and split=='test' else []
                z.writestr(f'{name}_{split}.csv',pd.DataFrame(rows,columns=['image','mask']).to_csv(index=False))
        im=io.BytesIO();Image.fromarray(np.zeros((8,8,3),dtype=np.uint8)).save(im,format='TIFF');z.writestr('RIGA/fundus.tif',im.getvalue())
        mask=np.zeros((8,8),dtype=np.uint8);mask[1:7,1:7]=128;mask[3:5,3:5]=255
        for g in range(1,7):
            b=io.BytesIO();Image.fromarray(mask).save(b,format='TIFF');z.writestr(f'RIGA/mask-{g}.tif',b.getvalue())
    out=tmp_path/'out';assert prepare_riga(archive,out)==2
    disc=read_rows(out/'riga_disc.json')[0];cup=read_rows(out/'riga_cup.json')[0]
    from ssrb.data import IMAGE_KEYS
    feature_rows=read_rows(out/'feature_images.json')
    assert len(feature_rows)==2 and all(set(r)==IMAGE_KEYS for r in feature_rows)
    assert {r['endpoint'] for r in feature_rows}=={'riga_disc','riga_cup'}
    assert {r['image_id'] for r in feature_rows}=={disc['image_id'],cup['image_id']}
    assert disc['source_id']==cup['source_id']
    a=np.asarray(Image.open(out/disc['mask_paths'][0]))>0
    b=np.asarray(Image.open(out/cup['mask_paths'][0]))>0
    assert a.sum()==36 and b.sum()==4 and np.all(a[b])


def test_ima_annotation_order_and_visible_cycle(tmp_path):
    source=tmp_path/'source'
    for name in ('raw','segs','images'):(source/name).mkdir(parents=True)
    meta=[]
    for i,split in enumerate(('train','val','test')):
        name=f'ISIC_{i:07d}'
        pd.DataFrame({'ISIC_id':[name]}).to_csv(source/f'raw/{split}.csv',index=False)
        Image.fromarray(np.zeros((8,8,3),dtype=np.uint8)).save(source/f'images/{name}.jpg')
        for j in range(3):
            seg=f'{name}_{j}.png'
            array=np.zeros((8,8),dtype=np.uint8);array[:,j:j+2]=255
            Image.fromarray(array).save(source/'segs'/seg)
            meta.append(dict(ISIC_id=name,seg_filename=seg,annotator=f'A{j}'))
    pd.DataFrame(meta[::-1]).to_csv(source/'raw/seg_metadata_multiannotator_subset.csv',index=False)
    out=tmp_path/'out';assert prepare_ima(source,out)['rows']==3
    choices=[]
    for seed in range(3):
        dest=tmp_path/f'prepared{seed}'
        freeze_manifests(out/'all_annotations.json',out,dest,seed)
        rows=read_rows(dest/'trainer/visible.json')
        choices.append((dest/'trainer'/rows[0]['visible_mask_path']).read_bytes())
    # A three-annotation cycle has distinct indices even when masks coincide.
    assert len(set(choices))==3
    rows=read_rows(out/'all_annotations.json')
    assert all(len(r['mask_paths'])==3 for r in rows)


def test_lidc_union_crop_and_patient_split():
    a=np.zeros((160,160),dtype=bool);b=a.copy()
    a[60:65,60:65]=True;b[70:75,80:85]=True
    y0,y1,x0,x1=bbox_from_masks([a,b],pad=24,min_size=96)
    assert y1-y0==x1-x0==96
    assert a[y0:y1,x0:x1].sum()==a.sum()
    assert b[y0:y1,x0:x1].sum()==b.sum()
    assert stable_split('synthetic_patient',.7,.15)==stable_split('synthetic_patient',.7,.15)
