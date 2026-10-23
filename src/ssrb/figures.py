"""Portable vector plots from frozen evaluator outputs, without bundled images."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image
from .data import read_rows, within, sha256, write_json, prediction_probability


def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.size':9})
    return plt


def results(summary, output):
    output = Path(output)
    if output.exists():raise FileExistsError(output)
    frame=pd.read_csv(summary)
    method='method_canonical' if 'method_canonical' in frame else 'method'
    regime='regime' if 'regime' in frame else 'endpoint'
    if not {method,regime,'MLQ@8','Waste@8'}<=set(frame):
        raise ValueError('requires method, regime, MLQ@8 and Waste@8 columns')
    output.mkdir(parents=True);plt=plotting()
    for endpoint,rows in frame.groupby(regime,sort=True):
        fig,axes=plt.subplots(1,2,figsize=(7.0,2.8),layout='constrained')
        for _,row in rows.iterrows():
            axes[0].scatter(row['Waste@8'],row['MLQ@8'],label=row[method])
            prefixes=[k for k in range(1,9) if f'MLQ@{k}' in row and pd.notna(row[f'MLQ@{k}'])]
            axes[1].plot(prefixes,[row[f'MLQ@{k}'] for k in prefixes],marker='.',label=row[method])
        axes[0].set(xlabel='Waste at eight masks',ylabel='MLQ at eight masks',title=endpoint)
        axes[1].set(xlabel='Number of masks',ylabel='MLQ',xticks=range(1,9))
        axes[0].legend(fontsize=6);fig.savefig(output/(endpoint.lower().replace('/','_')+'.pdf'));plt.close(fig)
    write_json(output/'provenance.json',dict(summary_sha256=sha256(summary),style='portable vector export'))
    return str(output)


def banks(predictions, root, manifest, output, image_id):
    predictions,output=Path(predictions),Path(output)
    if output.exists():raise FileExistsError(output)
    frozen=json.loads((predictions/'frozen.json').read_text())
    if sha256(predictions/'predictions.pt')!=frozen['predictions_sha256']:
        raise ValueError('predictions changed after freezing')
    bank=torch.load(predictions/'predictions.pt',map_location='cpu',weights_only=True)
    rows={r['image_id']:r for r in read_rows(manifest)}
    if any(k not in rows or k not in bank for k in image_id):raise ValueError('unknown example')
    output.mkdir(parents=True);plt=plotting()
    for index,key in enumerate(image_id):
        with Image.open(within(root,rows[key]['image_path'])) as im:image=np.asarray(im.convert('RGB'))
        masks=prediction_probability(bank[key]);count=len(masks)
        fig,axes=plt.subplots(1,count+1,figsize=(1.3*(count+1),1.8),squeeze=False,layout='constrained')
        axes[0,0].imshow(image);axes[0,0].set_title('Image')
        for j in range(count):
            axes[0,j+1].imshow(masks[j]>=.5,cmap='gray',vmin=0,vmax=1)
            axes[0,j+1].set_title(f'{j+1}')
        for ax in axes[0]:ax.axis('off')
        fig.savefig(output/f'bank_{index+1}.pdf');plt.close(fig)
    write_json(output/'provenance.json',dict(predictions_sha256=frozen['predictions_sha256'],image_ids=image_id,
               input_manifest_sha256=sha256(manifest)))
    return str(output)
