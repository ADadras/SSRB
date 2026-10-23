import sys
import types

import numpy as np
import pytest
import torch
from PIL import Image

from ssrb.data import prediction_probability, read_rows, sha256, write_json
from ssrb.proposals import SAM2_COMMIT, evaluate_pool, generated_positive_point, infer
from ssrb.synthetic import prepare_demo


def test_generated_point_interior_and_empty_fallback():
    probability=torch.zeros(9,9)
    probability[3:6,3:6]=1
    point,source=generated_positive_point(probability,(9,9))
    np.testing.assert_array_equal(point,np.array([[4.,4.]],dtype=np.float32))
    assert source=='single_threshold_mask_maximum_interior_distance'
    probability.zero_();probability[2,7]=.3
    point,source=generated_positive_point(probability,(9,9))
    np.testing.assert_array_equal(point,np.array([[7.,2.]],dtype=np.float32))
    assert source=='single_probability_argmax_empty_mask_fallback'


def test_sam_modes_keep_top_eight_separate_from_pool_oracle(tmp_path,monkeypatch):
    package=types.ModuleType('sam2');package.__path__=[]
    build=types.ModuleType('sam2.build_sam');build.build_sam2=lambda **kwargs:object()
    automatic=types.ModuleType('sam2.automatic_mask_generator')
    prompted=types.ModuleType('sam2.sam2_image_predictor')
    points=[]

    class Generator:
        def __init__(self,model):pass
        def generate(self,image):
            return [dict(segmentation=np.full(image.shape[:2],i==8,dtype=np.bool_),
                         predicted_iou=1-i*.01,stability_score=.9) for i in range(9)]

    class Predictor:
        def __init__(self,model):pass
        def set_image(self,image):self.hw=image.shape[:2]
        def predict(self,*,point_coords,point_labels,multimask_output,return_logits):
            assert multimask_output and not return_logits
            np.testing.assert_array_equal(point_labels,np.ones(1,dtype=np.int32))
            points.append(point_coords.copy())
            masks=np.zeros((3,*self.hw),dtype=np.bool_)
            masks[0]=True;masks[1,10:21,10:21]=True
            return masks,np.array([.2,.9,.9],dtype=np.float32),None

    automatic.SAM2AutomaticMaskGenerator=Generator
    prompted.SAM2ImagePredictor=Predictor
    for name,module in [('sam2',package),('sam2.build_sam',build),
                        ('sam2.automatic_mask_generator',automatic),('sam2.sam2_image_predictor',prompted)]:
        monkeypatch.setitem(sys.modules,name,module)
    monkeypatch.setattr('ssrb.proposals.subprocess.check_output',
                        lambda command,**kwargs:SAM2_COMMIT if 'rev-parse' in command else '')
    demo=prepare_demo(tmp_path/'demo');images=demo/'prepared/inference';references=demo/'prepared/evaluator'
    weights=tmp_path/'sam.pt';weights.write_bytes(b'external checkpoint placeholder')
    automatic_output=tmp_path/'automatic'
    bank=infer(images,images/'images.json',automatic_output,tmp_path,weights,device='cpu',retain_full_pool=True)
    assert all(len(prediction_probability(value))==8 and not prediction_probability(value).any() for value in bank.values())
    single=tmp_path/'single';single.mkdir()
    probability=torch.zeros(1,32,32);probability[:,10:21,10:21]=1
    torch.save({r['image_id']:dict(probability=probability,scores=torch.ones(1))
                for r in read_rows(images/'images.json')},single/'predictions.pt')
    write_json(single/'frozen.json',dict(predictions_sha256=sha256(single/'predictions.pt'),
        images_sha256=sha256(images/'images.json'),method='dino_single',checkpoint_sha256='synthetic-reference'))
    point_bank=infer(images,images/'images.json',tmp_path/'point',tmp_path,weights,device='cpu',single_predictions=single)
    assert all(len(prediction_probability(value))==3 for value in point_bank.values())
    for point in points:np.testing.assert_array_equal(point,np.array([[15.,15.]],dtype=np.float32))
    for value in point_bank.values():
        torch.testing.assert_close(value['scores'],torch.tensor([.9,.9,.2]))
    # Only the evaluator reads these references, after both banks are frozen.
    for row in read_rows(references/'hidden.json'):
        for name in row['heldout_mask_paths']:
            path=references/name
            with Image.open(path) as image:shape=image.size
            Image.new('L',shape,255).save(path)
    summary=evaluate_pool(automatic_output,references,references/'hidden.json',tmp_path/'oracle')
    assert summary['synthetic']['OracleMLQ_full_pool']==1
    assert summary['synthetic']['pool_size']==9
    with pytest.raises(ValueError,match='only in automatic'):
        infer(images,images/'images.json',tmp_path/'invalid',tmp_path,weights,
              single_predictions=single,retain_full_pool=True)
    (automatic_output/'full_pool.pt').write_bytes(b'changed after generation')
    with pytest.raises(ValueError,match='changed after freezing'):
        evaluate_pool(automatic_output,references,references/'hidden.json',tmp_path/'changed')
