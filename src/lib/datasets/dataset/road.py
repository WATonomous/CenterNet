from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data

from inspect import currentframe

TRAINVAL_SPLIT=1

def get_linenumber():
    cf = currentframe()
    return cf.f_back.f_lineno

class ROAD(data.Dataset):
  num_classes = 11
  default_resolution = [960, 1280]
  # python ../src/tools/eval_mean_and_std.py ../data/road/annotations/tracking_val_1.json ../data/road/
  # R_mean is 0.509613, G_mean is 0.448581, B_mean is 0.447528
  # R_var is 0.332705, G_var is 0.322863, B_var is 0.319861
  mean = np.array([0.509613, 0.448581, 0.447528],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.sqrt(np.array([0.332705, 0.322863, 0.319861],
                   dtype=np.float32).reshape(1, 1, 3))

  def __init__(self, opt, split):
    super(ROAD, self).__init__()
    self.data_dir = os.path.join(opt.data_dir, 'road')
    self.img_dir = self.data_dir
    if split == 'test':
      raise Exception(f"{__file__}:{get_linenumber()}: not implemented")
    else:
      if opt.task == 'exdet':
        raise Exception(f"{__file__}:{get_linenumber()}: not implemented")
      else:
        self.annot_path = os.path.join(self.data_dir, 'annotations', f'tracking_{split}_{TRAINVAL_SPLIT}.json')

    print(f'==> initializing with {self.annot_path}')
    self.coco = coco.COCO(self.annot_path)
    # self.coco.dataset['']
    self.max_objs = 30
    self._valid_ids = list(self.coco.cats.keys())
    self.class_name = [self.coco.cats[i]['name'] for i in self._valid_ids]
    assert(self.num_classes == len(self.class_name))
    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
    self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                      for v in range(1, self.num_classes + 1)]
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

    self.split = split
    self.opt = opt

    self.images = self.coco.getImgIds()
    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    # import pdb; pdb.set_trace()
    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = self._valid_ids[cls_ind - 1]
        for bbox in all_bboxes[image_id][cls_ind]:
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = bbox[4]
          bbox_out  = list(map(self._to_float, bbox[0:4]))

          detection = {
              "image_id": int(image_id),
              "category_id": int(category_id),
              "bbox": bbox_out,
              "score": float("{:.2f}".format(score))
          }
          if len(bbox) > 5:
              extreme_points = list(map(self._to_float, bbox[5:13]))
              detection["extreme_points"] = extreme_points
          detections.append(detection)
    return detections

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    json.dump(self.convert_eval_format(results), 
                open('{}/results.json'.format(save_dir), 'w'))
  
  def run_eval(self, results, save_dir):
    # result_json = os.path.join(save_dir, "results.json")
    # detections  = self.convert_eval_format(results)
    # json.dump(detections, open(result_json, "w"))
    self.save_results(results, save_dir)
    coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
    coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
