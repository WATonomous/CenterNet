from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import json

from opts import opts
from detectors.detector_factory import detector_factory

from pathlib import Path

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

def demo(opt):
  if opt.demo_output is not None:
    Path(opt.demo_output).mkdir(parents=True, exist_ok=True)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 0)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)

  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    detector.pause = False
    frameNum = 0
    while True:
        _, img = cam.read()
        if img is None:
          break
        frameNum = frameNum + 1
        if opt.debug >= 1:
          cv2.imshow('input', img)
        ret = detector.run(img)
        time_str = ''
        for stat in time_stats:
          time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print(f"frame {frameNum}: {time_str}")
        if opt.debug >= 1:
          if cv2.waitKey(1) == 27:
              return  # esc to quit
        if opt.demo_output is not None:
          video_name = os.path.splitext(os.path.basename(opt.demo))[0]
          output_file = f"{video_name}.jsonl"
          output_path = os.path.join(opt.demo_output,output_file)
          detections = []
          for classId in ret['results']:
            for detection in ret['results'][classId]:
              bbox = detection[:4].tolist()
              score = detection[4].tolist()
              detections.append({'bbox': bbox, 'score': score, 'class': classId})
          output_obj = {
            'frameName': f"{video_name}.{frameNum}",
            'detections': detections
          }
          with open(output_path,'a') as appender:
            appender.write(f"{json.dumps(output_obj)}\n")
  else:
    if opt.demo_output is not None:
      raise Exception("demo_output not implemented for non-video inputs")
    detector.pause = False # comment this out to step through frame by frame
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]
    
    for (image_name) in image_names:
      ret = detector.run(image_name)
      time_str = ''
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      print(time_str)
if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
