#!/bin/bash

echo "Please run commands in this file manually"
exit 1

# CenterNet
mkdir -p /project/tmp-${HOSTNAME}
cd /project/tmp-${HOSTNAME}
# thor
CUDA_VISIBLE_DEVICES=0,1 python ../src/main.py ctdet --exp_id road-${HOSTNAME} --batch_size 20 --lr 5e-4 --gpus 0,1 --num_workers 8 --num_epochs 230 --lr_step 180,210 --dataset road
# delta
CUDA_VISIBLE_DEVICES=0,1 python ../src/main.py ctdet --exp_id road-${HOSTNAME} --batch_size 10 --lr 5e-4 --gpus 0,1 --num_workers 8 --num_epochs 230 --lr_step 180,210 --dataset road --load_model ../models/ctdet_coco_dla_2x.pth



