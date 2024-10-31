#!/bin/sh

SWEEP_ID="$1"

CUDA_VISIBLE_DEVICES=0 python3 pythonScript/VisionTransformer.py $SWEEP_ID &
CUDA_VISIBLE_DEVICES=1 python3 pythonScript/VisionTransformer.py $SWEEP_ID &