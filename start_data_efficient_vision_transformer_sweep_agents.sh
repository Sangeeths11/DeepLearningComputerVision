#!/bin/sh

SWEEP_ID="$1"

python3 pythonScript/DataEfficientImageTransformer.py $SWEEP_ID &
python3 pythonScript/DataEfficientImageTransformer.py $SWEEP_ID &