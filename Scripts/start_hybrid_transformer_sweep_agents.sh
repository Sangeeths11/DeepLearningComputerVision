#!/bin/sh

SWEEP_ID="$1"

python3 pythonScript/HybridTransformer.py $SWEEP_ID &
python3 pythonScript/HybridTransformer.py $SWEEP_ID &