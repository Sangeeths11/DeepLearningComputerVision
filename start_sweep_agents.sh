#!/bin/sh

SWEEP_ID="$1"

python3 pythonScript/CNN.py $SWEEP_ID &
python3 pythonScript/CNN.py $SWEEP_ID &
python3 pythonScript/CNN.py $SWEEP_ID &
python3 pythonScript/CNN.py $SWEEP_ID &
python3 pythonScript/CNN.py $SWEEP_ID &
python3 pythonScript/CNN.py $SWEEP_ID &