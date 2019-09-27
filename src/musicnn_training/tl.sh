#!/bin/bash

interpreter=python
with_gpu=1

data_dir=/home/palonso/data/tlcce_musicnn_tl
exp_dir=/home/palonso/models/tlcce_musicnn_tl
model_dir=/home/palonso/reps/mtgdb-models/src/musicnn_training/weights/MTT_musicnn/
source ~/venvs/py3.6_musicnn/bin/activate

${interpreter} -u tl.py ${data_dir} ${exp_dir} ${model_dir} -n 5 --with_gpu --seed 6 -fte
