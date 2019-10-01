#!/bin/bash

interpreter=python
with_gpu=1

data_dir=/home/palonso/data/ccetl/librosa
exp_dir=/home/palonso/models/ccetl/vgg_tl_MTT
model_dir=/home/palonso/reps/mtgdb-models/src/musicnn_training/weights/MTT_vgg/
source ~/venvs/py3.6_musicnn/bin/activate

${interpreter} -u tl.py ${data_dir} ${exp_dir} ${model_dir} 2 100 -n 5 --with_gpu --seed 6 -te
