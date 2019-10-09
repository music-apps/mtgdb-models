#!/bin/bash

interpreter=python
with_gpu=1

data_dir=/home/palonso/data/ccetl/audioset
exp_dir=/home/palonso/models/ccetl/vggish_tl_audioset
model_dir=/home/palonso/reps/mtgdb-models/src/musicnn_training/weights/audioset_vggish/checkpoint
source ~/venvs/py3.6_musicnn/bin/activate

${interpreter} -u tl.py ${data_dir} ${exp_dir} ${model_dir} 20 100 -n 5 --with_gpu --seed 6 -fte
