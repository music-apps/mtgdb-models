#!/bin/bash

interpreter=python
with_gpu=1

data_dir=/home/palonso/data/ccetl/essentia
exp_dir=/home/palonso/models/ccetl/vgg_baseline_den

source ~/venvs/py3.6_musicnn/bin/activate

${interpreter} -u run.py ${data_dir} ${exp_dir} -n 5 --with_gpu --seed 6 -te
