#!/bin/bash

interpreter=python
with_gpu=1
lib=essentia
data_dir=/home/palonso/data/ccetl/${lib}/ab_msd_audio
exp_dir=/home/palonso/models/ccetl/vgg_baseline_den/
index=/home/palonso/data/ccetl/essentia/ab_msd_audio/index.tsv 
base_dir=/home/palonso/data/tlcce_msd_audio
dataset_name=ab_msd_audio

source ~/venvs/py3.6_musicnn/bin/activate

${interpreter} -u predict.py ${data_dir} ${exp_dir} ${index} ${base_dir} ${dataset_name} ${lib} --with_gpu -e
