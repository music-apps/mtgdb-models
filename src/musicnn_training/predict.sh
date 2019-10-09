#!/bin/bash

interpreter=python
with_gpu=1
lib=librosa

# data stuff
data_dir=/home/palonso/data/ccetl/${lib}/jamendo_test_split0/
index=/home/palonso/reps/mtg-jamendo-dataset/scripts/index_jam_split0.tsv
base_dir=/home/palonso/data/mtg-jamendo/

# classifier
exp_dir=/home/palonso/models/ccetl/musicnn_tl_MSD/
dataset_name=jamendo_test_split0

source ~/venvs/py3.6_musicnn/bin/activate

${interpreter} -u predict.py ${data_dir} ${exp_dir} ${index} ${base_dir} ${dataset_name} ${lib} --with_gpu -fe
