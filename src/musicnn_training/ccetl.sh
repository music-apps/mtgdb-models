#!/bin/bash

interpreter=python
with_gpu=1

data_dir=/home/palonso/data/tlcce_crosseval_tl
exp_dir=/home/palonso/data/tlcce_musicnn_tl_den_MTT
groundtruth=/home/palonso/reps/gt_crosseval.tsv
index=/home/palonso/data/tlcce_crosseval_tl/index_clean.tsv
base_dir=/home/palonso/data/tlcce_msd_audio

source ~/venvs/py3.6_musicnn/bin/activate

${interpreter} -u ccetl.py ${data_dir} ${exp_dir} ${groundtruth} ${index} ${base_dir} librosa -e --with_gpu --transfer_learning
