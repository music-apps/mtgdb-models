#!/bin/bash

source ~/venvs/py3.7_musicnn/bin/activate

interpreter=python

data_dir=/home/pablo/data/tlcce_musicnn_baseline
exp_dir=/home/pablo/models/tlcce_musicnn_baseline

${interpreter} -u extract_features.py ${data_dir} ${exp_dir} -s

# ${interpreter} -u train_classifiers.py ${data_dir} -o ${exp_dir} -t 10
