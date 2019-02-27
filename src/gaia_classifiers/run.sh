#!/bin/bash

interpreter=python2.7 

data_dir=data/
exp_dir=exp/

cd "${0%/*}"

${interpreter} extract_features.py ${data_dir} -t yaml -s

${interpreter} train_classifiers.py ${data_dir} -o ${exp_dir} -t 10
