#!/bin/bash

interpreter=python2.7  

data_dir=${PWD}/data/
exp_dir=${PWD}/exp/

cd "${0%/*}"

${interpreter} -u extract_features.py ${data_dir} -t yaml -s

${interpreter} -u train_classifiers.py ${data_dir} -o ${exp_dir} -t 10
