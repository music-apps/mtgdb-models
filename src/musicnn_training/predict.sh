#!/bin/bash

interpreter=python
with_gpu=1

# CLASSIFIER
exp_dir=/home/palonso/models/ccetl/vggish_tl_audioset/

# USE THE CORRECT FEATS
lib=audioset

# DATA STUFF. Uncomment one of the bocks
# jamendo
data_dir=/home/palonso/data/ccetl/${lib}/jamendo_test_split0/
index=/home/palonso/reps/mtg-jamendo-dataset/scripts/index_jam_split0.tsv
base_dir=/home/palonso/data/mtg-jamendo/
dataset_name=jamendo_test_split0

# msd
# data_dir=/home/palonso/data/ccetl/${lib}/lastfm_test_audio/
# index=/home/palonso/exp/ccetl/lastfm_test_audio/index_crosseval.tsv
# base_dir=/home/palonso/data/lastfm_test_audio/
# dataset_name=lastfm_test_audio

source ~/venvs/py3.6_musicnn/bin/activate

${interpreter} -u predict.py ${data_dir} ${exp_dir} ${index} ${base_dir} ${dataset_name} ${lib} --with_gpu -f
