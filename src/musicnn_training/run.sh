#!/bin/bash


#SBATCH -J features
#SBATCH -p high
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=out
#STATCH --error=err


source ~/venvs/py3.6_musicnn/bin/activate

interpreter=python

data_dir=/homedtic/palonso/data/tlcce_musicnn_baseline
exp_dir=/homedtic/palonso/models/tlcce_musicnn_baseline

${interpreter} -u run.py ${data_dir} ${exp_dir} -s -n 2

