#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --time=5-12:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


ENV_NAME=$1
EXP_NAME=$2
LOGGING_DIR=$3
SAVING_DIR=$4

source activate myenv

export OMP_NUM_THREADS=1

#export CUDA_VISIBLE_DEVICES="4"



python gpl_only_train.py --lr=0.00025 --env-name="$ENV_NAME" \
    --google-cloud="False" --designated-cuda="cuda:0" \
    --seed=2 --eval-seed=700 \
	--q-loss-weight=1.0 --act-reconstruction-weight=0.05 --lrf-rank=6  \
	--exp-name="$ENV_NAME" --logging-dir="$LOGGING_DIR" --saving-dir="$SAVING_DIR" 
