#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --time=6-12:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

ENV_NAME=$1
EXP_NAME=$2
LOGGING_DIR=$3
SAVING_DIR=$4

source activate myenv

export OMP_NUM_THREADS=1

# To prevent memory leak in LIAM select cuda device here


python gpl_only_state_recon_train.py --lr=0.00025  --env-name="$ENV_NAME" \
	--google-cloud="False" --designated-cuda="cuda:0" \
	--seed=2 --eval-seed=700 \
	--act-reconstruction-weight=0.005  --states-reconstruction-weight=0.001  \
	--agent-existence-reconstruction-weight=0.02 --s-dim=100 --h-dim=100 --lrf-rank=6  \
	--q-loss-weight=1.0 --exp-name="$EXP_NAME"  --logging-dir="$LOGGING_DIR" --saving-dir="$SAVING_DIR" 
