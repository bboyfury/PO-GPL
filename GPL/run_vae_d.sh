#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --time=6-12:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


ENV_NAME=$1
EXP_NAME=$2
NUM_PARTICLES=$3
LOGGING_DIR=$4
SAVING_DIR=$5
source activate myenv

export OMP_NUM_THREADS=1

export CUDA_VISIBLE_DEVICES="0"
python gpl_only_stochastic_state_recon_train.py --lr=0.00025 --env-name="$ENV_NAME" \
    --seed=6 --eval-seed=5 --kl-div-loss-weight=0.001 \
	--q-loss-weight=1.0 --states-reconstruction-weight=0.001 --agent-existence-reconstruction-weight=0.02 --act-reconstruction-weight=0.05 --lrf-rank=6 --num-particles="$NUM_PARTICLES" \
	--exp-name="$EXP_NAME" --logging-dir="$LOGGING_DIR" --saving-dir="$SAVING_DIR" 

