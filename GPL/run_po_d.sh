#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --time=6-12:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Parse inputs


ENV_NAME=$1
EXP_NAME=$2
NUM_PARTICLES=$3
LOGGING_DIR=$4
SAVING_DIR=$5

source activate myenv

export OMP_NUM_THREADS=1 
export CUDA_VISIBLE_DEVICES="0"

python train_state.py --lr=0.00025 --num-particles="$NUM_PARTICLES" --env-name="$ENV_NAME" \
 --google-cloud="False" --designated-cuda="cuda:0" \
 --seed=6 --eval-init-seed=65 --eval-seed=5 \
 --logging-dir="$LOGGING_DIR" --saving-dir="$SAVING_DIR" --exp-name="$EXP_NAME" --load-from-checkpoint=-1 \
 --state-gnn-hid-dims1=50 --state-gnn-hid-dims2=100 --state-gnn-hid-dims3=100 --state-gnn-hid-dims4=50 --s-dim=100 \
 --with-noise="False" --no-bptt="False" --add-prev-log-prob="False" --no-gradient-joint-action="False" --with-rnn-s-processing="False" --lrf-rank=6 \
 --stdev-regularizer-weight=0.0 --s-var-noise=0.0 --q-loss-weight=1.0 --agent-existence-prediction-weight=5. --encoding-weight=0.5 --act-reconstruction-weight=0.5 --state-reconstruction-weight=0.5 --entropy-weight=0.05
