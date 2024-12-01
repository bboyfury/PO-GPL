#!/bin/bash

#SBATCH --job-name=create-demo-episode
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1

# Define environments and parameters
environments=("PO-Adhoc-Foraging-2-12x12-3f-v0")
exp_names=("exp_1" "exp_2" "exp_3" "exp_4" "exp_5" "exp_6" "exp_7" "exp_8")
pf_particles=("1" "5" "10" "20") # Particle counts for PO-GPL
vae_particles=("1" "5" "10" "20") # Particle counts for VAE-GPL
seeds=(1 2 3 4 5)       # Seed values
max_checkpoint=10       # Maximum checkpoint number

# Loop through each environment
for env in "${environments[@]}"; do

    # Map each environment to the desired env_base_name
    case $env in
        ("PO-Adhoc-Foraging-2-12x12-3f-v0")
            env_base_name="LBF"
            ;;
        ("Adhoc-wolfpack-v5")
            env_base_name="Wolfpack"
            ;;
        ("PO-Navigation-2-12x12-v1")
            env_base_name="CooperativeNavigation"
            ;;
        ("fortattack-v0")
            env_base_name="FortAttack"
            ;;
        (*)
            env_base_name=$(echo $env | sed 's/-/_/g')  # Fallback for unknown environments
            ;;
    esac

    # AE-GPL
    for exp in "${exp_names[@]}"; do
        for checkpoint in $(seq 0 $max_checkpoint); do
                echo " "
                echo "AE-GPL LBF"
                echo "seed number" $seed "checkpoint number" $checkpoint
                python test_ae_demo.py \
                    --lr=0.00025 \
                    --env-name="$env" \
                    --google-cloud="False" \
                    --designated-cuda="gpu:0" \
                    --seed=6 \
                    --eval-init-seed=65 \
                    --eval-seed=6 \
                    --load-from-checkpoint="$checkpoint" \
                    --q-loss-weight=1.0 \
                    --act-reconstruction-weight=0.05 \
                    --lrf-rank=6 \
                    --exp-name="AE/$exp" \
                    --logging-dir="logs_${env_base_name}_AE" \
                    --saving-dir="param_${env_base_name}_AE"
        done
    done

    # GPL-Q
    for exp in "${exp_names[@]}"; do
        for checkpoint in $(seq 0 $max_checkpoint); do
                echo " "
                echo "GPL-Q LBF"
                echo "seed number" $seed "checkpoint number" $checkpoint
                python test_gpl_demo.py \
                    --lr=0.00025 \
                    --env-name="$env" \
                    --google-cloud="False" \
                    --designated-cuda="cuda:0" \
                    --seed=6 \
                    --eval-init-seed=65 \
                    --eval-seed=6 \
                    --load-from-checkpoint="$checkpoint" \
                    --q-loss-weight=1.0 \
                    --act-reconstruction-weight=0.05 \
                    --lrf-rank=6 \
                    --exp-name="GPL/$exp" \
                    --logging-dir="logs_${env_base_name}_GPL" \
                    --saving-dir="param_${env_base_name}_GPL"
        done
    done

    # PO-GPL with specific particles
    for exp in "${exp_names[@]}"; do
        for checkpoint in $(seq 0 $max_checkpoint); do
                for particles in "${pf_particles[@]}"; do
                    echo " "
                    echo "PO-GPL LBF with $particles particles"
                    python test_po_demo.py \
                        --lr=0.00025 \
                        --num-particles="$particles" \
                        --env-name="$env" \
                        --google-cloud="False" \
                        --designated-cuda="cuda:0" \
                        --seed=6 \
                        --eval-init-seed=65 \
                        --eval-seed=6 \
                        --logging-dir="logs_${env_base_name}_PO_GPL_${particles}" \
                        --saving-dir="param_${env_base_name}_PO_GPL_${particles}" \
                        --exp-name="$exp" \
                        --load-from-checkpoint="$checkpoint" \
                        --state-gnn-hid-dims1=50 \
                        --state-gnn-hid-dims2=100 \
                        --state-gnn-hid-dims3=100 \
                        --state-gnn-hid-dims4=50 \
                        --s-dim=100 \
                        --with-noise="False" \
                        --no-bptt="False" \
                        --add-prev-log-prob="False" \
                        --no-gradient-joint-action="False" \
                        --with-rnn-s-processing="False" \
                        --lrf-rank=6 \
                        --stdev-regularizer-weight=0.0 \
                        --s-var-noise=0.0 \
                        --q-loss-weight=1.0 \
                        --agent-existence-prediction-weight=5.0 \
                        --encoding-weight=0.5 \
                        --act-reconstruction-weight=0.5 \
                        --state-reconstruction-weight=0.5 \
                        --entropy-weight=0.05
                done
        done
    done

    # VAE-GPL with specific particles
    for exp in "${exp_names[@]}"; do
        for checkpoint in $(seq 0 $max_checkpoint); do
                for particles in "${vae_particles[@]}"; do
                    echo " "
                    echo "VAE-GPL LBF with $particles particles"
                    python test_vae_demo.py \
                        --lr=0.00025 \
                        --env-name="$env" \
                        --seed=6 \
                        --eval-init-seed=65 \
                        --eval-seed=6 \
                        --load-from-checkpoint="$checkpoint" \
                        --q-loss-weight=1.0 \
                        --act-reconstruction-weight=0.05 \
                        --lrf-rank=6 \
                        --num-particles="$particles" \
                        --exp-name="$exp" \
                        --logging-dir="logs_${env_base_name}_VAE_${particles}" \
                        --saving-dir="param_${env_base_name}_run_vae_${particles}_d"
                done
        done
    done

done

# Final plotting
