#!/bin/bash

#SBATCH --job-name=eval_state_vae_d
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1


#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


source activate myenv
# Define environments and parameters
environments=("PO-Adhoc-Foraging-2-12x12-3f-v0")
exp_names=("exp_1" "exp_2" "exp_3" "exp_4" "exp_5" "exp_6" "exp_7" "exp_8")
pf_particles=("1" "5" "10" "20") # Particle counts for PO-GPL
vae_particles=("1" "5" "10" "20") # Particle counts for VAE-GPL
max_checkpoint=40       # Maximum checkpoint number

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
                        --eval-seed=5 \
                        --load-from-checkpoint="$checkpoint" \
                        --q-loss-weight=1.0 \
                        --act-reconstruction-weight=0.05 \
                        --lrf-rank=6 \
                        --num-particles="$particles" \
                        --exp-name="$exp" \
                        --logging-dir="logs_${env_base_name}_run_vae_d_${particles}" \
                        --saving-dir="param_${env_base_name}_run_vae_d_${particles}"
                done
            
        done
    done
done

# # Final plotting
# python plot_demo_all_seeds_checkpoint.py --saving-dir="test_lbf" --env-name="LBF"
