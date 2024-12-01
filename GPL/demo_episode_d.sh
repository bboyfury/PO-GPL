#!/bin/bash

#SBATCH --job-name=create-demo-episode-d
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=4G



#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source activate myenv

# Define environments and parameters

environments=("Adhoc-wolfpack-v5" "PO-Navigation-2-12x12-v1")
exp_names=("exp_1")
num_particles=("10")

# Loop through each environment, experiment name, and particle count
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
            env_base_name=$(echo $env | sed 's/-/_/g')  # Fallback if a new environment is added
            ;;
    esac
    #I used separate loops for debugging purposes
    for particles in "${num_particles[@]}"; do
        for exp in "${exp_names[@]}"; do
                
            # Set dynamic saving and logging directories
            bash_file_name=$(basename "$0" .sh)         # Get the name of the current script without .sh
            saving_dir="param_${env_base_name}_run_po_${particles}"
            logging_dir="logs_${env_base_name}_run_po_${particles}"
            
            # Run the training job
            python test_po_demo.py \
                --lr=0.00025 \
                --num-particles="$particles" \
                --env-name="$env" \
                --google-cloud="False" \
                --designated-cuda="cuda:0" \
                --seed=6 \
                --eval-init-seed=65 \
                --eval-seed=5 \
                --logging-dir="$logging_dir" \
                --saving-dir="$saving_dir" \
                --exp-name="$exp" \
                --load-from-checkpoint=40 \
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
