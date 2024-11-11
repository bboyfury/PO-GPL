#!/bin/bash

# Define environments and parameters
environments=("PO-Adhoc-Foraging-2-12x12-3f-v0" "Adhoc-wolfpack-v5" "PO-Navigation-2-12x12-v1" "fortattack-v0")
exp_names=("exp_1" "exp_2" "exp_3" "exp_4" "exp_5" "exp_6" "exp_7" "exp_8")
num_particles=("1" "5" "10" "20")

# Loop through each environment, experiment name, and particle count
for env in "${environments[@]}"; do
    for exp in "${exp_names[@]}"; do
        for particles in "${num_particles[@]}"; do
            
            # Set dynamic saving and logging directories
            env_base_name=$(echo $env | sed 's/-/_/g')  # Replace '-' with '_' for naming
            bash_file_name=$(basename "$0" .sh)         # Get the name of the current script without .sh
            saving_dir="param_${env_base_name}_${bash_file_name}"
            logging_dir="logs_${env_base_name}_${bash_file_name}"
            
            # Run the training job
            sbatch --job-name="${env_base_name}_${exp}_${particles}" run_po_d.sh "$env" "$exp" "$particles" "$logging_dir" "$saving_dir"
            sbatch --job-name="${env_base_name}_${exp}_${particles}" run_vae_d.sh "$env" "$exp" "$particles" "$logging_dir" "$saving_dir"
        
        done
            sbatch --job-name="${env_base_name}_${exp}_${particles}" run_gpl_d.sh "$env"  "$exp" "$logging_dir" "$saving_dir"
            sbatch --job-name="${env_base_name}_${exp}_${particles}" run_ae_d.sh "$env"  "$exp" "$logging_dir" "$saving_dir"

    done
done
