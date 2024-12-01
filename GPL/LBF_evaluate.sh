#!/bin/bash

# Define environments and parameters
environments=("PO-Adhoc-Foraging-2-12x12-3f-v0")
exp_names=("exp_1" "exp_2" "exp_3" "exp_4" "exp_5" "exp_6" "exp_7" "exp_8")
num_particles=("1" "5" "10" "20")

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
            sbatch --job-name="${env_base_name}_${exp}_${particles}_po" run_po_d.sh "$env" "$exp" "$particles" "$logging_dir" "$saving_dir"
            
            
            saving_dir="param_${env_base_name}_run_vae_d_${particles}"
            logging_dir="logs_${env_base_name}_run_vae_d_${particles}"
            sbatch --job-name="${env_base_name}_${exp}_${particles}_vae" run_vae_d.sh "$env" "$exp" "$particles" "$logging_dir" "$saving_dir"
        done
    done


    for exp in "${exp_names[@]}"; do
            
        # Set dynamic saving and logging directories
        bash_file_name=$(basename "$0" .sh)         # Get the name of the current script without .sh
        
        saving_dir="param_${env_base_name}_run_gpl_d"
        logging_dir="logs_${env_base_name}_run_gpl_d"
        # Submit jobs that don’t use particle parameters
        sbatch --job-name="${env_base_name}_${exp}_gpl" run_gpl_d.sh "$env" "$exp" "$logging_dir" "$saving_dir"
        saving_dir="param_${env_base_name}_run_ae_d"
        logging_dir="logs_${env_base_name}_run_ae_d"
        sbatch --job-name="${env_base_name}_${exp}_ae" run_ae_d.sh "$env" "$exp" "$logging_dir" "$saving_dir"
    done
    
done
