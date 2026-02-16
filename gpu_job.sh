#!/bin/bash
#SBATCH --job-name=test_gpu_job
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --time=22:30:00    
#SBATCH --output=output_gpu_test.txt
#SBATCH --error=error_gpu_test.txt
#SBATCH --constraint="type_e"

set -e  # crash early if something fails, f for h100, e for a100, a b c or none for v100,  #SBATCH --constraint="type_e" 

source ~/.bashrc
conda activate flow_matching

cd ~/flow

python promoter_flow.py # to stop, scancel <job_id>