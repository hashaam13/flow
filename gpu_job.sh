#!/bin/bash
#SBATCH --job-name=test_gpu_job
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --output=output_gpu_test.txt
#SBATCH --error=error_gpu_test.txt