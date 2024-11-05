#!/bin/bash
#SBATCH --job-name=bbdm3       # Job name
#SBATCH --output=log_slurm/result_cuongdm_3.txt      # Output file
#SBATCH --error=log_slurm/error_cuongdm_3.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --gpus=1                 # Number of GPUs per node
#SBATCH --cpus-per-task=20                             # Number of CPU cores per task
sh ./train-bash-3.sh

