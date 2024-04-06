#!/bin/bash
#SBATCH -p defq
#SBATCH --job-name=SLCS
#SBATCH --mem=16G
#SBATCH -o job.o
#SBATCH -e job.e
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
srun python sim_full_pipeline_sLCS_final.py