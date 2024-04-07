#!/bin/bash
#SBATCH -p defq
#SBATCH --job-name=SLCS2
#SBATCH --mem=16G
#SBATCH -o job1.o
#SBATCH -e job1.e
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
srun python run_complete_pipeline_hpc.py