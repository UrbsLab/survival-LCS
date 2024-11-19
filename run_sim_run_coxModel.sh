#!/bin/bash
#SBATCH -p defq
#SBATCH --job-name=SLCS
#SBATCH --mem=4G
#SBATCH -o logs/sim_run_coxModel.o
#SBATCH -e logs/sim_run_coxModel.e
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
python sim_run_coxModelRun.py