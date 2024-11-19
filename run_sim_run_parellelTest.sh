#!/bin/bash
#SBATCH -p defq
#SBATCH --job-name=SLCS
#SBATCH --mem=1G
#SBATCH -o logs/test.o
#SBATCH -e logs/test.e
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
python sim_run_parellelTest.py