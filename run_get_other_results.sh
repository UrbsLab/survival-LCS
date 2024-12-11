#!/bin/bash
#SBATCH -p defq
#SBATCH --job-name=SLCS
#SBATCH --mem=16G
#SBATCH -o logs/otheroutput.o
#SBATCH -e logs/otheroutput.e
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
python get_other_results.py