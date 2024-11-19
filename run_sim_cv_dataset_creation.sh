#!/bin/bash
#SBATCH -p defq
#SBATCH --job-name=SLCS
#SBATCH --mem=8G
#SBATCH -o logs/cv_dataset_creation.o
#SBATCH -e logs/cv_dataset_creation.e
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
python sim_cv_dataset_creation.py