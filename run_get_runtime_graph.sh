#!/bin/bash
#SBATCH -p defq
#SBATCH --job-name=SLCS
#SBATCH --mem=16G
#SBATCH -o logs/runtimegraph.o
#SBATCH -e logs/runtimegraph.e
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
python get_runtime_graph.py