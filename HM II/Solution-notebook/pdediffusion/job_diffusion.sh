#!/bin/bash
#SBATCH --job-name=diffusion_metal
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --partition=cpu
#SBATCH --output=outputfolder/diffusion_metal_16.out
#SBATCH --error=outputfolder/diffusion_metal_16.err

# Make a folder for the outputs
mkdir -p outputfolder

#GPU enabled commands
echo 'Starting simulation job with 1 core...'
python3 metalconduction.py 1 > outputfolder/log_n1.txt

echo 'Starting simulation job with 2 core...'
python3 metalconduction.py 2 > outputfolder/log_n2.txt

echo 'Starting simulation job with 4 core...'
python3 metalconduction.py 4 > outputfolder/log_n4.txt

echo 'Starting simulation job with 8 core...'
python3 metalconduction.py 8 > outputfolder/log_n8.txt

echo 'Starting simulation job with 16 core...'
python3 metalconduction.py 16 > outputfolder/log_n16.txt

echo 'Simulation Job finished.'

