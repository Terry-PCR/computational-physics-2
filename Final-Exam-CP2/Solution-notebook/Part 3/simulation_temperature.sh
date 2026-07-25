#!/bin/bash
#SBATCH --job-name=thermostat_mpi
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16                     
#SBATCH --cpus-per-task=1               
#SBATCH --mem=8G
#SBATCH --partition=cpu
#SBATCH --output=outsimulation/thermostat_mpi.out
#SBATCH --error=outsimulation/thermostat_mpi.err

# Make a folder for the outputs
mkdir -p outsimulation

#CPU enabled commands
echo 'Starting simulation job with 1 core...'
mpirun -n 1 python3 thermostat_mpi.py

echo 'Starting simulation job with 2 core...'
mpirun -n 2 python3 thermostat_mpi.py

echo 'Starting simulation job with 4 core...'
mpirun -n 4 python3 thermostat_mpi.py

echo 'Starting simulation job with 8 core...'
mpirun -n 8 python3 thermostat_mpi.py

echo 'Starting simulation job with 16 core...'
mpirun -n 16 python3 thermostat_mpi.py

echo 'Simulation Job finished.'


