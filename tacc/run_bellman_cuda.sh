#!/bin/bash
#SBATCH -J run_bellman           # Job name
#SBATCH -o cuda_out_%j       # Name of stdout output file
#SBATCH -e cuda_err_%j       # Name of stderr error file
#SBATCH -p gtx          # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 02:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=sgarland@utexas.edu
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A EE-382C-EE-361C-Mult       # Allocation name (req'd if you have more than 1)
cd /home1/07460/garlands/gpu_graph_algorithms/build
pwd
date
if [ $# -gt 0 ]; then
  echo "Input file : $1"
  echo "BLOCK Size : $2"
  ./bellman cuda $1 1024 1024
fi
date
#Submit a job using command $sbatch run_bellman_cuda.sh ../input/USA-road-d.NY.gr 1024
