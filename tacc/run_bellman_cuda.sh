#!/bin/bash
#SBATCH -J run_bellman           # Job name
#SBATCH -o cuda_out_%j       # Name of stdout output file
#SBATCH -e cuda_err_%j       # Name of stderr error file
#SBATCH -p gtx          # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 00:15:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=raj.sengo@utexas.edu
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A EE-382C-EE-361C-Mult       # Allocation name (req'd if you have more than 1)
cd /home1/06362/rsengott/project/build
pwd
date
if [ $# -gt 0 ]; then
  echo "Input file : $1"
  echo "BLOCK Size : $2"
  ./bellman cuda $1 $2 0
fi
date
#Submit a job using command $sbatch run_bellman_cuda.sh ../input/USA-road-d.NY.gr 1024
