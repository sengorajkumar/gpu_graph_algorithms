#!/bin/bash
#SBATCH -J run_bellman           # Job name
#SBATCH -o seq_out_%j       # Name of stdout output file
#SBATCH -e seq_err_%j       # Name of stderr error file
#SBATCH -p gtx          # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 01:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=raj.sengo@utexas.edu
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A EE-382C-EE-361C-Mult       # Allocation name (req'd if you have more than 1)
cd /home1/06362/rsengott/project/build
pwd
date
if [ $# -gt 0 ]; then
  echo "Input file : $1"
  echo "Debug      : $2"
  ./bellman seq $1 0 $2
fi
date

