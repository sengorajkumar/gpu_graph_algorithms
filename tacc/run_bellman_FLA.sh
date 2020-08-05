#!/bin/bash
#SBATCH -J run_bellman           # Job name
#SBATCH -o run_bellman.o%j       # Name of stdout output file
#SBATCH -e run_bellman.e%j       # Name of stderr error file
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
./bellman cuda ../input/USA-road-d.FLA.gr 1024 0
date
