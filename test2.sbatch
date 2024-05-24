#!/bin/bash
#
#SBATCH --time=00:30:00                     # Job run time (hh:mm:ss)
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --ntasks-per-node=1                 # Number of MPI processes per node
#SBATCH --job-name=test2                    # Name of batch job
#SBATCH --partition=secondary-eth                   # Partition (queue)
#SBATCH --output=test2_out.o%j              # Name of batch job output file
#SBATCH --error=test2.e%j                   # Name of batch job error file
##SBATCH --mail-user=spgula2@illinois.edu   # Send email notifications
##SBATCH --mail-type=BEGIN,END              # Type of email notification to send
###########################################################################

module load python/3
python3 Poincare.py
