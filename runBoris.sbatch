#!/bin/bash
#
#SBATCH --time=02:00:00                     # Job run time (hh:mm:ss)
#SBATCH --nodes=1                           # Number of nodes to request
#SBATCH --ntasks-per-node=64                # Number of cpus per node
#SBATCH --job-name=borisMB                  # Name of batch job
#SBATCH --partition=secondary               # Partition (queue)
#SBATCH --gpus-per-node=1                   # Avoid getting emails from IT
#SBATCH --output=borisMB.o%j                # Name of batch job output file
#SBATCH --error=borisMB.e%j                 # Name of batch job error file
#SBATCH --mail-user=spgula2@illinois.edu    # Send email notifications
#SBATCH --mail-type=BEGIN,END               # Type of email notification
#SBATCH --account=dcurreli-npre-eng         # Who is your handler?
###########################################################################

module load python/3
export PYTHONPATH=${HOME}/scratch/mypython3:${PYTHONPATH}

python3 runBorisMB.py