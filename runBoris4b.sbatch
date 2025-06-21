#!/bin/bash
#
#SBATCH --time=04:00:00                     # Job run time (hh:mm:ss)
#SBATCH --nodes=1                           # Number of nodes to request
#SBATCH --ntasks-per-node=64                # Number of cpus per node
#SBATCH --job-name=Boris4                   # Name of batch job
#SBATCH --partition=secondary               # Partition (queue)
#SBATCH --gpus-per-node=1                   # Avoid getting emails from IT
#SBATCH --output=Boris4.o%j                 # Name of batch job output file
#SBATCH --error=Boris4.e%j                  # Name of batch job error file
#SBATCH --mail-user=spgula2@illinois.edu    # Send email notifications
#SBATCH --mail-type=BEGIN,END               # Type of email notification to send
#SBATCH --account=dcurreli-npre-eng         # Who is your handler?

###########################################################################


module load python/3.11.11
export PYTHONPATH=${HOME}/scratch/mypython3:${PYTHONPATH}

python3 runBorisEfield4.py