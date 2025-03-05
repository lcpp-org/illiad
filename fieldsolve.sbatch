#!/bin/bash
#
#SBATCH --time=00:10:00                     # Job run time (hh:mm:ss)
#SBATCH --nodes=1                           # Number of nodes to request
#SBATCH --ntasks-per-node=32                # Number of cpus per node
#SBATCH --job-name=fieldSolve               # Name of batch job
#SBATCH --partition=eng-research-gpu        # Partition (queue)
#SBATCH --gpus-per-node=1                   # Avoid getting emails from IT
#SBATCH --output=fieldSolve.o%j             # Name of batch job output file
#SBATCH --error=fieldSolve.e%j              # Name of batch job error file
#SBATCH --mail-user=spgula2@illinois.edu    # Send email notifications
#SBATCH --mail-type=END                     # Type of email notification to send
#SBATCH --account=dcurreli-npre-eng         # Who is your handler?

###########################################################################


module load python/3
export PYTHONPATH=${HOME}/scratch/mypython3:${PYTHONPATH}

python3 fieldsolve.py