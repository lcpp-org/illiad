#!/bin/bash
#
#SBATCH --time=02:00:00                     # Job run time (hh:mm:ss)
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --cpus-per-task=32                  # Number of compute threads per proc
#SBATCH --job-name=test11                   # Name of batch job
#SBATCH --partition=eng-research-gpu        # Partition (queue)
#SBATCH --gpus-per-node=0                   # Avoid getting emails from IT
#SBATCH --output=test11_out.o%j             # Name of batch job output file
#SBATCH --error=test11.e%j                  # Name of batch job error file
##SBATCH --mail-user=spgula2@illinois.edu   # Send email notifications
##SBATCH --mail-type=BEGIN,END              # Type of email notification to send
###########################################################################

module load python/3.9.16
export PYTHONPATH=/home/${USER}/scratch/mypython3:${PYTHONPATH}

python3 Poincare.py
