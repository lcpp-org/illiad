#!/bin/bash
#
#SBATCH --time=02:00:00                     # Job run time (hh:mm:ss)
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --cpus-per-task=32                  # Number of compute threads per proc
#SBATCH --job-name=realRun1q3                  # Name of batch job
#SBATCH --partition=eng-research-gpu        # Partition (queue)
#SBATCH --gpus-per-node=0                   # Avoid getting emails from IT
#SBATCH --output=realRun1q3.o%j            # Name of batch job output file
#SBATCH --error=realRun1q3.e%j             # Name of batch job error file
#SBATCH --mail-user=spgula2@illinois.edu    # Send email notifications
#SBATCH --mail-type=END               # Type of email notification to send
#SBATCH --account=dcurreli-npre-eng         # Who is your handler?

###########################################################################

module load python/3.9.16
export PYTHONPATH=/home/${USER}/scratch/mypython3:${PYTHONPATH}

python3 Poincare.py
