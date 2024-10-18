#!/bin/bash
#
#SBATCH --time=02:00:00                     # Job run time (hh:mm:ss)
#SBATCH --nodes=2                           # Number of nodes to request
#SBATCH --ntasks-per-node=40               # Number of cpus per node
#SBATCH --job-name=1q3ions                  # Name of batch job
#SBATCH --partition=eng-research-gpu               # Partition (queue)
#SBATCH --gpus-per-node=0                   # Avoid getting emails from IT
#SBATCH --output=1q3ions.o%j                # Name of batch job output file
#SBATCH --error=1q3ions.e%j                 # Name of batch job error file
#SBATCH --mail-user=spgula2@illinois.edu    # Send email notifications
#SBATCH --mail-type=END                     # Type of email notification to send
#SBATCH --account=dcurreli-npre-eng         # Who is your handler?

###########################################################################


module load python/3.9.16
#module load gcc/11.2.0
export PYTHONPATH=/home/${USER}/scratch/mypython3:${PYTHONPATH}

python3 runIonShells.py