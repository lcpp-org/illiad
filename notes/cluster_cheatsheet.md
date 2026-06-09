# Cluster Cheatsheet

### Project folder
``` Python
cd /u/basov2/scratch/code/fieldlines-uiuc
pwd # Print working directory
ls  # Lists visible files and folders
sinfo # Check available queues
```

### Running job
``` Python
sbatch [filename].batch # Start the job
scancel JOB_ID          # Cancel the job
```

### Monitor the job
``` Python
squeue -u $USER 
watch squeue -u $USER
To stop watching Ctrl + C
```

### Job States
PD = PENDING -- Job is awaiting resource allocation(queued).\
R = RUNNING -- Job currently has an allocation. \
Q -- Job is queued, eligible to run or routed. \
H -- Job is held.

### Syncying files
``` Python
scp [filename].py basov2@cc-login.campuscluster.illinois.edu:/u/basov2/scratch/code/fieldlines-uiuc/ # Copies file(s) onto the cluster

rsync -avz basov2@cc-login.campuscluster.illinois.edu:/u/basov2/scratch/code/fieldlines-uiuc/output/ ./output/ # Downloads output folder from the cluster into the local output folder
```

Run `scp` and `rsync` from the local terminal/WSL, not from inside the cluster, when copying files between the computer and the cluster.
