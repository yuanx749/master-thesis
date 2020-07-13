#!/bin/bash -l

#SBATCH -A snic2020-15-148
#SBATCH -p core
#SBATCH -n 3
#SBATCH -t 1-00:00:00
#SBATCH -o "./slurm/slurm-%j.out"
#SBATCH -J louvain

log_file=./logs/$(date +%Y%m%d-%H%M%S).log
python -u ${1} 2>&1 | tee $log_file
echo $(date +%Y%m%d-%H%M%S) >> $log_file
