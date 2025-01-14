#!/bin/bash

#SBATCH -D /data/glusterfs/agurianova
#SBATCH -N 1
#SBATCH -o /data/glusterfs/agurianova/logs/sbatch.log
#SBATCH -e /data/glusterfs/agurianova/logs/sbatch.err
#SBATCH -c 64
#SBATCH --mem-per-cpu 1843M
#SBATCH -t 0
#SBATCH -G 2

python training-2025-W2.py