#!/bin/bash

#SBATCH -D /data/glusterfs/agurianova
#SBATCH -N 1
#SBATCH -o /data/glusterfs/agurianova/rolling-in-the-deep/experiments/2025-W3-01-13/logs/sbatch.log
#SBATCH -e /data/glusterfs/agurianova/rolling-in-the-deep/experiments/2025-W3-01-13/logs/sbatch.err
#SBATCH -c 64
#SBATCH --mem-per-cpu 1843M
#SBATCH -t 0
#SBATCH -G 2





#echo "Start"
#source rolling-in-the-deep/venv/bin/activate
#echo "venv"
#which python
#python rolling-in-the-deep/experiments/2025-W3-01-13/training.py --config rolling-in-the-deep/experiments/2025-W3-01-13/configs/EfficientNet_b0.yaml





#docker build -t sunny:0.1 .

docker run --rm -it --gpus=0 --ipc=host \
	-v /data/glusterfs/agurianova/:/data/glusterfs/agurianova/ \
	-w /data/glusterfs/agurianova/rolling-in-the-deep \
	sunny:0.1 \
	python experiments/2025-W3-01-13/training.py \
	--config experiments/2025-W3-01-13/configs/inception_v3.yaml 

docker run --rm -it --gpus=0 --ipc=host \
	-v /data/glusterfs/agurianova/:/data/glusterfs/agurianova/ \
	-w /data/glusterfs/agurianova/rolling-in-the-deep \
	sunny:0.1 \
	python experiments/2025-W3-01-13/training.py \
	--config experiments/2025-W3-01-13/configs/regnety_002.yaml 

docker run --rm -it --gpus=0 --ipc=host \
	-v /data/glusterfs/agurianova/:/data/glusterfs/agurianova/ \
	-w /data/glusterfs/agurianova/rolling-in-the-deep \
	sunny:0.1 \
	python experiments/2025-W3-01-13/training.py \
	--config experiments/2025-W3-01-13/configs/efficientnet_b0.yaml 