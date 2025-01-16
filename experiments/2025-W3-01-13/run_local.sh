docker build -t sunny:0.1 .

docker run --rm -it --gpus=0 --ipc=host \
	-v /data/glusterfs/agurianova/:/data/glusterfs/agurianova/ \
	-w /data/glusterfs/agurianova/rolling-in-the-deep \
	sunny:0.1 \
	python experiments/2025-W3-01-13/training.py \
	--config experiments/2025-W3-01-13/configs/EfficientNet_b0.yaml 

#python experiments/2025-W3-01-13/training.py --config experiments/2025-W3-01-13/configs/EfficientNet_b0.yaml

