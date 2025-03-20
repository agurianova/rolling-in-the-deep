docker build -t sunny:0.1 .

docker run --rm -itd --gpus=0 --ipc=host \
	-v /data/glusterfs/agurianova/:/data/glusterfs/agurianova/ \
	-w /data/glusterfs/agurianova/rolling-in-the-deep \
	-p 10000:10000 \
	sunny:0.1 \
	python experiments/2025-W8-02-20/training.py \
	--config experiments/2025-W8-02-20/configs/efficientnet_b3_20000.yaml 

#python experiments/2025-W3-01-13/training.py --config experiments/2025-W3-01-13/configs/EfficientNet_b0.yaml

docker run --rm -itd --gpus=0 --ipc=host \
	-v /data/glusterfs/agurianova/:/data/glusterfs/agurianova/ \
	-w /data/glusterfs/agurianova/rolling-in-the-deep \
	-p 10000:10000 \
	sunny:0.1 \
	bash

docker run --rm -itd --gpus=0 --ipc=host \
	-v /data/glusterfs/agurianova/:/data/glusterfs/agurianova/ \
	-w /data/glusterfs/agurianova/rolling-in-the-deep \
	-p 10000:10000 \
	sunny:0.1 \
	python experiments/2025-W8-02-20/training.py \
	--config experiments/2025-W8-02-20/configs/inception_v3_20000.yaml 

python experiments/2025-W8-02-20/training.py --config experiments/2025-W8-02-20/configs/inception_v3_20000.yaml 

python experiments/2025-W8-02-20/training.py --config experiments/2025-W8-02-20/configs/regnety_032_20000.yaml 

python experiments/2025-W8-02-20/training.py --config experiments/2025-W8-02-20/configs/efficientnet_b7_20000.yaml