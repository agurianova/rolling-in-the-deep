FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

RUN sed -i 's|^deb.*developer.download.nvidia.com|#&|g' /etc/apt/sources.list.d/*.list || true

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    numpy \
    pandas \
    torch \
    torchvision \
    scikit-learn \
    tqdm \
    tensorboard \
    pyyaml \
    timm \
    seaborn \ 
    matplotlib \
    Pillow \
    tensorboard

WORKDIR /workspace
COPY . /workspace

ENV PYTHONPATH="/workspace/src:${PYTHONPATH}"

EXPOSE 10000
