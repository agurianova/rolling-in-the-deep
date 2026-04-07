<div align="center">

<img src="https://news.cuanschutz.edu/hubfs/Department%20of%20Biomedical%20Infomatics/52524%20junk%20DNA.png"
     alt="DNA visualization"
     width="800"
     style="border-radius: 10px; opacity: 0.25;" />

<h1>
Deep Learning Methods for Variant Calling<br>
</h1>

<h3><strong>Master's Thesis Project</strong></h3>

<p align="center">
  <a href="https://youtu.be/i6OqCpfdnAY?si=xjITalZX-aISt_lr&t=3298">
    <img src="Defense.svg" height="30" style="border-radius: 16px 0 0 16px; margin-right: -2px;"/>
  </a><a href="https://drive.google.com/file/d/1Wv_D9xOa79J2z3SO22s0QFRjZ7WBd-Fw/view?usp=sharing">
    <img src="Slides.svg" height="30" style="border-radius: 0 16px 16px 0;"/>
  </a>
</p>


</div>

---

## Overview
<img src="https://fonts.gstatic.com/s/i/productlogos/googleg/v6/24px.svg" height="20" style="vertical-align:middle"/> **[DeepVariant](https://github.com/google/deepvariant)** transforms variant calling from NGS data into an image classification problem:

**Input:** Sequencing data encoded as images (nucleotides, alignment, quality etc)  
**Output:** Images classified as hom ref/het/hom alt

## Objective
Test if modern architectures beat **Inception baseline**:

| Model                | Role       | Parameters | F1 Score             |
|----------------------|------------|------------|----------------------|
| Inception         | Baseline   | 27.2M      | 0.9302 ± 0.0156      |
| ResNeXt     | 🧪 Tested  | 83.5M      | 0.9426 ± 0.0167 ↑ †  |
| RegNet           | 🧪 Tested  | 145.0M     | 0.9458 ± 0.0184 ↑ *  |
| **EfficientNet**  | 🧪 Tested  | 66.3M  | 0.9565 ± 0.0178 ↑ ** |

** p<0.01, * p<0.05, † p<0.1

## Key Findings
EfficientNet and RegNet improve F1-score and reduce false positives compared to Inception. However, superior performance on general image benchmarks does not necessarily translate to variant calling (ResNeXt). Domain-specific evaluation remains critical.

## Acknowledgments
<p align="center">
  Conducted as part of the Master’s program at
</p>

<p align="center">
  <a href="https://www.facebook.com/itmo.uni/">
    <img src="https://itmo.ru/file/pages/213/slogan_na_plashke_belyy-01.png" height="60"/>
  </a>
</p>

<p align="center">
  <span style="font-size: 14px;">
    <img src="https://api.iconify.design/tabler/map-pin-filled.svg?color=%239d4edd" height="18" style="vertical-align: -3px; margin-right: 6px;"/>
    <b>Saint Petersburg, Russia</b>
  </span>
</p>

## Directory Structure

- `data/deepvariant/`: Contains PNG files and a CSV file (`data_subset_size.csv`) with metadata about the dataset (needed for image downloading in training.py).
- `docker/`: Includes a `Dockerfile` for creating environment.
- `experiments/`: Main directory - stores experiment configurations, results, and logs. Each experiment is organized by timestamp (e.g., `2025-W12-03-20`), containing:
  - `configs/`: files for training models (keep training and validation parameters).
  - `results/`: intermediate results for each architecture and experiment.
  - `tables/`: models statistics and summaries.
  - `tb/`: TensorBoard logs for visualizing training performance.
  - `training.py`: Script for training models.
- `notebooks/`: Jupyter notebooks for data analysis and visualization:
- `src/`: Contains the main source code including custom class for combining datasets with relevant number of channels.
  - `model.py`: Defines model architectures (models are downloaded from timm lib).
  - `transforms.py`: Data preprocessing and transforms.
  - `utils.py`: Utility functions.
- `requirements.txt`: Alternative way to create environment.

## Setup Instructions

### 1. Clone the repository:
```bash
git clone https://github.com/agurianova/rolling-in-the-deep.git
```
### 2. Create a virtual environment:
```bash
python -m venv venv
pip install -r requirements.txt
```
### 3. Build and run the Docker container:
If you prefer using Docker, you can follow these steps to build and run the container:

Build the Docker image:
```bash
cd docker
docker build -t your_image:0.1 .
```
Run the Docker container:
```bash
docker run --rm -itd --gpus '"device=0"' --cpus=32 --ipc=host \
  -v /path/to/rolling-in-the-deep/:/path/to/rolling-in-the-deep/ \
  -w /path/to/rolling-in-the-deep \
  -p 10000:10000 \
  your_image:0.1 bash
```
### 4. Run the experiment:
Once inside the container, run the training script with the desired configuration. For example, to run the experiment with the ResNeXt101-64x4d architecture, use:
```bash
python experiments/2025-W12-03-20/training.py --config experiments/2025-W12-03-20/configs/resnext101_64x4d_30000.yaml
```
This will start the training process for the selected model architecture with the specified configuration file.
