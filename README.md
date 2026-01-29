# D-PMBM Net: End-to-End Learning for RFS Filtering

This repository contains the official PyTorch implementation for the paper "End-to-End Learning for Random Finite Set Filtering: A Differentiable PMBM-Net Architecture".

## 1. Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd d_pmbm_net
    ```

2.  **Create a Conda environment:**
    ```bash
    conda create -n pmbm_net python=3.8
    conda activate pmbm_net
    ```

3.  **Install PyTorch:**
    Visit the [official PyTorch website](https://pytorch.org/get-started/locally/) to get the correct command for your CUDA version. For example:
    ```bash
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
    ```

4.  **Install PyTorch Geometric:**
    Follow the official instructions from [PyTorch Geometric documentation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

5.  **Install other dependencies:**
    ```bash
    pip install nuscenes-devkit tqdm scipy numpy
    ```

## 2. Dataset Preparation

1.  **nuScenes:**
    * Download the full nuScenes dataset (or the mini version for quick tests) from the [nuScenes website](https://www.nuscenes.org/download).
    * Extract the data to a directory, e.g., `/data/sets/nuscenes`.
    * Update the `nusc_dataroot` path in `train.py` and `evaluate.py` accordingly.

2.  **Synthetic Dataset:**
    * The synthetic dataset is generated on-the-fly using `data/synthetic_dataset.py`. No preparation is needed.

## 3. Training

To train the D-PMBM Net on nuScenes from scratch, run the main training script:

```bash
python train.py