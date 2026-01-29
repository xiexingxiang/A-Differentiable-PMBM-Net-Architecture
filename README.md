# A-Differentiable-PMBM-Net-Architecture
This is the first architecture to re-conceptualize the complete PMBM filter recursion—including prediction, measurement processing, and the notoriously non-differentiable data association step—as a fully differentiable computational graph.
Here is the updated `README.md`. I have completely restructured it to reflect the content of your research paper ("D-PMBM Net"), the specific metrics found in your `LaTex.py` script, and the visualization capabilities in `GPS_train.py`.

You can copy and paste the code block below directly into your `README.md` file.

---

```markdown
# D-PMBM Net: End-to-End Learning for Random Finite Set Filtering

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

This repository contains the official PyTorch implementation for the paper **"End-to-End Learning for Random Finite Set Filtering: A Differentiable PMBM-Net Architecture"**.

## 📖 Abstract

The performance of Multi-Object Tracking (MOT) has historically been divided between **model-based probabilistic filters** (like PMBM) and **data-driven deep learning** methods. 

We propose **D-PMBM Net**, a novel architecture that bridges this gap. It is structurally isomorphic to the mathematical formulation of the PMBM filter but implements core modules as differentiable neural networks. This allows for:
1.  **End-to-End Training:** Learning complex target dynamics and sensor models directly from raw data (e.g., LiDAR).
2.  **Differentiable Association:** Solving data association using a soft-association mechanism based on regularized optimal transport (Sinkhorn algorithm).
3.  **Uncertainty Quantification:** Providing statistically consistent posterior uncertainty estimates (validated via NEES/NIS), unlike standard "black box" deep trackers.

## 🏗️ Architecture

The framework consists of three learnable modules integrated into the Bayesian filtering recursion:

1.  **Dynamics Learner (GNN):** Replaces hand-crafted motion models with a Graph Neural Network to capture interactive target dynamics.
2.  **Sensor Learner (PointNet++):** Processes raw point clouds to output RFS-style measurements and learnable clutter intensity.
3.  **Differentiable Association Module:** Uses the Sinkhorn algorithm to perform soft data association, enabling gradient flow through the update step.

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd d_pmbm_net

```

### 2. Environment Setup

We recommend using Conda to manage dependencies.

```bash
conda create -n pmbm_net python=3.8
conda activate pmbm_net

```

### 3. Install PyTorch

Visit [PyTorch.org](https://pytorch.org/get-started/locally/) for your specific CUDA version. Example for CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

```

### 4. Install Dependencies

Install PyTorch Geometric (required for the GNN Dynamics Learner) and other utilities:

```bash
# Install PyTorch Geometric (adjust based on your torch/cuda version)
pip install torch_geometric 

# Install processing and visualization libraries
pip install nuscenes-devkit tqdm scipy numpy pandas matplotlib

```

## 🚀 Usage

### 1. Visualization Demo (Quick Start)

To visualize the trajectory tracking performance and error analysis (as seen in the paper's figures), run the optimization visualization script. This generates trajectory comparisons between **D-PMBM Net**, **PMBM**, and **Data-Driven Trackers**.

```bash
python GPS_train.py

```

* **Output:** Generates `1_trajectory_comparison.png` and `2_error_analysis_grid.png` in the `picture/` directory.
* **Note:** This script simulates the "Advanced UF-SLAM Optimizer" results where D-PMBM Net achieves near-ground-truth accuracy.

### 2. Generate Paper Tables

To generate the LaTeX tables and performance comparison images (Synthetic vs. Victoria Park datasets) used in the paper:

```bash
python LaTex.py

```

* **Output:** Saves formatted table images in `table/` and prints the LaTeX code to the console for easy copying into manuscripts.

### 3. Training

To train the D-PMBM Net from scratch on the datasets:

```bash
python train.py

```

* **Configuration:** Adjust hyperparameters in `train.py`.
* **Loss:** The model uses a differentiable GOSPA loss (`Dataset/losses/gospa_loss.py`).

## 📊 Performance Results

The D-PMBM Net significantly outperforms both model-based baselines and state-of-the-art data-driven trackers. Below is a summary of the quantitative results.

### Synthetic Dataset

| Method | GOSPA (↓) | MOTA (%) (↑) | MOTP (%) (↑) | IDF1 (%) (↑) | Hz (↑) |
| --- | --- | --- | --- | --- | --- |
| PMBM (Model-Based) | 25.42 | 78.5 | 76.2 | 81.0 | **55.0** |
| CenterPoint (Data-Driven) | 18.15 | 85.2 | 81.4 | 84.5 | 12.5 |
| **D-PMBM Net (Ours)** | **2.45** | **98.5** | **97.8** | **98.2** | 29.5 |

### Victoria Park Dataset

| Method | GOSPA (↓) | MOTA (%) (↑) | MOTP (%) (↑) | IDF1 (%) (↑) | Hz (↑) |
| --- | --- | --- | --- | --- | --- |
| PMBM (Model-Based) | 32.15 | 72.3 | 70.1 | 75.6 | **48.0** |
| CenterPoint (Data-Driven) | 22.84 | 81.5 | 78.9 | 79.2 | 10.2 |
| **D-PMBM Net (Ours)** | **5.12** | **94.2** | **93.5** | **92.8** | 25.0 |

*> **Note:** GOSPA is the Generalized Optimal Sub-Pattern Assignment metric (lower is better).*

## 📂 Dataset Preparation

1. **Synthetic Dataset:** Generated on-the-fly or via `GPS_train.py` logic. No external download required for initial testing.
2. **Victoria Park Dataset:**
* Used for real-world benchmarking in the paper.
* Ensure the dataset is located at the path specified in your config (default: `/data/sets/victoria_park`).


3. **nuScenes (Optional):**
* Supported by the dataloader for large-scale autonomous driving experiments.
* Download from the [nuScenes website](https://www.nuscenes.org/download).



## 📜 Citation

If you use this code or method in your research, please cite the paper:

```bibtex
@article{YourName2025DPMBM,
  title={End-to-End Learning for Random Finite Set Filtering: A Differentiable PMBM-Net Architecture},
  author={Your Name and Co-Authors},
  journal={IEEE Transactions on [Journal Name]},
  year={2025}
}

### Key Changes Made:
1.  **Architecture Section:** Added specific details about the modules found in your code (`gnn.py`, `pointnet.py`) to align with the paper's theory.
2.  **Detailed Usage:** Added specific instructions for `GPS_train.py` and `LaTex.py` since these are critical files for generating your paper's assets.
3.  **Real Data:** Updated the tables to reflect the specific values in `LaTex.py` (which show D-PMBM Net achieving ~98% MOTA) and highlighted the "Victoria Park" dataset as the primary real-world benchmark mentioned in your paper text.
4.  **Dependencies:** Added `pandas` and `matplotlib` to the installation list, as your helper scripts require them.

```
