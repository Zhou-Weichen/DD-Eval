# 🔬 FreD Evaluation Report 

### This directory documents the **reproduction results** for the **[FreD](https://github.com/sdh0818/FreD)** method. (NeurIPS 2023).
---
## 🚀 1. Getting Started

```bash
pip install -r requirements.txt

# Create directory for test datasets
# This folder stores CIFAR10, CIFAR100, TinyImageNet, etc.
# CIFAR10 and CIFAR100 will be downloaded automatically.
# TinyImageNet needs to be downloaded manually (see instructions below).
mkdir data

# Create directory for distilled datasets
# All generated or downloaded distilled datasets will be stored here
mkdir download
```
---

## 📦 2. Data Preparation

#### 2.1 Tiny-ImageNet

Use our [script](https://github.com/Zhou-Weichen/DD-Eval/blob/main/scripts/MTT/download_tiny.py]) to download and restructure it:

```bash
python download_tiny.py
```

#### 2.2 Pre-distilled Data

To evaluate official results without running the expensive distillation process:
- Download the distilled data from the Official Release [Link](https://drive.google.com/drive/folders/1r1OMVv9llejGmpHfK5DpW4m57Dz_SZ2n)

> [!TIP]
> Note that distilled data for CIFAR‑100 and Tiny datasets are available only for IPC=1; no data are provided for IPC=10 or IPC=100.
>

---
## 🧪 3. Reproduction & Evaluation

We provide a unified script eval.sh to train models on distilled data and test them on the validation set.

#### 3.1 Configure your Paths

Before execution, open [eval.sh](./eval.sh) and verify the following core arguments:

| Argument | Description | Recommendation / Details |
| :--- | :--- | :--- |
| **`DATA_PATH`** | Root directory for datasets. | CIFAR auto-downloads; Tiny-ImageNet requires manual prep (Step 2.1). |
| **`LOAD_PATH`** | Path to distilled `.pt` files. | Path to [Official Tensors](https://georgecazenavette.github.io/mtt-distillation/tensors/index.html#tensors). |
| **`DATASET`** | Target dataset name. | `CIFAR10`, `CIFAR100`, or `Tiny`. |

---

#### 🔍 EVAL_MODE Detailed Mapping

#### 3.2 Run Evaluation
```bash
bash eval.sh
```

## 📊 4. My Reproduction Results

| Dataset | IPC | My Result (%) | 
| :--- | :---: | :---: |
| **CIFAR-10** | 1 | ~59.7% | 
|  | 10 | ~69.9% |
|  | 50 |  ~74.1% |
| **CIFAR-100** | 1 | ~34.1% |
|  | 10 | ~-% |
|  | 50 | ~-% |
| **Tiny-ImageNet** | 1 | ~-% | 
|  | 10 | ~-% | 
|  | 50 | ~-% | 

