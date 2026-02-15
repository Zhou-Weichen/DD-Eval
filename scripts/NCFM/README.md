# 🔬 NCFM Evaluation Report 

### This directory documents the **reproduction results** for the **[NCFM](https://github.com/gszfwsb/NCFM)** method. (CVPR 2025).

---

## 🛠️ Getting Started

To get started with NCFM, follow the installation instructions below.

#### 0. Clone the repo #### 

```sh
git clone https://github.com/gszfwsb/NCFM.git
```

####  1. Install dependencies #### 
   
```sh
pip install -r requirements.txt
```
---
## 📦 2. Data Preparation
####  1. Download the condensed dataset from [huggingface](https://huggingface.co/maomaocun/NCFM) #### 

```bash
cd NCFM

# Create directory for distilled datasets
# All generated or downloaded distilled datasets will be stored here
mkdir download
cd download
git clone https://huggingface.co/maomaocun/NCFM
mv NCFM/NCFM_distillation_dataset/Tiny\ ImageNet/  NCFM/NCFM_distillation_dataset/Tiny_ImageNet

# Create directory for test datasets
# This folder stores CIFAR10, CIFAR100, TinyImageNet, etc.
# CIFAR10 and CIFAR100 will be downloaded automatically.
# TinyImageNet needs to be downloaded manually (see instructions below).
cd ..
mkdir dataset
```

#### 2. Prepare Tiny-ImageNet

Use our [script](https://github.com/Zhou-Weichen/DD-Eval/blob/main/scripts/MTT/download_tiny.py]) to download and restructure it:

```bash
python download_tiny.py
```

⚠️ Important:

For validation data, **change the output path** from 

`dataset/tiny/val/images` to `dataset/tinyimagenet/val/` to ensure compatibility with the data loading logic in [`utils.py`](https://github.com/gszfwsb/NCFM/blob/9e3e60b855fa918337013c8c3d460601690eb58e/utils/utils.py)

---

## 🧪 3. Reproduction & Evaluation

####  1. Evaluation #### 

All commands for running the evaluation are provided in the commands.txt file. 
For convenience, it is recommended to copy this file into the `NCFM/` directory before execution.
Here are examples of how to run them:

```sh
cd evaluation 
torchrun --nproc_per_node=2 --nnodes=1 evaluation_script.py --gpu=0,1 --ipc=1 --config_path=../config/ipc1/cifar10.yaml --load_path=../download/NCFM/NCFM_distillation_dataset/CIFAR-10/CIFAR10_ipc1.pt

torchrun --nproc_per_node=2 --nnodes=1 evaluation_script.py --gpu=0,1 --ipc=10 --config_path=../config/ipc10/cifar10.yaml --load_path=../download/NCFM/NCFM_distillation_dataset/CIFAR-10/CIFAR10_ipc10.pt

torchrun --nproc_per_node=2 --nnodes=1 evaluation_script.py --gpu=0,1 --ipc=50 --config_path=../config/ipc50/cifar10.yaml --load_path=../download/NCFM/NCFM_distillation_dataset/CIFAR-10/CIFAR10_ipc50.pt
```

> [!TIP]
> **Note 1:** During the first execution, the validation dataset download may occasionally fail due to network instability. If this happens, simply rerun the same command and it should proceed normally.
>
> **Note 2:** Please adjust `--nproc_per_node`, `--nnodes`, and `--gpu` according to your available hardware and GPU configuration.
>
> **Note 3:** Detailed experimental settings and hyperparameters can be found in the corresponding configuration files under the [`config/`](https://github.com/gszfwsb/NCFM/tree/9e3e60b855fa918337013c8c3d460601690eb58e/config) directory.

---
## 📊 4. My Reproduction Results

| Dataset | IPC | My Result (%) | 
| :--- | :---:| :---: |
| **CIFAR-10** | 1  |~48.247% | 
|              | 10 |~70.157% |
|               | 50 | ~77.500% |
| **CIFAR-100** | 1  | ~30.590% |
|               | 10 | ~49.264% |
|               | 50 | ~54.386% |
| **Tiny-ImageNet** | 1  | ~12.968% | 
|               | 10 | ~24.327% | 
|               | 50 | ~26.991% | 






