# 🔬 MTT Evaluation Report 

### This directory documents the **reproduction results** for the **[MTT](https://github.com/GeorgeCazenavette/mtt-distillation)** method. (CVPR 2022).
---
## 🚀 1. Getting Started

Choose the environment configuration based on your GPU architecture to ensure CUDA compatibility.

#### For RTX 30XX / 40XX / 50XX
```bash
conda env create -f requirements_11_3.yaml
```

#### For RTX 20XX or older
```bash
conda env create -f requirements_10_2.yaml
```

####  Activate environment:
```bash
conda activate distillation
```
---

## 📦 2. Data Preparation

#### 2.1 Tiny-ImageNet

Use our script to download and restructure it:

```bash
python download_tiny.py
```

#### 2.2 Pre-distilled Data

To evaluate official results without running the expensive distillation process:
- Download the distilled data from the Official Release [Link](https://georgecazenavette.github.io/mtt-distillation/tensors/index.html#tensors)

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
| **`EVAL_MODE`** | Model Evaluation Pool. | Controls which architectures will be used for testing (see table below). |

---

#### 🔍 EVAL_MODE Detailed Mapping
`EVAL_MODE` determines the `model_eval_pool`. Use the following codes to select your evaluation strategy:

| Mode | Strategy | Model Pool / Architectures Included |
| :---: | :--- | :--- |
| **`M`** | **Multiple Architectures** | `ConvNet`, `AlexNet`, `VGG11`, `ResNet18_AP`, `ResNet18` |
| **`W`** | **Ablation: Width** | `ConvNetW32`, `W64`, `W128`, `W256` |
| **`D`** | **Ablation: Depth** | `ConvNetD3`, `D4`, `D5` |
| **`3`** | **Single Depth (3)** | `ConvNetD3` |
| **`4`** | **Single Depth (4)** | `ConvNetD4` |
| **`A`** | **Ablation: Activation** | `ConvNetAS` (Sigmoid), `AR` (ReLU), `AL` (LeakyReLU) |
| **`P`** | **Ablation: Pooling** | `ConvNetNP` (None), `MP` (Max), `AP` (Average) |
| **`N`** | **Ablation: Normalization**| `NN` (None), `BN`, `LN`, `IN`, `GN` |

> [!TIP]
> Please align your IPC and EVAL_MODE as follows [Link](https://user-images.githubusercontent.com/18726777/184226412-7bd0d577-225b-487c-8c9c-23f6462ca7d0.png)


#### 3.2 Run Evaluation
```bash
bash eval.sh
```

## 📊 4. My Reproduction Results

| Dataset | IPC | Evaluation Model | EVAL_MODE | My Result (%) | 
| :--- | :---: | :--- | :---: | :---: |
| **CIFAR-10** | 1 | ConvNetD3 | `3` | ~46.0% | 
| **CIFAR-10** | 10 | ConvNetD3 | `3` | ~63.1% |
| **CIFAR-10** | 50 | ConvNetD3 | `3` | ~69.6% |
| **CIFAR-100** | 1 | ConvNetD3 | `3` | ~23.40% |
| **CIFAR-100** | 10 | ConvNetD3 | `3` | ~38.10% |
| **CIFAR-100** | 50 | ConvNetD3 | `3` | ~46.7% |
| **Tiny-ImageNet** | 1 | ConvNetD4 | `4` | ~8.9% | 
| **Tiny-ImageNet** | 10 | ConvNetD4 | `4` | ~20.3% | 
| **Tiny-ImageNet** | 50 | ConvNetD4 | `4` | ~28.4% | 
