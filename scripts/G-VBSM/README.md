# 🔬 G-VBSM Evaluation Report 

### This directory documents the **reproduction results** for the **[G-VBSM](https://github.com/shaoshitong/G_VBSM_Dataset_Condensation)** method. (CVPR 2024).
---
## 🚀 1. Getting Started

```bash
git clone https://github.com/shaoshitong/G_VBSM_Dataset_Condensation.git
```

```bash
pip install numpy --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu118 # torch 2.0
pip install einops timm kornia tqdm wandb prefetch_generator scipy
```

---

## 📦 2. Data Preparation

To evaluate official results without running the expensive distillation process:
- Download the distilled data from [link](https://github.com/shaoshitong/G_VBSM_Dataset_Condensation)

```bash
cd G_VBSM_Dataset_Condensation
mkdir squeeze_Models
cd squeeze_Models
# pretrained models
wget https://github.com/shaoshitong/G_VBSM_Dataset_Condensation/releases/download/v0.0.1/squeeze_wo_ema_cifar10.zip
wget https://github.com/shaoshitong/G_VBSM_Dataset_Condensation/releases/download/v0.0.1/squeeze_wo_ema_cifar100.zip
wget https://github.com/shaoshitong/G_VBSM_Dataset_Condensation/releases/download/v0.0.1/squeeze_wo_ema_tiny_imagenet.zip
unzip squeeze_wo_ema_cifar10.zip
unzip squeeze_wo_ema_cifar100.zip
unzip squeeze_wo_ema_tiny_imagenet.zip
```

```bash
cd G_VBSM_Dataset_Condensation
mkdir recover_Data
cd recover_Data
# download distilled dataset
```


---
## 🧪 3. Reproduction & Evaluation

```bash
# No need to execute squeeze and recover
cd ../relabel
# Adjust the weights and dataset address according to your personal needs.
# Note the default value for `--pre-train-path`
bash ./relabel.sh # (Soft Label Generation Phase)

cd ../train
# Adjust the weights and dataset address according to your personal needs.
bash ./train.sh # (Evaluation Phase)
```
> [!NOTE]
> The code in each subfolder is essentially the same, modified only according to the corresponding dataset for ease of management.

---
## 📊 4. My Reproduction Results

| Dataset | IPC | My Result (%) | 
| :--- | :---: | :---: |
| **CIFAR-10** | 1   | - | 
| **CIFAR-10** | 10  | ~54.07% |
| **CIFAR-10** | 50  | ~80.18% |
| **CIFAR-100** | 1  | ~12.33% |
| **CIFAR-100** | 10 | ~50.38% |
| **CIFAR-100** | 50 | ~59.95% |
| **Tiny-ImageNet** | 1  | -  | 
| **Tiny-ImageNet** | 10 | -  | 
| **Tiny-ImageNet** | 50 | ~% | 



