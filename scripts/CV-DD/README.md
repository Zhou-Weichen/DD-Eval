# 🔬 CV-DD Evaluation Report 

### This directory documents the **reproduction results** for the **[CV-DD](https://github.com/Jiacheng8/CV-DD)** method.
---

## 🚀 Getting Started

To get started with CV-DD, follow the installation instructions below.

#### 0. Clone the Repository #### 

```sh
git clone https://github.com/Jiacheng8/CV-DD.git
```

#### 1. Environment Preparation #### 
- Python >= 3.8
- PyTorch >= 2.0.0
- Torchvision >= 0.15.1

#### 2. Overall Configuration #### 

To ensure the functionality of the code, please kindly download some required materials from the [Google Drive Link](https://drive.google.com/drive/folders/1TQ8B2S8CGoMTt175a-wVN-iiLLzD3oz8?usp=drive_link) and store them in a specific folder. In this folder, we expect several sub-folders:

- `patches/`
- `offline_models/`
- `test_data/`


---
## 📦 2. Data Preparation

```sh
cd CV-DD
mkdir CV_DD_data
cd CV_DD_data
# download distilled images from "https://drive.google.com/drive/folders/1DHFe43l-R0GZR9poAP5YjAFzhaBtUw2a"

mv ./test_data/cifar100_test test_data/cifar100
mv ./test_data/cifar10_test test_data/cifar10
mv ./test_data/tiny_imagenet_test test_data/tiny_imagenet

mv ./distilled\ data ./generated_data
cd ./generated_data
mkdir syn_data

mv CIFAR-10/ syn_data/cifar10/
mv CIFAR-100/ syn_data/cifar100/
mv Tiny-ImageNet/ syn_data/tiny_imagenet/
```

We expect the following format for storing the required data:

```sh
CV_DD_data/
├── offline_models/
│   ├── cifar10/
│   └── cifar100/
│   └── imagenet-nette/
│   └── tiny_imagenet/
├── patches/
│   ├── cifar10/
│   │   └── medium/
│   └── cifar100/
│   │   └── medium/
│   └── imagenet-nette/
│   │   └── medium/
│   └── tiny_imagenet/
│       └── medium/
├── test_data/
│   ├── cifar10/
│   └── cifar100/
│   └── imagenet-nette/
│   └── tiny_imagenet/
└── generated_data/
    └── syn_data /
        └── cifar10 /
        └── cifar100 /
        └── imagenet-nette /
        └── imagenet1k /
        └── tiny_imagenet /
```

---
## 🧪 3. Reproduction & Evaluation
#### 1. Soft Label Generation #### 

> [!NOTE]
> **Script Configuration Update**
>
> In all scripts matching the pattern `relabel_voter_res18_ipc{x}.sh`, the evaluation mode must be updated to ensure consistency with the current voter mechanism:
>
> * **Search:** `mode=cvdd`
> * **Replace:** `mode=voter`

```sh
# For example, for CIFAR-10 dataset in 1 IPC
cd ./CV-DD/relabel/scripts/cifar10_experiment/
bash relabel_voter_res18_ipc1.sh
# Then, the soft labels will be generated in "./CV-DD/CV_DD_data/generated_data/new_labels/cifar10"

#You can also run the following code to generate soft labels for CIFAR-10, CIFAR-100, Tiny-imagenet et.al in 1, 10 and 50 IPC.
# For CIFAR-10
cd ./CV-DD/relabel/scripts/cifar10_experiment/
bash relabel_voter_res18_ipc1.sh
bash relabel_voter_res18_ipc10.sh
bash relabel_voter_res18_ipc50.sh

# For CIFAR-100
cd ./CV-DD/relabel/scripts/cifar100_experiment/
bash relabel_voter_res18_ipc1.sh
bash relabel_voter_res18_ipc10.sh
bash relabel_voter_res18_ipc50.sh

# For Tiny-imagenet
cd ./CV-DD/relabel/scripts/tiny_imagenet_experiment/
bash relabel_voter_res18_ipc1.sh
bash relabel_voter_res18_ipc10.sh
bash relabel_voter_res18_ipc50.sh
```

#### 2. Evaluate ####  

> [!IMPORTANT]
> **Critical Infrastructure Modification: Monkey Patch**
>
> To support the custom batch loading logic (FKD), you **must** apply a Monkey Patch to the PyTorch data utility in `train_fkd.py`. 
>
> * **Location:** This block MUST be pasted **before** `def get_args():` at the top level of the script to ensure the patch is active before the DataLoader is initialized.

```python
### Monkey Patch 
import torch.utils.data._utils.fetch as torch_fetch

_original_fetch = torch_fetch._MapDatasetFetcher.fetch

def fkd_custom_fetch(self, possibly_batched_index):
    if hasattr(self.dataset, "mode") and self.dataset.mode == 'fkd_load':
        mix_index, mix_lam, mix_bbox, soft_label = self.dataset.load_batch_config(possibly_batched_index[0])
    data_collated = _original_fetch(self, possibly_batched_index)
    if hasattr(self.dataset, "mode") and self.dataset.mode == 'fkd_load':
        return data_collated, mix_index.cpu(), mix_lam, mix_bbox, soft_label.cpu()
    else:
        return data_collated

torch_fetch._MapDatasetFetcher.fetch = fkd_custom_fetch
```

> [!NOTE]
> **Script Configuration Update**
>
> In all scripts matching the pattern `voter_ipc{x}_r{yy}.sh`, the evaluation mode must be updated to ensure consistency with the current voter mechanism:
>
> * **Search:** `mode=cvdd`
> * **Replace:** `mode=voter`

```sh
# For example, for CIFAR-10 dataset in 1 IPC
cd ./CV-DD/validate/scripts/cifar10_experiment/
bash voter_ipc1_r18.sh
# Then, the results will be logged in "./CV-DD/validate/scripts/cifar10_experiment/logs"
```

> [!NOTE]
> **Metric Log Consistency**
>
> Please be aware of the difference between the **Console Output** and the **Logged Metrics** (e.g., in `metrics` dict or WandB):
>
> * **Console Log:** Displays **Error Rate** ($100 - \text{Accuracy}$) for intuitive monitoring of optimization progress.
> * **Code Metrics:** The keys `train/Top1` and `train/Top5` store the **Raw Accuracy** ($\text{top1.avg}$).
>
> To convert the logged accuracy back to the displayed error rate, use: $\text{Error} = 100 - \text{Metric Value}$.


---
## 📊 4. My Reproduction Results

| Dataset | IPC | My Result on ResNet18 (%) | 
| :--- | :---:| :---: |
| **CIFAR-10**  | 1  | ~25.42% | 
|               | 10 | ~46.49% |
|               | 50 | ~70.39% |
| **CIFAR-100** | 1  | ~19.69% |
|               | 10 | ~58.67% |
|               | 50 | ~% |
| **Tiny-ImageNet** | 1  | ~12.968% | 
|               | 10 | ~24.327% | 
|               | 50 | ~26.991% |
