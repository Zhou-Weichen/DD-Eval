## Prepare distilled dataset and generate soft labels

```sh
cd CV-DD
mkdir CV_DD_data
cd CV_DD_data
# download distilled images from "https://drive.google.com/drive/folders/1DHFe43l-R0GZR9poAP5YjAFzhaBtUw2a"

mv ./distilled\ data ./generated_data
cd ./generated_data
mkdir syn_data

mv CIFAR-10/ syn_data/cifar10/
mv CIFAR-100/ syn_data/cifar100/
mv Tiny-ImageNet/ syn_data/tiny_imagenet/
mv ImageNet-1k/ syn_data/imagenet1k
mv ImageNette/ syn_data/imagenet-nette
```

Following format for storing the required data:

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

Generate soft labels:

- NOTE : All 'relabel_voter_res18_ipc**x**.sh' should have 'mode=cvdd' changed to 'mode=voter'.

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

Evaluate:

- Importly, add patch in 'train_fkd.py', paste follow code before 'def get_args():'

```sh
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

- NOTE : All 'voter_ipc**1**_r**xx**.sh' should have 'mode=cvdd' changed to 'mode=voter'.

```sh
# For example, for CIFAR-10 dataset in 1 IPC
cd ./CV-DD/validate/scripts/cifar10_experiment/
bash voter_ipc1_r18.sh 
# Then, the results will be logged in "./CV-DD/validate/scripts/cifar10_experiment/logs"

```

