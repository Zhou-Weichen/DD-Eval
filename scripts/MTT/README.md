# 🔬 MTT Evaluation Report (Independent Audit)

This directory documents the **reproduction results** for the **MTT** method. (CVPR 2022).

## 🚀 Getting Started
If you have an RTX 30XX GPU (or newer), run

```bash
conda env create -f requirements_11_3.yaml
```

If you have an RTX 20XX GPU (or older), run

```bash
conda env create -f requirements_10_2.yaml
```

You can then activate your conda environment with
```bash
conda activate distillation
```

#### 📊 Data Preparation
1. For Tiny-ImageNet, run
```bash
python download_tiny.py 
```

2. Pre-distilled Data
If you wish to evaluate pre-distilled synthetic images directly without running the full MTT distillation process:
- Download the distilled data from the Official Release Link

---
