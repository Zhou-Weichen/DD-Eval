<p align="center">
  <img src="./assets/logo.png" width="120" />
</p>

# DD-Eval: Dataset Distillation Independent Evaluation Hub

> **"Trust paper results, but verify with tensors."**

`DD-Eval` is an independent reproducibility benchmarking suite for **Dataset Distillation (DD)** research. In a field where distillation pipelines are often computationally heavy and environment-sensitive, this repository serves as a third-party verification baseline to ensure scientific integrity.

We bridge the gap between reported paper claims and real-world reproducibility by providing a clean, decoupled environment for verifying synthetic data.

---

## 🛠️ Our Methodology

To ensure maximum fidelity to original research while maintaining independent oversight, `DD-Eval` operates under a two-phase protocol:

### Phase I: Implementation Fidelity (Current)
* **Code Isolation**: We extract and decouple the **original evaluation subroutines** from source repositories to eliminate framework-level side effects.
* **Data Sourcing**: We prioritize official synthetic tensors. Where unavailable, we re-distill datasets using official training pipelines and host the resulting tensors on [**Hugging Face Datasets**](https://github.com/Zhou-Weichen/DD-Eval) for transparency.
* **Logic Integrity**: By using scripts derived directly from source code, we ensure reported metrics reflect the authors' intended logic, free from external pipeline interference.
### Phase II: Unified Benchmarking (Upcoming)
* **Standardization**: All verified datasets will be subjected to a **unified evaluation platform** featuring standardized architectures, fixed training loops, and synchronized seeds to facilitate objective head-to-head comparisons.

---

## 📈 Verification Status & Roadmap

We focus on whether the results are **consistent** with paper claims rather than chasing precise decimals. Minor performance gaps (±1-3%) are expected due to hardware and environmental variance.

| Method | Venue | Verification | Quality | Resources | Data Source |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **[MTT](https://github.com/GeorgeCazenavette/mtt-distillation)** | CVPR '22 | ✅ | 🟢 | [`scripts/mtt/`](./scripts/MTT/) | [✔️](https://georgecazenavette.github.io/mtt-distillation/tensors/index.html#tensors) |
| **[FreD](https://github.com/sdh0818/FreD)** | NIPS '23 | 🕒 | 🟡 | - |[✔️](https://drive.google.com/drive/folders/1r1OMVv9llejGmpHfK5DpW4m57Dz_SZ2n)|
| **[NCFM](https://github.com/gszfwsb/NCFM)** | CVPR '25 | 📅 |  - | - | - |

***Verification Status***
> ✅ **Verified**: Independent reproduction verified.  
> 🕒 **Ongoing**: Currently being decoupled or benchmarked.  
> 📅 **Backlog**: Planned for future verification.

***Quality Legend:***
> 🟢 **Consistent**: Results are within a reasonable margin of the reported performance.  
> 🟡 **Evaluating**: Initial tests running; benchmarks pending.  
> 🔴 **Significant Gap**: Reproduced results differ substantially from claims.

---

## 🚀 Quick Start

### 1. Requirements
```bash
conda create -n ddeval python=3.9
pip install torch torchvision numpy tqdm
```

---

## ⚖️ Disclaimer & Attribution

* **Source Integrity**: The validation logic provided in this repository is a **simplified and isolated version** of the code found in the original project repositories. All core algorithmic intellectual property and original implementation rights belong to the respective authors.
* **Reference Only**: This project is intended for **benchmarking and independent research reference**. Results provided here are community-driven sanity checks and should not be considered official implementation results.
* **Accuracy & Feedback**: We strive for absolute accuracy. If you identify any errors or believe a **🔴 Significant Gap** status (documented in [`audits/`](./audits/)) results from a misunderstanding of the original method, please **open an issue or [contact us](mailto:zweic008@gmail.com)**. We actively welcome technical clarification to ensure the integrity of this hub.
* **The Gold Standard**: For official results, hyperparameter tuning details, or the full distillation pipeline, **always refer to the original author's repository.** Links to official sources are provided in the verification table above.
