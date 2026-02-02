# DD-Eval: Dataset Distillation Independent Evaluation Hub

> **"Trust paper results, but verify with tensors."**

`DD-Eval` is an independent auditing suite for **Dataset Distillation (DD)** research. In a field where distillation pipelines are often computationally heavy and environment-sensitive, this repository serves as a third-party "Sanity Check." 

We bridge the gap between reported paper claims and real-world reproducibility by providing a clean, decoupled environment for verifying synthetic data.

---

## 🛠️ Our Methodology

To ensure the highest level of fidelity to the original research, `DD-Eval` follows a strict **"Decouple & Verify"** protocol:

1. **Code Separation**: We isolate the **original evaluation subroutines** from the complex distillation frameworks of the source repositories.
2. **Original Logic Preservation**: We use the original author's validation code (refined for readability) to ensure model training, data augmentation, and preprocessing exactly match the paper's intended environment.
3. **Tensor-First**: We prioritize the use of **officially released synthetic tensors**. Where official tensors are unavailable, we provide results from our own distillation runs using the original hyper-parameters.

---

## 📈 Verification Status & Roadmap

We focus on whether the results are **consistent** with paper claims rather than chasing precise decimals. Minor performance gaps (±1-3%) are expected due to hardware and environmental variance.

| Method | Venue | Verification | Quality | Data Source |
| :--- | :---: | :---: | :---: | :---: |
| **[MTT](https://github.com/GeorgeCazenavette/mtt-distillation)** | CVPR '22 | ✅ | 🟢 | [✔️](https://georgecazenavette.github.io/mtt-distillation/tensors/index.html#tensors) |
| **[FreD](https://github.com/sdh0818/FreD)** | NIPS '23 | 🕒 | 🟡 |[✔️](https://drive.google.com/drive/folders/1r1OMVv9llejGmpHfK5DpW4m57Dz_SZ2n)|
| **[NCFM](https://github.com/gszfwsb/NCFM)** | CVPR '25 | 📅 |  - | - |

***Verification Status***
> ✅ **Verified**: Independent audit completed.  
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



## ⚖️ Disclaimer & Attribution

* **Source Integrity**: The validation logic provided in this repository is a **simplified and isolated version** of the code found in the original project repositories. All core algorithmic intellectual property and original implementation rights belong to the respective authors.
* **Reference Only**: This project is intended for **benchmarking and independent research reference**. Results provided here are community-driven sanity checks and should not be considered official implementation results.
* **Open to Feedback**: We strive for absolute accuracy in our research. If you identify any errors in our scripts or believe there is a misunderstanding regarding the evaluation process, **please open an issue or contact us.** We are committed to actively investigating and correcting any verified mistakes.
* **Audit Transparency**: The **🔴 Significant Gap** status is assigned based on technical observations and logs documented in the [`audits/`](./audits/) directory. If you are the author of a project and believe our evaluation is flawed or based on a misinterpretation of your work, we sincerely welcome technical clarification to ensure the accuracy of this hub.
* **The Gold Standard**: For official results, hyperparameter tuning details, or the full distillation pipeline, **always refer to the original author's repository.** Links to official sources are provided in the verification table above.
