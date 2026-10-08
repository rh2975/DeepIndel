# DeepIndel: A ResNet-Based Method for Accurate Insertion and Deletion Detection from Long-Read Sequencing

[![Paper](https://img.shields.io/badge/Paper-IJACSA%202025-blue)](https://doi.org/10.14569/IJACSA.2025.0160850)

This repository contains the implementation and supporting materials for the research paper:

> **Sifat, M. S. H., & Hossain, K. M. R. (2025).**
> **DeepIndel: A ResNet-Based Method for Accurate Insertion and Deletion Detection from Long-Read Sequencing.**
> *International Journal of Advanced Computer Science and Applications (IJACSA), 16(8).*
> https://doi.org/10.14569/IJACSA.2025.0160850

---

## 📌 Overview

**DeepIndel** is a deep learning-based approach for detecting **insertions and deletions (indels)** from long-read sequencing data.

The method uses a **Residual Neural Network (ResNet)-based architecture** to learn discriminative patterns from long-read sequencing signals and improve the identification of insertion and deletion events.

The project was developed to investigate the application of deep learning to long-read variant detection, with particular emphasis on accurate indel identification.

---

## 🧬 Motivation

Insertions and deletions are important forms of genetic variation, but their accurate detection can be challenging, particularly when working with sequencing reads containing errors and complex genomic patterns.

Long-read sequencing provides substantially longer reads than traditional short-read technologies, enabling improved representation of genomic regions that are difficult to resolve using short reads. However, computational methods are still required to distinguish genuine variants from sequencing errors.

DeepIndel addresses this problem by using a **ResNet-based deep learning model** to learn relevant representations for insertion and deletion detection.

---

## 🧠 DeepIndel Approach

The overall DeepIndel workflow can be summarized as:

```text
Long-Read Sequencing Data
          │
          ▼
     Data Processing
          │
          ▼
   Feature Representation
          │
          ▼
   ResNet-Based Model
          │
          ▼
    Model Prediction
          │
          ▼
   Indel Detection
          │
          ├──────────────┐
          ▼              ▼
     Insertion       Deletion
```

The central component of the proposed approach is a **Residual Network (ResNet)** architecture designed to learn useful patterns for insertion/deletion classification.

---

## ✨ Key Features

* ResNet-based deep learning architecture for indel detection
* Designed for **long-read sequencing data**
* Focuses on both **insertions and deletions**
* End-to-end machine learning-based prediction
* Supporting preprocessing and experimental code
* Implementation associated with the published research paper

---

## 📂 Repository Structure

The repository contains the source code, preprocessing components, experimental scripts, and supporting resources required for the DeepIndel project.

A typical structure is:

```text
DeepIndel/
│
├── deepindel_preprocess_train/
│   └── Preprocessing and model-training components
│
├── pepper/
│   └── Supporting genome inference components
│
├── pepper_variant/
│   └── Variant-related components
│
├── docs/
│   └── Documentation
│
├── img/
│   └── Figures and images
│
├── requirements.txt
├── setup.py
├── CMakeLists.txt
├── Makefile
└── README.md
```

The exact contents may vary depending on the version of the implementation.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/rh2975/DeepIndel.git
cd DeepIndel
```

### 2. Create a Python environment

It is recommended to use a dedicated Python environment.

For example:

```bash
python -m venv deepindel_env
```

Activate the environment on Windows:

```bash
deepindel_env\Scripts\activate
```

On Linux/macOS:

```bash
source deepindel_env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Additional system dependencies may be required for building the native components.

---

## 🔧 Build

The project includes CMake-based components.

The repository provides:

* `CMakeLists.txt`
* `Makefile`
* `setup.py`

Build the required extensions using the project's setup configuration.

For example:

```bash
python setup.py build_ext --inplace
```

---

## 🚀 Usage

The repository contains the preprocessing, training, and inference components required for running the DeepIndel workflow.

The general workflow is:

```text
1. Prepare long-read sequencing data
              ↓
2. Preprocess sequencing reads
              ↓
3. Generate model-ready representations
              ↓
4. Train / load the DeepIndel model
              ↓
5. Run inference
              ↓
6. Identify insertion and deletion events
```

Please refer to the source code and accompanying documentation for the specific commands and configuration required for each stage.

---

## 🧪 Experiments

The experiments reported in the paper investigate the effectiveness of the proposed ResNet-based approach for insertion and deletion detection from long-read sequencing data.

For reproducibility, users should keep the following experimental settings consistent with the paper:

* Sequencing dataset
* Reference genome
* Data preprocessing procedure
* Training/validation/test configuration
* Model architecture
* Hyperparameters
* Software dependencies

---

## 📊 Results

The experimental results and analysis are reported in the published paper.

**Paper:**
[Sifat, M. S. H., & Hossain, K. M. R. (2025). DeepIndel: A ResNet-Based Method for Accurate Insertion and Deletion Detection from Long-Read Sequencing. IJACSA, 16(8).](https://doi.org/10.14569/IJACSA.2025.0160850)

For the complete quantitative evaluation, comparisons, and discussion, please refer to the paper.

---

## 📄 Publication

### DeepIndel: A ResNet-Based Method for Accurate Insertion and Deletion Detection from Long-Read Sequencing

**Authors**

* **Md. Shadmim Hasan Sifat**
* **Khandokar Md. Rahat Hossain**

**Journal:** International Journal of Advanced Computer Science and Applications (IJACSA)

**Volume:** 16
**Issue:** 8
**Year:** 2025

**DOI:**
https://doi.org/10.14569/IJACSA.2025.0160850

---

## 📚 Citation

If you use this work, code, or methodology in your research, please cite:

### APA

```text
Sifat, M. S. H., & Hossain, K. M. R. (2025). DeepIndel: A ResNet-Based
Method for Accurate Insertion and Deletion Detection from Long-Read
Sequencing. International Journal of Advanced Computer Science and
Applications, 16(8).
https://doi.org/10.14569/IJACSA.2025.0160850
```

### BibTeX

```bibtex
@article{sifat2025deepindel,
  title   = {DeepIndel: A ResNet-Based Method for Accurate Insertion and Deletion Detection from Long-Read Sequencing},
  author  = {Sifat, Md. Shadmim Hasan and Hossain, K. M. Rezaul},
  journal = {International Journal of Advanced Computer Science and Applications},
  volume  = {16},
  number  = {8},
  year    = {2025},
  doi     = {10.14569/IJACSA.2025.0160850}
}
```

---

## 👨‍💻 Authors

### Md. Shadmim Hasan Sifat
Faculty Member,
Department of Computer Science and Engineering,
Shahjalal University of Science and Technology, Sylhet, Bangladesh
& Graduate Student
Department of Computer Science and Engineering,
Bangladesh University of Science and Technology, Dhaka, Bangladesh

### Khandokar Md. Rahat Hossain
Faculty Member,
Department of Computer Science and Engineering,
United International University, Dhaka, Bangladesh
& Graduate Student
Department of Computer Science and Engineering,
Bangladesh University of Science and Technology, Dhaka, Bangladesh


---

## 📜 License

Please refer to the license included in this repository for information regarding the use, modification, and redistribution of the source code.

If no explicit license is provided, please contact the authors before redistributing the code.

---

## 📬 Contact

For questions regarding the implementation or research, please contact the authors through the contact information provided in the associated publication.

---

## ⭐ Acknowledgement

If this repository or the DeepIndel methodology is useful for your research, please consider citing the associated publication.

**DeepIndel — Deep learning for accurate indel detection from long-read sequencing.**
