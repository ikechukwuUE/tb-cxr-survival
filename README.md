# 🫁 Multimodal Survival Analysis for Tuberculosis

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**Author:** Dr. Ikechukwu Ephraim Ugbo, MD  
**Focus:** Multimodal Deep Learning (CXR + Clinical Data) for Prognosis

## 📌 Project Overview
This repository implements a **State-of-the-Art (SOTA) Multimodal AI** framework for predicting time-to-event (survival analysis) in Pulmonary Tuberculosis patients. 

Unlike traditional "Late Fusion" models that simply concatenate features, this project utilizes **Cross-Modal Attention mechanisms** [Wang et al., 2025; Zhou et al., 2023]. This allows clinical covariates (e.g., HIV status, Age) to dynamically "attend" to specific spatial regions of the Chest X-ray, mimicking how a radiologist incorporates clinical context into their visual assessment.

### 🔬 Key Features
* **Multimodal Architecture:** Integrates unstructured imaging data (CXR) with structured clinical tabular data.
* **Cross-Modal Attention:** Uses a Transformer-based fusion layer to model non-linear interactions between modalities.
* **Backbone:** `DenseNet121` (ImageNet weights) without pooling to preserve spatial feature grids ($7 \times 7$).
* **Loss Function:** Cox Partial Likelihood (Neural Cox Model).
* **Explainability:** Integrated Grad-CAM visualization to show risk-contributing lung regions.
* **Data Strategy:** Syncs real Shenzhen CXR images with synthetic clinical covariates to simulate realistic data heterogeneity.

---

## 🏗️ Architecture

The model (`TBSurvivalNet`) follows a 3-stage pipeline:

1.  **Visual Encoder:** `DenseNet121` extracts a $7 \times 7 \times 1024$ feature map.
2.  **Clinical Encoder:** Multi-layer Perceptron (MLP) projects variables into a shared embedding dimension ($d=256$).
3.  **Fusion Layer:** A **Multi-Head Cross-Attention** block where:
    * *Query (Q)* = Clinical Embeddings
    * *Key (K) / Value (V)* = Visual Spatial Features
    * *Output* = Context-aware feature vector passed to the Survival Head.

---

## 🚀 Getting Started

### 1. Installation
Clone the repo and install dependencies in a virtual environment:

```bash
git clone [https://github.com/ikechukwuUE/tb-cxr-survival.git](https://github.com/ikechukwuUE/tb-cxr-survival.git)
cd tb-cxr-survival

# Create virtual env
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# Install requirements
pip install numpy pandas tensorflow matplotlib scikit-learn lifelines kagglehub

```

### 2. Data Setup

Due to size constraints, the raw data is **not** included in the repo.
The code automatically handles data fetching:

1. **Images:** The notebook automatically downloads the **Shenzhen Tuberculosis Dataset** via `kagglehub`.
2. **Clinical Data:** A synthetic data generator creates clinical variables (Age, Sex, HIV, etc.) matched 1:1 with the downloaded images.

### 3. Usage

Run the main Jupyter Notebook:

```bash
jupyter notebook notebooks/01_tb_cxr_survival.ipynb

```

* **Step 1:** The notebook will download images and generate the `master_dataset.csv`.
* **Step 2:** It trains the Cross-Modal Attention model.
* **Step 3:** It outputs the Concordance Index (C-Index) and displays Grad-CAM heatmaps.

---

## 📂 Repository Structure

```
tb-cxr-survival/
├── data/                   # (Ignored by Git)
│   ├── raw/                # Images from Kaggle
│   └── processed/          # master_dataset.csv
├── notebooks/
│   └── 01_tb_cxr_survival.ipynb  # Main training & analysis notebook
├── src/
│   ├── config.py           # Hyperparameters (Batch size, LR, etc.)
│   ├── data_utils.py       # Loaders & Synthetic Data Generator
│   ├── model_utils.py      # SOTA Architecture (CrossModalAttention)
│   ├── survival_utils.py   # Cox Loss & C-Index functions
│   └── explainability_utils.py # Grad-CAM implementation
└── outputs/                # Saved models and logs

```

---

## 📚 References

This project implements concepts from recent literature in medical image analysis:

1. **Wang et al. (2025).** "Missing-modality enabled multi-modal fusion architecture for medical data." *Journal of Biomedical Informatics.*
2. **Zhou et al. (2023).** "A transformer-based representation-learning model with unified processing of multimodal input." *Nature Biomedical Engineering.*
3. **D'Souza et al. (2023).** "Fusing modalities by multiplexed graph neural networks for outcome prediction." *Medical Image Analysis.*
4. **Dong et al. (2025).** "Convolutional neural network... to predict outcome from tuberculosis meningitis." *PLOS One.*

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

**License:** MIT

```

```