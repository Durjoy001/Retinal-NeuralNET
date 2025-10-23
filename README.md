# TI-RADS-NeuralNET

This repository contains a comprehensive deep learning framework for **retinal fundus multi-disease detection** using Convolutional Neural Networks (CNNs). The project focuses on comparing different CNN architectures for classifying retinal diseases from fundus images using the RFMiD (Retinal Fundus Multi-Disease Image Dataset). The research is based on the following articles:

[**"Research Article on Thyroid Nodule Classification"**](https://www.sciencedirect.com/science/article/pii/S1877050924031235)
[**"Evaluating the Performance and Clinical Applications of Multiclass Deep Learning Models for Skin Cancer Pathology Diagnosis (ISIC): A Comparative Analysis of CNN, ViT, and VLM"**](https://dl.acm.org/doi/10.1145/3731763.3731793)
[**"Retinal Fundus Multi-Disease Image Dataset (RFMiD): A Dataset for Multi-Disease Detection Research"**](https://ieee-dataport.org/open-access/retinal-fundus-multi-disease-image-dataset-rfmid)
[**"Comparative Analysis of Vision Transformers and Conventional Convolutional Neural Networks in Detecting Referable Diabetic Retinopathy"**](https://www.ophthalmologyscience.org/article/S2666-9145(24)00088-5/fulltext)

---

## 🎯 Project Overview

This project implements **multi-label retinal disease classification** using four different CNN architectures:

### **Primary Focus: Retinal Disease Detection**
- **Task**: Multi-label classification of retinal diseases from fundus images
- **Dataset**: RFMiD Challenge Dataset (3,200 fundus images)
- **Diseases**: 29 different retinal disease categories
- **Architectures**: ResNet50, DenseNet121, InceptionV3, EfficientNet-B3
- **Analysis**: Comprehensive statistical comparison using Delong and McNemar tests

### **Key Features**
- Multi-label disease classification (29 disease categories)
- CNN architecture comparison and benchmarking
- Statistical significance testing
- Comprehensive evaluation metrics (AUC, Sensitivity, Specificity, F1-Score)
- Clinical-grade performance analysis

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Durjoy001/TI-RADS-NeuralNET
cd TI-RADS-NeuralNET
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Retinal Disease Classification
```bash
python src/train_cnn.py
```

### 4. Generate Visualizations
```bash
python src/visualize_data.py
```

---

## 📊 Dataset Information

### **RFMiD Retinal Fundus Dataset**
- **Format**: PNG fundus images with CSV labels
- **Size**: 3,200 retinal fundus images
  - Training: 1,920 images
  - Validation: 640 images  
  - Testing: 640 images
- **Resolution**: High-resolution fundus photographs
- **Labels**: Multi-label annotations for 29 disease categories

### **Retinal Disease Categories (29 total)**
| Disease | Full Name | Cases | % |
|---------|-----------|-------|---|
| DR | Diabetic Retinopathy | 376 | 19.6% |
| ARMD | Age-related Macular Degeneration | 100 | 5.2% |
| MH | Macular Hole | 317 | 16.5% |
| MYA | Myopia | 101 | 5.3% |
| BRVO | Branch Retinal Vein Occlusion | 73 | 3.8% |
| TSLN | Tessellated Fundus | 186 | 9.7% |
| ODC | Optic Disc Cupping | 282 | 14.7% |
| *...and 22 other retinal conditions* | | | |

---

## 🏗️ CNN Architecture Comparison

### **Models Evaluated**
1. **ResNet50** - Residual Network with 50 layers
2. **DenseNet121** - Densely Connected Convolutional Network
3. **InceptionV3** - Inception Network Version 3
4. **EfficientNet-B3** - EfficientNet with compound scaling

### **Training Configuration**
- **Batch Size**: 16
- **Epochs**: 20 (with early stopping)
- **Learning Rate**: 1e-4
- **Optimizer**: Adam with differential learning rates
- **Loss Function**: BCEWithLogitsLoss with class weighting
- **Data Augmentation**: Random horizontal flipping
- **Early Stopping**: Patience of 5 epochs

---

## 📈 Results and Performance

### **CNN Architecture Performance Comparison**

| Rank | Architecture | AUC (%) | Precision (%) | Recall (%) | F1-Score |
|------|--------------|---------|---------------|------------|----------|
| 🥇 | **ResNet50** | **95.42** | **97.96** | **85.57** | **0.947** |
| 🥈 | DenseNet121 | 89.38 | 96.45 | 75.10 | 0.909 |
| 🥉 | InceptionV3 | 85.82 | 94.29 | 75.10 | 0.892 |
| 4th | EfficientNet-B3 | 84.40 | 93.18 | 72.92 | 0.888 |

### **Key Findings**
- **ResNet50** achieves the best overall performance across all metrics
- **Statistical Significance**: Delong and McNemar tests confirm ResNet50 is significantly better than all other architectures (p < 0.001)
- **Clinical Relevance**: High precision (97.96%) reduces false positives in clinical screening
- **Balanced Performance**: Good recall (85.57%) ensures most diseases are detected

---

## 🔬 Statistical Analysis

### **Delong Test Results**
- **Purpose**: Compare AUC differences between architectures
- **Findings**: ResNet50 significantly outperforms all other models
- **Significance Level**: p < 0.001 for all comparisons

### **McNemar Test Results**  
- **Purpose**: Test classification accuracy differences
- **Findings**: Significant accuracy improvements with ResNet50
- **Clinical Impact**: Validates superior diagnostic performance

---

## 📁 Project Structure

```
TI-RADS-NeuralNET/
├── data/
│   └── RFMiD_Challenge_Dataset/          # Retinal fundus dataset
│       ├── 1. Original Images/           # Training, validation, test images
│       └── 2. Groundtruths/              # CSV labels for all splits
├── src/
│   ├── train_cnn.py                      # Multi-CNN training script
│   └── visualize_data.py                 # Results visualization
├── results/
│   └── CNN/                              # CNN model results
│       ├── ResNet50/                     # Best performing model
│       ├── Densenet121/                  # Second best
│       ├── InceptionV3/                  # Third best  
│       └── EfficientNetB3/               # Fourth best
└── statistical_tests/                    # Statistical analysis results
```

---

## 📊 Output Files

### **Per-Model Results**
Each CNN architecture generates:
- **Model Checkpoint**: Best performing model (`.pth`)
- **Training Curves**: Loss and accuracy progression
- **Metrics CSV**: Detailed per-epoch performance
- **Test Results**: Final evaluation metrics
- **Statistical Data**: NPZ files for statistical tests

### **Visualization Outputs**
- Overall performance comparison charts
- Per-disease sensitivity/specificity plots
- Top performing disease analysis
- Performance distribution histograms

---

## ⚙️ Configuration

### **Reproducibility**
- Python: `random.seed(42)`
- NumPy: `np.random.seed(42)`
- PyTorch: `torch.manual_seed(42)`
- CUDA: Deterministic operations enabled

### **Hardware Requirements**
- **GPU**: Recommended for CNN training (CUDA support)
- **Memory**: 8GB+ RAM recommended
- **Storage**: ~2GB for dataset
- **CPU**: Multi-core recommended for data loading

---

## 🚀 Quick Start

1. **Setup**:
   ```bash
   git clone https://github.com/Durjoy001/TI-RADS-NeuralNET
   cd TI-RADS-NeuralNET
   pip install -r requirements.txt
   ```

2. **Train CNN Models**:
   ```bash
   python src/train_cnn.py
   ```

3. **View Results**:
   ```bash
   python src/visualize_data.py
   ```

---

## 📝 Clinical Applications

This project provides:
- **Automated Retinal Disease Screening**: Multi-disease detection from fundus images
- **Architecture Benchmarking**: Evidence-based model selection for clinical deployment
- **Statistical Validation**: Rigorous performance comparison with clinical significance
- **Research Foundation**: Comprehensive framework for retinal disease classification research

---

## 🧪 Reproducibility

To ensure reproducibility, the following seeds are fixed:

- Python `random.seed(42)`
- NumPy `np.random.seed(42)`
- PyTorch `torch.manual_seed(42)`
