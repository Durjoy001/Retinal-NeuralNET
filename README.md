# TI-RADS-NeuralNET

This repository contains the implementation of a Vision Transformer (ViT)-based neural network for classifying thyroid nodules according to the TI-RADS composition categories. The project mirrors the research described in the following article:

[**"Research Article on Thyroid Nodule Classification"**](https://www.sciencedirect.com/science/article/pii/S1877050924031235)

---

## 📖 Purpose

Thyroid nodules are a common medical condition, and their classification is critical for determining the appropriate course of treatment. This project leverages Vision Transformers (ViTs) to classify thyroid nodules into TI-RADS composition categories based on ultrasound images. The dataset and metadata used in this project are provided in HDF5 format and are automatically downloaded when the project is run.

---

## 🛠️ Setup Instructions
1. Clone the Repository
```
git clone https://github.com/Durjoy001/TI-RADS-NeuralNET

cd TI-RADS-NeuralNET
```
2. Install Dependencies

```
pip install -r requirements.txt
```
3. Run the Project

```
python main.py
```

# 📥 Dataset and Metadata
The dataset and metadata are automatically downloaded to the data/ directory when you run the project. The files include:

- `thyroid_dataset.h5`: Contains ultrasound images of thyroid nodules.
- `metadata.csv`: Contains labels and additional information for the dataset.

# 📊 Results
The training process generates the following outputs in the results/ directory:

- Best Model Checkpoint: `vit_tiny_patch16_224_composition_best.pth`
- Training Curves: `training_curves.png`
- Confusion Matrix: `confusion_matrix.png`
- Metrics CSV: `metrics.csv`

# ⚙️ Parameters
The training script uses the following hard-coded parameters:

- Model: vit_tiny_patch16_224 (pretrained on ImageNet)
- Image Size: 224x224
- Batch Size: 8
- Epochs: 3
- Learning Rate: 3e-5
- Test Fraction: 20%

You can modify these parameters in `src/train_vit.py.`

# 🧪 Reproducibility
To ensure reproducibility, the following seeds are fixed:

- Python `random.seed(42)`
- NumPy `np.random.seed(42)`
- PyTorch `torch.manual_seed(42)`