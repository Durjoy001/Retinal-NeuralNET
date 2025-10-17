# src/train_cnn.py
# DenseNet121 training script for RFMiD retinal fundus multi-disease classification
# Multi-label classification with 45 disease classes using ImageNet-pretrained DenseNet121
# ✅ Now includes batch-wise Sensitivity and Specificity in progress bar

import os
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
import random, warnings, numpy as np, pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.exceptions import UndefinedMetricWarning
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------
# Hard-coded parameters
# ----------------------
SEED, IMAGE_SIZE, BATCH_SIZE, EPOCHS, LR, NUM_WORKERS = 42, 224, 16, 20, 1e-4, 0
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ----------------------
# Paths (shortened)
# ----------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "RFMiD_All_Classes_Dataset"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PATH = RESULTS_DIR / "densenet121_rfmid_best.pth"
METRICS_CSV = RESULTS_DIR / "densenet121_metrics.csv"

# ----------------------
# Reproducibility
# ----------------------
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# ----------------------
# Dataset
# ----------------------
class RFMiDDataset(Dataset):
    def __init__(self, img_dir, labels_df, transform=None):
        self.img_dir, self.labels_df, self.transform = img_dir, labels_df, transform
        self.label_columns = [c for c in labels_df.columns if c != 'ID']
        self.num_classes = len(self.label_columns)

    def __len__(self): return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]; img_id = row['ID']
        img_path = self.img_dir / f"{img_id}.png"
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), color='black')
        if self.transform: image = self.transform(image)
        labels = torch.tensor(row[self.label_columns].values, dtype=torch.float32)
        return image, labels

# ----------------------
# DenseNet Model
# ----------------------
class RFMiDDenseNet121(nn.Module):
    def __init__(self, num_classes=45):
        super().__init__()
        self.backbone = models.densenet121(pretrained=True)
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.backbone(x)

# ----------------------
# Metrics helper
# ----------------------
def compute_sens_spec(preds: torch.Tensor, labels: torch.Tensor):
    """
    preds, labels: [batch, num_classes] binary tensors
    Returns sensitivity (recall for positives) and specificity (recall for negatives)
    """
    tp = (preds * labels).sum(dim=0)
    tn = ((1 - preds) * (1 - labels)).sum(dim=0)
    fp = (preds * (1 - labels)).sum(dim=0)
    fn = ((1 - preds) * labels).sum(dim=0)
    sens = (tp / (tp + fn + 1e-6)).mean().item()
    spec = (tn / (tn + fp + 1e-6)).mean().item()
    return sens, spec

# ----------------------
# Training Function with live metrics
# ----------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, total, acc_sum = 0.0, 0, 0.0
    sens_sum, spec_sum, n_batches = 0.0, 0.0, 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward(); optimizer.step()

        running_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()

        # Accuracy, sensitivity, specificity
        sample_acc = (preds == labels).float().mean(dim=1)
        acc_sum += sample_acc.sum().item()
        total += labels.size(0)
        sens, spec = compute_sens_spec(preds, labels)
        sens_sum += sens; spec_sum += spec; n_batches += 1

        # Show in tqdm bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{sample_acc.mean().item():.4f}",
            "sens": f"{sens:.4f}",
            "spec": f"{spec:.4f}"
        })

    return (running_loss / len(loader),
            acc_sum / total,
            sens_sum / n_batches,
            spec_sum / n_batches)

# ----------------------
# Evaluation Function (same metrics)
# ----------------------
@torch.no_grad()
def evaluate_model(model, loader, criterion, device):
    model.eval()
    running_loss, total, acc_sum = 0.0, 0, 0.0
    sens_sum, spec_sum, n_batches = 0.0, 0.0, 0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        preds = (torch.sigmoid(outputs) > 0.5).float()
        sample_acc = (preds == labels).float().mean(dim=1)
        acc_sum += sample_acc.sum().item()
        total += labels.size(0)
        sens, spec = compute_sens_spec(preds, labels)
        sens_sum += sens; spec_sum += spec; n_batches += 1

        all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    try:
        auc_score = roc_auc_score(all_labels, all_preds, average='macro')
    except Exception:
        auc_score = 0.0

    return (running_loss / len(loader),
            acc_sum / total,
            sens_sum / n_batches,
            spec_sum / n_batches,
            auc_score)

# ----------------------
# Main Training Loop (shortened)
# ----------------------
def main():
    print("🚀 Starting DenseNet121 training with Sens/Spec tracking...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_labels = pd.read_csv(DATA_DIR / "2. Groundtruths" / "a. RFMiD_Training_Labels.csv")
    val_labels = pd.read_csv(DATA_DIR / "2. Groundtruths" / "b. RFMiD_Validation_Labels.csv")

    train_dataset = RFMiDDataset(
        DATA_DIR / "1. Original Images" / "a. Training Set", train_labels,
        transform=transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    )
    val_dataset = RFMiDDataset(
        DATA_DIR / "1. Original Images" / "b. Validation Set", val_labels,
        transform=transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = RFMiDDenseNet121(num_classes=train_dataset.num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    with open(METRICS_CSV, "w") as f:
        f.write("epoch,train_loss,train_acc,train_sens,train_spec,val_loss,val_acc,val_sens,val_spec,val_auc\n")

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss, train_acc, train_sens, train_spec = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_sens, val_spec, val_auc = evaluate_model(model, val_loader, criterion, device)
        print(f"Train Acc: {train_acc:.4f}, Sens: {train_sens:.4f}, Spec: {train_spec:.4f}")
        print(f"Val Acc: {val_acc:.4f}, Sens: {val_sens:.4f}, Spec: {val_spec:.4f}, AUC: {val_auc:.4f}")
        with open(METRICS_CSV, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{train_acc:.6f},{train_sens:.6f},{train_spec:.6f},"
                    f"{val_loss:.6f},{val_acc:.6f},{val_sens:.6f},{val_spec:.6f},{val_auc:.6f}\n")

        if val_acc > best_val_acc:
            torch.save(model.state_dict(), SAVE_PATH)
            best_val_acc = val_acc
            print("💾 Best model saved!")

    print("🎉 Training complete!")

if __name__ == "__main__":
    main()
