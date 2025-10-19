# src/train_cnn.py
# DenseNet121 training script for RFMiD retinal fundus multi-disease classification
# Multi-label classification with 45 disease classes using ImageNet-pretrained DenseNet121
# ✅ Includes Sensitivity/Specificity masking, per-class threshold calibration, balanced accuracy, and AUC-based model saving

import os
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
import random, warnings, numpy as np, pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import DenseNet121_Weights
from sklearn.metrics import roc_auc_score, roc_curve
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
# Paths
# ----------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "RFMiD_All_Classes_Dataset"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PATH = RESULTS_DIR / "densenet121_rfmid_best.pth"
METRICS_CSV = RESULTS_DIR / "densenet121_metrics.csv"
THRESHOLDS_PATH = RESULTS_DIR / "optimal_thresholds.npy"

# ----------------------
# Reproducibility
# ----------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
        row = self.labels_df.iloc[idx]
        img_id = row['ID']
        img_path = self.img_dir / f"{img_id}.png"
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), color='black')
        if self.transform:
            image = self.transform(image)
        labels = torch.tensor(row[self.label_columns].values, dtype=torch.float32)
        return image, labels

# ----------------------
# DenseNet Model
# ----------------------
class RFMiDDenseNet121(nn.Module):
    def __init__(self, num_classes=45):
        super().__init__()
        self.backbone = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# ----------------------
# Metric Helper
# ----------------------
def compute_sens_spec(preds: torch.Tensor, labels: torch.Tensor, average="macro"):
    """Compute sensitivity and specificity with masking for valid classes"""
    tp = (preds * labels).sum(dim=0).float()
    tn = ((1 - preds) * (1 - labels)).sum(dim=0).float()
    fp = (preds * (1 - labels)).sum(dim=0).float()
    fn = ((1 - preds) * labels).sum(dim=0).float()

    if average == "micro":
        TP, TN, FP, FN = tp.sum(), tn.sum(), fp.sum(), fn.sum()
        sens = (TP / (TP + FN + 1e-6)).item()
        spec = (TN / (TN + FP + 1e-6)).item()
        return sens, spec

    pos_mask = (tp + fn) > 0
    neg_mask = (tn + fp) > 0
    sens = (tp[pos_mask] / (tp[pos_mask] + fn[pos_mask])).mean().item() if pos_mask.any() else float("nan")
    spec = (tn[neg_mask] / (tn[neg_mask] + fp[neg_mask])).mean().item() if neg_mask.any() else float("nan")
    return sens, spec

def compute_per_class_sens_spec(preds: torch.Tensor, labels: torch.Tensor):
    """Compute per-class sensitivity and specificity"""
    tp = (preds * labels).sum(dim=0).float()
    tn = ((1 - preds) * (1 - labels)).sum(dim=0).float()
    fp = (preds * (1 - labels)).sum(dim=0).float()
    fn = ((1 - preds) * labels).sum(dim=0).float()
    
    # Per-class sensitivity and specificity
    sens_per_class = tp / (tp + fn + 1e-6)  # Add small epsilon to avoid division by zero
    spec_per_class = tn / (tn + fp + 1e-6)
    
    return sens_per_class.cpu().numpy(), spec_per_class.cpu().numpy()

# ----------------------
# Threshold Calibration
# ----------------------
def compute_optimal_thresholds(y_true, y_pred, target_spec=0.8):
    """Find per-class thresholds that achieve target specificity (e.g. 80%)"""
    thresholds = []
    for i in range(y_true.shape[1]):
        try:
            fpr, tpr, thr = roc_curve(y_true[:, i], y_pred[:, i])
            spec = 1 - fpr
            idx = np.argmin(np.abs(spec - target_spec))
            thresholds.append(thr[idx])
        except Exception:
            thresholds.append(0.5)
    return np.array(thresholds)

# ----------------------
# Training Loop
# ----------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch with correct epoch-level sensitivity/specificity."""
    model.train()
    running_loss = 0.0

    # Initialize total counters
    TP = TN = FP = FN = None

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Convert outputs to binary predictions
        preds = (torch.sigmoid(outputs) > 0.5).float()

        # Compute counts per class
        tp = (preds * labels).sum(dim=0)
        tn = ((1 - preds) * (1 - labels)).sum(dim=0)
        fp = (preds * (1 - labels)).sum(dim=0)
        fn = ((1 - preds) * labels).sum(dim=0)

        # Accumulate totals
        if TP is None:
            TP, TN, FP, FN = tp, tn, fp, fn
        else:
            TP += tp; TN += tn; FP += fp; FN += fn

    # Compute epoch-level metrics
    sens = (TP.sum() / (TP.sum() + FN.sum() + 1e-6)).item()
    spec = (TN.sum() / (TN.sum() + FP.sum() + 1e-6)).item()
    bal_acc = 0.5 * (sens + spec)

    avg_loss = running_loss / len(loader)
    return avg_loss, bal_acc, sens, spec

# ----------------------
# Evaluation Loop
# ----------------------
@torch.no_grad()
def evaluate_model(model, loader, criterion, device, thresholds=None):
    """Evaluate with global epoch-level sensitivity/specificity and per-class metrics."""
    model.eval()
    running_loss = 0.0
    TP = TN = FP = FN = None
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="Evaluating", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        probs = torch.sigmoid(outputs).cpu().numpy()
        labels_np = labels.cpu().numpy()

        if thresholds is None:
            preds = (probs > 0.5).astype(float)
        else:
            preds = (probs > thresholds).astype(float)

        preds_t = torch.tensor(preds)
        labels_t = torch.tensor(labels_np)

        # Compute counts and accumulate
        tp = (preds_t * labels_t).sum(dim=0)
        tn = ((1 - preds_t) * (1 - labels_t)).sum(dim=0)
        fp = (preds_t * (1 - labels_t)).sum(dim=0)
        fn = ((1 - preds_t) * labels_t).sum(dim=0)

        if TP is None:
            TP, TN, FP, FN = tp, tn, fp, fn
        else:
            TP += tp; TN += tn; FP += fp; FN += fn

        all_preds.append(probs)
        all_labels.append(labels_np)

    # Convert accumulated data
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Compute global metrics
    sens = (TP.sum() / (TP.sum() + FN.sum() + 1e-6)).item()
    spec = (TN.sum() / (TN.sum() + FP.sum() + 1e-6)).item()
    bal_acc = 0.5 * (sens + spec)

    # Per-class sensitivity/specificity
    sens_per_class = (TP / (TP + FN + 1e-6)).cpu().numpy()
    spec_per_class = (TN / (TN + FP + 1e-6)).cpu().numpy()

    # AUC (masked to valid classes)
    try:
        valid_cols = (np.sum(all_labels, axis=0) > 0) & (np.sum(all_labels == 0, axis=0) > 0)
        if np.any(valid_cols):
            auc_score = roc_auc_score(all_labels[:, valid_cols], all_preds[:, valid_cols], average='macro')
        else:
            auc_score = 0.0
    except Exception:
        auc_score = 0.0

    avg_loss = running_loss / len(loader)
    return avg_loss, bal_acc, sens, spec, auc_score, all_labels, all_preds, sens_per_class, spec_per_class


# ----------------------
# Plotting
# ----------------------
def plot_training_curves(train_losses, train_accs, val_losses, val_accs, out_path):
    try:
        epochs = np.arange(0, len(train_losses))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(epochs, train_losses, label="Train Loss", color='blue')
        ax1.plot(epochs, val_losses, label="Val Loss", color='red')
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)
        ax1.set_title("Training and Validation Loss")
        ax2.plot(epochs, train_accs, label="Train Balanced Acc", color='blue')
        ax2.plot(epochs, val_accs, label="Val Balanced Acc", color='red')
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Balanced Accuracy"); ax2.legend(); ax2.grid(True, alpha=0.3)
        ax2.set_title("Training and Validation Balanced Accuracy")
        plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()
    except Exception as e:
        print(f"[WARN] Failed to plot training curves: {e}")

def plot_loss_curves(train_losses, val_losses, out_path):
    """Plot training and validation loss curves"""
    try:
        epochs = np.arange(0, len(train_losses))
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, label="Training Loss", color='blue', linewidth=2, marker='o', markersize=4)
        plt.plot(epochs, val_losses, label="Validation Loss", color='red', linewidth=2, marker='s', markersize=4)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Training and Validation Loss Over Time", fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ Loss curves saved to: {out_path}")
    except Exception as e:
        print(f"[WARN] Failed to plot loss curves: {e}")

def plot_sensitivity_specificity_curves(train_sens, train_spec, val_sens, val_spec, out_path):
    """Plot training and validation sensitivity and specificity curves"""
    try:
        epochs = np.arange(0, len(train_sens))
        plt.figure(figsize=(10, 6))
        
        # Plot training curves
        plt.plot(epochs, train_sens, label="Training Sensitivity", color='blue', linewidth=2, marker='o', markersize=4)
        plt.plot(epochs, train_spec, label="Training Specificity", color='lightblue', linewidth=2, marker='^', markersize=4)
        
        # Plot validation curves
        plt.plot(epochs, val_sens, label="Validation Sensitivity", color='red', linewidth=2, marker='s', markersize=4)
        plt.plot(epochs, val_spec, label="Validation Specificity", color='lightcoral', linewidth=2, marker='d', markersize=4)
        
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Sensitivity / Specificity", fontsize=12)
        plt.title("Training and Validation Sensitivity & Specificity Over Time", fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)  # Set y-axis from 0 to 1 for better visualization
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ Sensitivity/Specificity curves saved to: {out_path}")
    except Exception as e:
        print(f"[WARN] Failed to plot sensitivity/specificity curves: {e}")

# ----------------------
# Main
# ----------------------
def main():
    print("🚀 Starting DenseNet121 training with Sens/Spec tracking + AUC threshold calibration")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_labels = pd.read_csv(DATA_DIR / "2. Groundtruths" / "a. RFMiD_Training_Labels.csv")
    val_labels = pd.read_csv(DATA_DIR / "2. Groundtruths" / "b. RFMiD_Validation_Labels.csv")
    test_labels = pd.read_csv(DATA_DIR / "2. Groundtruths" / "c. RFMiD_Testing_Labels.csv")

    print(f"Training samples: {len(train_labels)}, Validation: {len(val_labels)}, Test: {len(test_labels)}")

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_dataset = RFMiDDataset(DATA_DIR / "1. Original Images" / "a. Training Set", train_labels, train_transform)
    val_dataset = RFMiDDataset(DATA_DIR / "1. Original Images" / "b. Validation Set", val_labels, val_test_transform)
    test_dataset = RFMiDDataset(DATA_DIR / "1. Original Images" / "c. Testing Set", test_labels, val_test_transform)

    train_loader = DataLoader(train_dataset, BATCH_SIZE, True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, False, num_workers=NUM_WORKERS)

    model = RFMiDDenseNet121(num_classes=train_dataset.num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()

    backbone_params = list(model.backbone.features.parameters())
    classifier_params = list(model.backbone.classifier.parameters())
    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': LR * 0.1},
        {'params': classifier_params, 'lr': LR}
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # Create CSV header with per-class columns
    class_names = train_dataset.label_columns
    header = "epoch,train_loss,train_bal_acc,train_sens,train_spec,val_loss,val_bal_acc,val_sens,val_spec,val_auc"
    
    # Add per-class validation sensitivity and specificity columns
    for i, class_name in enumerate(class_names):
        header += f",val_sens_{class_name},val_spec_{class_name}"
    
    with open(METRICS_CSV, "w") as f:
        f.write(header + "\n")

    best_val_auc = 0.0
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    train_sens_list, train_spec_list, val_sens_list, val_spec_list = [], [], [], []

    # ----------------------
    # Epoch 0: Initial model performance (before training)
    # ----------------------
    print("\n📊 Epoch 0: Evaluating initial model performance...")
    # Only evaluate, don't train for epoch 0
    train_loss_0, train_bal_acc_0, train_sens_0, train_spec_0, _, _, _, _, _ = evaluate_model(model, train_loader, criterion, device)
    val_loss_0, val_bal_acc_0, val_sens_0, val_spec_0, val_auc_0, _, _, val_sens_per_class_0, val_spec_per_class_0 = evaluate_model(model, val_loader, criterion, device)
    
    # Store initial metrics
    train_losses.append(train_loss_0); val_losses.append(val_loss_0)
    train_accs.append(train_bal_acc_0); val_accs.append(val_bal_acc_0)
    train_sens_list.append(train_sens_0); train_spec_list.append(train_spec_0)
    val_sens_list.append(val_sens_0); val_spec_list.append(val_spec_0)
    
    print(f"Initial Train Balanced Acc: {train_bal_acc_0:.4f} | Sens: {train_sens_0:.4f} | Spec: {train_spec_0:.4f}")
    print(f"Initial Val Balanced Acc: {val_bal_acc_0:.4f} | Sens: {val_sens_0:.4f} | Spec: {val_spec_0:.4f} | AUC: {val_auc_0:.4f}")
    
    # Write initial metrics to CSV
    csv_line = f"0,{train_loss_0:.6f},{train_bal_acc_0:.6f},{train_sens_0:.6f},{train_spec_0:.6f},"
    csv_line += f"{val_loss_0:.6f},{val_bal_acc_0:.6f},{val_sens_0:.6f},{val_spec_0:.6f},{val_auc_0:.6f}"
    
    # Add per-class validation metrics for epoch 0
    for i in range(len(class_names)):
        csv_line += f",{val_sens_per_class_0[i]:.6f},{val_spec_per_class_0[i]:.6f}"
    
    with open(METRICS_CSV, "a") as f:
        f.write(csv_line + "\n")

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss, train_bal_acc, train_sens, train_spec = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_bal_acc, val_sens, val_spec, val_auc, _, _, val_sens_per_class, val_spec_per_class = evaluate_model(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        train_losses.append(train_loss); val_losses.append(val_loss)
        train_accs.append(train_bal_acc); val_accs.append(val_bal_acc)
        train_sens_list.append(train_sens); train_spec_list.append(train_spec)
        val_sens_list.append(val_sens); val_spec_list.append(val_spec)

        print(f"Train Balanced Acc: {train_bal_acc:.4f} | Sens: {train_sens:.4f} | Spec: {train_spec:.4f}")
        print(f"Val Balanced Acc: {val_bal_acc:.4f} | Sens: {val_sens:.4f} | Spec: {val_spec:.4f} | AUC: {val_auc:.4f}")

        # Write comprehensive metrics to CSV
        csv_line = f"{epoch},{train_loss:.6f},{train_bal_acc:.6f},{train_sens:.6f},{train_spec:.6f},"
        csv_line += f"{val_loss:.6f},{val_bal_acc:.6f},{val_sens:.6f},{val_spec:.6f},{val_auc:.6f}"
        
        # Add per-class validation metrics
        for i in range(len(class_names)):
            csv_line += f",{val_sens_per_class[i]:.6f},{val_spec_per_class[i]:.6f}"
        
        with open(METRICS_CSV, "a") as f:
            f.write(csv_line + "\n")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({'model_state_dict': model.state_dict()}, SAVE_PATH)
            print(f"💾 Best model saved! (AUC={val_auc:.4f})")

        # Generate separate curve plots
        plot_training_curves(train_losses, train_accs, val_losses, val_accs, RESULTS_DIR / "training_curves.png")
        plot_loss_curves(train_losses, val_losses, RESULTS_DIR / "loss_curves.png")
        plot_sensitivity_specificity_curves(train_sens_list, train_spec_list, val_sens_list, val_spec_list, RESULTS_DIR / "sensitivity_specificity_curves.png")

    # ----------------------
    # Threshold calibration
    # ----------------------
    print("\n📊 Calibrating thresholds (target specificity=0.8)...")

    # ✅ Safety check: ensure the model file exists
    if os.path.exists(SAVE_PATH):
        checkpoint = torch.load(SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Loaded best saved model for calibration.")
    else:
        print("⚠️ No best model saved yet (AUC=nan or interrupted training). Using last trained model instead.")

    # Proceed with calibration using whichever model is loaded
    _, _, _, _, val_auc, y_true_val, y_pred_val, _, _ = evaluate_model(model, val_loader, criterion, device)
    thresholds = compute_optimal_thresholds(np.array(y_true_val), np.array(y_pred_val), target_spec=0.8)
    np.save(THRESHOLDS_PATH, thresholds)
    print(f"Optimal thresholds saved to: {THRESHOLDS_PATH}")


    # ----------------------
    # Final test evaluation
    # ----------------------
    print("\n🧪 Final evaluation on test set (using calibrated thresholds)...")
    test_loss, test_bal_acc, test_sens, test_spec, test_auc, _, _, test_sens_per_class, test_spec_per_class = evaluate_model(model, test_loader, criterion, device, thresholds)
    print(f"Test Balanced Acc: {test_bal_acc:.4f} | Sens: {test_sens:.4f} | Spec: {test_spec:.4f} | AUC: {test_auc:.4f}")
    
    # Save final test results with per-class metrics
    test_results_csv = RESULTS_DIR / "final_test_results.csv"
    with open(test_results_csv, "w") as f:
        f.write("class_name,test_sensitivity,test_specificity\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name},{test_sens_per_class[i]:.6f},{test_spec_per_class[i]:.6f}\n")
    
    # Save overall final test results
    overall_results_csv = RESULTS_DIR / "overall_test_results.csv"
    with open(overall_results_csv, "w") as f:
        f.write("metric,value\n")
        f.write(f"test_loss,{test_loss:.6f}\n")
        f.write(f"test_balanced_accuracy,{test_bal_acc:.6f}\n")
        f.write(f"test_sensitivity,{test_sens:.6f}\n")
        f.write(f"test_specificity,{test_spec:.6f}\n")
        f.write(f"test_auc,{test_auc:.6f}\n")
        f.write(f"best_validation_auc,{best_val_auc:.6f}\n")
    
    print(f"\n🎉 Training completed!")
    print(f"Best validation AUC: {best_val_auc:.4f}")
    print(f"Final test results:")
    print(f"  - Balanced Accuracy: {test_bal_acc:.4f}")
    print(f"  - Sensitivity: {test_sens:.4f}")
    print(f"  - Specificity: {test_spec:.4f}")
    print(f"  - AUC: {test_auc:.4f}")
    print(f"Model saved to: {SAVE_PATH}")
    print(f"Thresholds saved to: {THRESHOLDS_PATH}")
    print(f"Training metrics saved to: {METRICS_CSV}")
    print(f"Final test per-class results saved to: {test_results_csv}")
    print(f"Overall test results saved to: {overall_results_csv}")

if __name__ == "__main__":
    main()
