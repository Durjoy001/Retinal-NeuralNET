# -*- coding: utf-8 -*-
"""
Binary CNN training for RFMiD:
- Single diseases_risk head: 1 means any disease, 0 means normal
- Hemelings style preprocessing: center crop, resize to 224, Gaussian background subtraction,
  horizontal flip, brightness shift, gentle elastic deformation
- ImageNet pretrained CNNs (ResNet50, EfficientNet-B3, InceptionV3, DenseNet121)
- Freeze all early layers and train only last N parameter tensors
- Adam optimizer, BCEWithLogitsLoss, validation AUC checkpointing
- Threshold calibration for target specificity ~0.80
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path
import random
import warnings
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import (
    DenseNet121_Weights,
    ResNet50_Weights,
    EfficientNet_B3_Weights,
    Inception_V3_Weights
)

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix
)
from sklearn.exceptions import UndefinedMetricWarning

from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------
# Hard coded parameters
# ----------------------
SEED = 42
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
NUM_WORKERS = 0       # 0 is safe on macOS
PATIENCE = 2          # stop if val balanced accuracy does not improve for 2 epochs
MIN_DELTA = 1e-4      # minimum improvement to be considered better
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ----------------------
# Paths
# ----------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "RFMiD_Challenge_Dataset"

RESULTS_DIR = None
SAVE_PATH = None
METRICS_CSV = None
THRESHOLDS_PATH = None

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
# Dataset for binary diseases_risk label
# ----------------------
class RFMiDDatasetBinary(Dataset):
    def __init__(self, img_dir, labels_df, label_column, transform=None):
        self.img_dir = img_dir
        self.labels_df = labels_df
        self.label_column = label_column
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_id = row["ID"]
        img_path = self.img_dir / f"{img_id}.png"

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        label = float(row[self.label_column])
        label = torch.tensor(label, dtype=torch.float32)
        return image, label


# ----------------------
# Hemelings style preprocessing
# ----------------------
class BackgroundSubtraction(object):
    """
    Approximate background subtraction:
    - Blur image with Gaussian filter to get background
    - Subtract background, rescale to [0, 255]
    """
    def __init__(self, kernel_size=31):
        self.blur = transforms.GaussianBlur(kernel_size=kernel_size)

    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            raise TypeError("BackgroundSubtraction expects a PIL Image")

        blurred = self.blur(img)

        img_np = np.asarray(img).astype(np.float32)
        bg_np = np.asarray(blurred).astype(np.float32)

        sub = img_np - bg_np
        sub = sub - sub.min()
        max_val = sub.max()
        if max_val > 0:
            sub = sub / max_val * 255.0
        sub = np.clip(sub, 0, 255).astype(np.uint8)
        return Image.fromarray(sub)


class IdentityTransform(object):
    def __call__(self, img):
        return img


def get_elastic_transform():
    # Use torchvision ElasticTransform if available
    if hasattr(transforms, "ElasticTransform"):
        return transforms.ElasticTransform(alpha=50.0, sigma=5.0)
    return IdentityTransform()


def get_transforms(model_name: str, train: bool):
    """
    Preprocessing for all CNNs:

    1) Center crop to 1016 x 1016
    2) Resize to 224 x 224
    3) Gaussian background subtraction
    4) For training: horizontal flip, brightness jitter, elastic deformation
    5) ToTensor + ImageNet normalization
    """
    base_transforms = [
        transforms.CenterCrop(1016),
        transforms.Resize((224, 224)),
        BackgroundSubtraction(kernel_size=31),
    ]

    aug_transforms = []
    if train:
        aug_transforms.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2),
            get_elastic_transform(),
        ])

    to_tensor_and_norm = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]

    return transforms.Compose(base_transforms + aug_transforms + to_tensor_and_norm)


# ----------------------
# Model builders with binary head (diseases_risk)
# ----------------------
def build_model(model_name: str):
    """
    Build a CNN with a single binary output:
    diseases_risk: 1 means any disease present, 0 means normal.
    """
    model_name = model_name.lower()

    if model_name == "densenet121":
        model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        in_f = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_f, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    elif model_name == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        in_f = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_f, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    elif model_name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_f, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    elif model_name == "inception_v3":
        model = models.inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1,
            aux_logits=True
        )
        in_f = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_f, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        # Aux head for training
        if model.AuxLogits is not None:
            aux_in = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(aux_in, 1)

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model


# ----------------------
# Freezing strategy
# ----------------------
def freeze_early_layers_hemelings_style(model: nn.Module, num_trainable_layers: int = 12):
    """
    Freeze all parameters except the last num_trainable_layers parameter tensors.
    """
    params = list(model.parameters())
    total = len(params)
    if num_trainable_layers >= total:
        return

    for p in params:
        p.requires_grad = False
    for p in params[total - num_trainable_layers:]:
        p.requires_grad = True


# ----------------------
# Metric helpers for binary classification
# ----------------------
def compute_confusion_from_preds(y_true: np.ndarray, y_pred_bin: np.ndarray):
    tn, fp, fn, tp = confusion_matrix(
        y_true.astype(int),
        y_pred_bin.astype(int),
        labels=[0, 1]
    ).ravel()
    return tp, tn, fp, fn


def compute_sens_spec_acc(tp, tn, fp, fn):
    sens = tp / (tp + fn + 1e-6)
    spec = tn / (tn + fp + 1e-6)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    bal_acc = 0.5 * (sens + spec)
    return sens, spec, acc, bal_acc


# ----------------------
# Training and evaluation
# ----------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    TP = TN = FP = FN = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        if isinstance(outputs, tuple):
            main_out, aux_out = outputs
            main_logits = main_out.view(-1)
            aux_logits = aux_out.view(-1)
            labels_flat = labels.view(-1)
            loss = criterion(main_logits, labels_flat) + 0.4 * criterion(aux_logits, labels_flat)
            logits_for_metrics = main_logits
        else:
            logits = outputs.view(-1)
            labels_flat = labels.view(-1)
            loss = criterion(logits, labels_flat)
            logits_for_metrics = logits

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        probs = torch.sigmoid(logits_for_metrics)
        preds = (probs >= 0.5).float()

        tp = ((preds == 1) & (labels_flat == 1)).sum().item()
        tn = ((preds == 0) & (labels_flat == 0)).sum().item()
        fp = ((preds == 1) & (labels_flat == 0)).sum().item()
        fn = ((preds == 0) & (labels_flat == 1)).sum().item()

        TP += tp
        TN += tn
        FP += fp
        FN += fn

    sens, spec, _, bal_acc = compute_sens_spec_acc(TP, TN, FP, FN)
    avg_loss = running_loss / len(loader)
    return avg_loss, bal_acc, sens, spec


@torch.no_grad()
def evaluate_model(model, loader, criterion, device, threshold: float = 0.5):
    model.eval()
    running_loss = 0.0

    TP = TN = FP = FN = 0
    all_probs = []
    all_labels = []

    pbar = tqdm(loader, desc="Evaluating", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        logits = outputs.view(-1)
        labels_flat = labels.view(-1)

        loss = criterion(logits, labels_flat)
        running_loss += loss.item()

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels_flat.cpu().numpy())

        tp = ((preds == 1) & (labels_flat == 1)).sum().item()
        tn = ((preds == 0) & (labels_flat == 0)).sum().item()
        fp = ((preds == 1) & (labels_flat == 0)).sum().item()
        fn = ((preds == 0) & (labels_flat == 1)).sum().item()

        TP += tp
        TN += tn
        FP += fp
        FN += fn

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    sens, spec, _, bal_acc = compute_sens_spec_acc(TP, TN, FP, FN)

    try:
        if len(np.unique(all_labels)) > 1:
            auc_score = roc_auc_score(all_labels, all_probs)
        else:
            auc_score = 0.0
    except Exception:
        auc_score = 0.0

    avg_loss = running_loss / len(loader)
    return avg_loss, bal_acc, sens, spec, auc_score, all_labels, all_probs


# ----------------------
# Plotting
# ----------------------
def plot_training_curves(train_losses, train_accs, val_losses, val_accs, out_path):
    try:
        epochs = np.arange(0, len(train_losses))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(epochs, train_losses, label="Train Loss")
        ax1.plot(epochs, val_losses, label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title("Training and Validation Loss")

        ax2.plot(epochs, train_accs, label="Train Balanced Acc")
        ax2.plot(epochs, val_accs, label="Val Balanced Acc")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Balanced Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Training and Validation Balanced Accuracy")

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"[WARN] Failed to plot training curves: {e}")


def plot_loss_curves(train_losses, val_losses, out_path):
    try:
        epochs = np.arange(0, len(train_losses))
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, label="Training Loss", linewidth=2, marker="o", markersize=4)
        plt.plot(epochs, val_losses, label="Validation Loss", linewidth=2, marker="s", markersize=4)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Training and Validation Loss Over Time", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"Saved loss curves to: {out_path}")
    except Exception as e:
        print(f"[WARN] Failed to plot loss curves: {e}")


def plot_sensitivity_specificity_curves(train_sens, train_spec, val_sens, val_spec, out_path):
    try:
        epochs = np.arange(0, len(train_sens))
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_sens, label="Training Sensitivity", linewidth=2, marker="o", markersize=4)
        plt.plot(epochs, train_spec, label="Training Specificity", linewidth=2, marker="^", markersize=4)
        plt.plot(epochs, val_sens, label="Validation Sensitivity", linewidth=2, marker="s", markersize=4)
        plt.plot(epochs, val_spec, label="Validation Specificity", linewidth=2, marker="d", markersize=4)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Sensitivity / Specificity", fontsize=12)
        plt.title("Training and Validation Sensitivity and Specificity", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"Saved sensitivity/specificity curves to: {out_path}")
    except Exception as e:
        print(f"[WARN] Failed to plot sensitivity/specificity curves: {e}")


# ----------------------
# Helpers for threshold selection
# ----------------------
def _pick_threshold_for_specificity(y_true_binary, y_score, target_spec=0.8):
    fpr, tpr, thr = roc_curve(y_true_binary, y_score)
    spec = 1.0 - fpr
    idx = np.argmin(np.abs(spec - target_spec))
    return float(thr[idx]), float(spec[idx]), float(tpr[idx])


def _compute_f1max(y_true_binary, y_score):
    precision, recall, thr = precision_recall_curve(y_true_binary, y_score)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    f1_use = f1[:-1]
    best_idx = int(np.nanargmax(f1_use))
    return float(f1_use[best_idx]), float(thr[best_idx]), float(precision[best_idx]), float(recall[best_idx])


# ----------------------
# Core training pipeline for one model
# ----------------------
def run_for_model(model_name: str):
    global RESULTS_DIR, SAVE_PATH, METRICS_CSV, THRESHOLDS_PATH

    pretty = {
        "densenet121": "Densenet121",
        "resnet50": "ResNet50",
        "efficientnet_b3": "EfficientNetB3",
        "inception_v3": "InceptionV3",
    }[model_name.lower()]

    RESULTS_DIR = ROOT_DIR / "results" / "CNN_binary" / pretty
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    SAVE_PATH = RESULTS_DIR / f"{model_name.lower()}_rfmid_binary_best.pth"
    METRICS_CSV = RESULTS_DIR / f"{model_name.lower()}_metrics.csv"
    THRESHOLDS_PATH = RESULTS_DIR / "optimal_threshold_spec80.npy"

    print(f"Starting {pretty} training (binary diseases_risk head).")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load labels
    train_labels = pd.read_csv(DATA_DIR / "2. Groundtruths" / "a. RFMiD_Training_Labels.csv")
    val_labels = pd.read_csv(DATA_DIR / "2. Groundtruths" / "b. RFMiD_Validation_Labels.csv")
    test_labels = pd.read_csv(DATA_DIR / "2. Groundtruths" / "c. RFMiD_Testing_Labels.csv")

    # Compute binary diseases_risk from disease columns only
    # CRITICAL: Exclude any existing risk columns to avoid circular logic
    all_cols = train_labels.columns.tolist()
    
    # Columns that represent individual diseases only (exclude ID and any risk columns)
    disease_cols = [
        c for c in all_cols
        if c not in ["ID", "Disease_Risk", "diseases_risk"]
    ]
    
    for df in [train_labels, val_labels, test_labels]:
        df["diseases_risk"] = (df[disease_cols].sum(axis=1) > 0).astype(int)

    print(
        f"Training samples: {len(train_labels)}, "
        f"Validation: {len(val_labels)}, Test: {len(test_labels)}"
    )

    # Class imbalance weighting based on binary diseases_risk
    y = train_labels["diseases_risk"].values.astype(np.float32)
    pos = y.sum()
    neg = y.shape[0] - pos
    device_dummy = torch.device("cpu")
    pos_weight = torch.tensor(neg / (pos + 1e-6), dtype=torch.float32).to(device_dummy)

    # Transforms
    train_transform = get_transforms(model_name, train=True)
    val_test_transform = get_transforms(model_name, train=False)

    train_dataset = RFMiDDatasetBinary(
        DATA_DIR / "1. Original Images" / "a. Training Set",
        train_labels,
        "diseases_risk",
        train_transform
    )
    val_dataset = RFMiDDatasetBinary(
        DATA_DIR / "1. Original Images" / "b. Validation Set",
        val_labels,
        "diseases_risk",
        val_test_transform
    )
    test_dataset = RFMiDDatasetBinary(
        DATA_DIR / "1. Original Images" / "c. Testing Set",
        test_labels,
        "diseases_risk",
        val_test_transform
    )

    train_loader = DataLoader(train_dataset, BATCH_SIZE, True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, False, num_workers=NUM_WORKERS)

    # Build model, freeze early layers, set up optimizer
    model = build_model(model_name)
    freeze_early_layers_hemelings_style(model, num_trainable_layers=12)
    model = model.to(device)

    pos_weight = pos_weight.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    # CSV header
    header = (
        "epoch,train_loss,train_bal_acc,train_sens,train_spec,"
        "val_loss,val_bal_acc,val_sens,val_spec,val_auc"
    )
    with open(METRICS_CSV, "w") as f:
        f.write(header + "\n")

    best_val_auc = 0.0
    best_val_acc_for_es = 0.0
    epochs_no_improve = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_sens_list, train_spec_list = [], []
    val_sens_list, val_spec_list = [], []

    # Epoch 0 initial evaluation
    print("\nEpoch 0: initial evaluation")
    train_loss_0, train_bal_acc_0, train_sens_0, train_spec_0, _, _, _ = evaluate_model(
        model, train_loader, criterion, device, threshold=0.5
    )
    val_loss_0, val_bal_acc_0, val_sens_0, val_spec_0, val_auc_0, _, _ = evaluate_model(
        model, val_loader, criterion, device, threshold=0.5
    )

    train_losses.append(train_loss_0)
    val_losses.append(val_loss_0)
    train_accs.append(train_bal_acc_0)
    val_accs.append(val_bal_acc_0)
    train_sens_list.append(train_sens_0)
    train_spec_list.append(train_spec_0)
    val_sens_list.append(val_sens_0)
    val_spec_list.append(val_spec_0)

    print(
        f"Initial Train BalAcc: {train_bal_acc_0:.4f} | "
        f"Sens: {train_sens_0:.4f} | Spec: {train_spec_0:.4f}"
    )
    print(
        f"Initial Val BalAcc: {val_bal_acc_0:.4f} | "
        f"Sens: {val_sens_0:.4f} | Spec: {val_spec_0:.4f} | AUC: {val_auc_0:.4f}"
    )

    csv_line = (
        f"0,{train_loss_0:.6f},{train_bal_acc_0:.6f},"
        f"{train_sens_0:.6f},{train_spec_0:.6f},"
        f"{val_loss_0:.6f},{val_bal_acc_0:.6f},"
        f"{val_sens_0:.6f},{val_spec_0:.6f},{val_auc_0:.6f}"
    )
    with open(METRICS_CSV, "a") as f:
        f.write(csv_line + "\n")

    best_val_auc = val_auc_0
    best_val_acc_for_es = val_bal_acc_0

    # Main training loop
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss, train_bal_acc, train_sens, train_spec = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_bal_acc, val_sens, val_spec, val_auc, _, _ = evaluate_model(
            model, val_loader, criterion, device, threshold=0.5
        )

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_bal_acc)
        val_accs.append(val_bal_acc)
        train_sens_list.append(train_sens)
        train_spec_list.append(train_spec)
        val_sens_list.append(val_sens)
        val_spec_list.append(val_spec)

        print(
            f"Train BalAcc: {train_bal_acc:.4f} | "
            f"Sens: {train_sens:.4f} | Spec: {train_spec:.4f}"
        )
        print(
            f"Val BalAcc: {val_bal_acc:.4f} | "
            f"Sens: {val_sens:.4f} | Spec: {val_spec:.4f} | AUC: {val_auc:.4f}"
        )

        # Write metrics to CSV for this epoch
        csv_line = (
            f"{epoch},{train_loss:.6f},{train_bal_acc:.6f},"
            f"{train_sens:.6f},{train_spec:.6f},"
            f"{val_loss:.6f},{val_bal_acc:.6f},"
            f"{val_sens:.6f},{val_spec:.6f},{val_auc:.6f}"
        )
        with open(METRICS_CSV, "a") as f:
            f.write(csv_line + "\n")

        # Save best AUC checkpoint
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({"model_state_dict": model.state_dict()}, SAVE_PATH)
            print(f"Saved best model with AUC {val_auc:.4f}")

        # Early stopping on validation balanced accuracy
        if val_bal_acc > (best_val_acc_for_es + MIN_DELTA):
            best_val_acc_for_es = val_bal_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(
                f"[ES] No val accuracy improvement for "
                f"{epochs_no_improve}/{PATIENCE} epoch(s)."
            )

        plot_training_curves(
            train_losses, train_accs, val_losses, val_accs,
            RESULTS_DIR / "training_curves.png"
        )
        plot_loss_curves(
            train_losses, val_losses,
            RESULTS_DIR / "loss_curves.png"
        )
        plot_sensitivity_specificity_curves(
            train_sens_list, train_spec_list,
            val_sens_list, val_spec_list,
            RESULTS_DIR / "sensitivity_specificity_curves.png"
        )

        if epochs_no_improve >= PATIENCE:
            print(f"[ES] Early stopping triggered (patience={PATIENCE}).")
            break

    # ----------------------
    # Threshold calibration on validation set
    # ----------------------
    print("\nCalibrating threshold for target specificity 0.80")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(SAVE_PATH):
        checkpoint = torch.load(SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded best saved model for calibration.")
    else:
        print("No best model checkpoint found, using last model weights.")

    _, _, _, _, val_auc, y_true_val, y_score_val = evaluate_model(
        model, val_loader, criterion, device, threshold=0.5
    )

    thr_any, spec_val, sens_val = _pick_threshold_for_specificity(
        y_true_val, y_score_val, target_spec=0.8
    )
    np.save(THRESHOLDS_PATH, np.array([thr_any], dtype=np.float32))
    print(
        f"Validation AUC: {val_auc:.4f} | "
        f"Threshold@spec80: {thr_any:.4f} | "
        f"Spec: {spec_val:.4f} | Sens: {sens_val:.4f}"
    )

    # Save validation per image outputs
    val_stats_npz = RESULTS_DIR / "binary_val_outputs.npz"
    np.savez(
        val_stats_npz,
        ids=val_labels["ID"].values,
        y_true=y_true_val.astype(np.int8),
        y_score=y_score_val.astype(np.float32)
    )
    print(f"Saved validation outputs to: {val_stats_npz}")

    # ----------------------
    # Final evaluation on test set using calibrated threshold
    # ----------------------
    print("\nFinal evaluation on test set with calibrated threshold")
    test_loss, test_bal_acc, test_sens, test_spec, test_auc, y_true_test, y_score_test = evaluate_model(
        model, test_loader, criterion, device, threshold=thr_any
    )

    y_pred_test = (y_score_test >= thr_any).astype(int)
    tp, tn, fp, fn = compute_confusion_from_preds(y_true_test, y_pred_test)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1max, thr_f1, prec_f1, rec_f1 = _compute_f1max(y_true_test, y_score_test)

    print(
        f"Test BalAcc: {test_bal_acc:.4f} | Sens: {test_sens:.4f} | "
        f"Spec: {test_spec:.4f} | AUC: {test_auc:.4f}"
    )

    overall_csv = RESULTS_DIR / "overall_test_results.csv"
    with open(overall_csv, "w") as f:
        f.write("metric,value\n")
        f.write(f"test_loss,{test_loss:.6f}\n")
        f.write(f"test_balanced_accuracy,{test_bal_acc:.6f}\n")
        f.write(f"test_sensitivity,{test_sens:.6f}\n")
        f.write(f"test_specificity,{test_spec:.6f}\n")
        f.write(f"test_auc,{test_auc:.6f}\n")
        f.write(f"test_precision,{precision:.6f}\n")
        f.write(f"test_recall,{recall:.6f}\n")
        f.write(f"test_tp,{int(tp)}\n")
        f.write(f"test_tn,{int(tn)}\n")
        f.write(f"test_fp,{int(fp)}\n")
        f.write(f"test_fn,{int(fn)}\n")
        f.write(f"thr_spec80,{thr_any:.6f}\n")
        f.write(f"f1max,{f1max:.6f}\n")
        f.write(f"f1max_threshold,{thr_f1:.6f}\n")
        f.write(f"f1max_precision,{prec_f1:.6f}\n")
        f.write(f"f1max_recall,{rec_f1:.6f}\n")

    print(f"Wrote overall test results to: {overall_csv}")

    # Save test per image outputs
    test_stats_npz = RESULTS_DIR / "binary_test_outputs.npz"
    np.savez(
        test_stats_npz,
        ids=test_labels["ID"].values,
        y_true=y_true_test.astype(np.int8),
        y_score=y_score_test.astype(np.float32),
        y_pred_at_spec80=y_pred_test.astype(np.int8),
        thr_spec80=float(thr_any)
    )
    print(f"Saved test outputs to: {test_stats_npz}")

    print(f"\nDone training {pretty}. Best validation AUC: {best_val_auc:.4f}")
    print(f"Best model saved to: {SAVE_PATH}")
    print(f"Threshold (spec80) saved to: {THRESHOLDS_PATH}")
    print(f"Training metrics saved to: {METRICS_CSV}")


# ----------------------
# Entry point
# ----------------------
if __name__ == "__main__":
    # You can change this list if you want other models
    model_names = ["resnet50"]
    for m in model_names:
        print(f"\n==================== {m.upper()} ====================")
        run_for_model(m)
