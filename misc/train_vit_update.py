# src/train_vit_update.py
# Updated ViT training script with binary Disease_Risk head for multi-task learning
# - Trains Disease_Risk as a separate binary classification task (not just one of 45 classes)
# - Uses multi-task learning: multi-label for diseases + binary for Disease_Risk
# - Only trains ViT-Small model
# - Fixes the pooling issue by training Disease_Risk directly as binary target

import os
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
import random, warnings, numpy as np, pandas as pd, math
from PIL import Image

# Suppress PyTorch deprecation warnings for mixed precision
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
from sklearn.exceptions import UndefinedMetricWarning
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------
# Hard-coded parameters
# ----------------------
SEED, BATCH_SIZE, EPOCHS, LR, NUM_WORKERS = 42, 16, 20, 1e-4, 0  # NUM_WORKERS=0 is macOS-safe
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ----------------------
# Paths (root)
# ----------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "RFMiD_Challenge_Dataset"

# These will be set per-model inside run_for_model(...)
RESULTS_DIR = None
SAVE_PATH = None
METRICS_CSV = None
THRESHOLDS_PATH = None
SAVE_PATH_ANY = None

PATIENCE = 10        # stop if val loss doesn't improve for 10 epochs (prevents overfitting)
MIN_DELTA = 1e-4    # minimum improvement to be considered "better"

# Multi-task loss weights
MULTILABEL_LOSS_WEIGHT = 1.0  # Weight for multi-label disease classification
BINARY_LOSS_WEIGHT = 1.0       # Weight for binary Disease_Risk classification

# Label smoothing (reduces overconfidence)
LABEL_SMOOTHING = 0.1  # 0.0 = no smoothing, 0.1 = 10% smoothing (recommended)

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
# Dataset (updated to return Disease_Risk separately)
# ----------------------
class RFMiDDataset(Dataset):
    def __init__(self, img_dir, labels_df, label_columns, transform=None, disease_risk_col="Disease_Risk"):
        self.img_dir, self.labels_df, self.transform = img_dir, labels_df, transform
        self.label_columns = label_columns
        self.num_classes = len(self.label_columns)
        self.disease_risk_col = disease_risk_col
        # Check if Disease_Risk exists
        self.has_disease_risk = disease_risk_col in labels_df.columns

    def __len__(self): return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_id = row['ID']
        img_path = self.img_dir / f"{img_id}.png"
        # fail loudly if image missing/unreadable to avoid poisoning the dataset
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Multi-label targets (all diseases)
        labels = torch.tensor(row[self.label_columns].values, dtype=torch.float32)
        
        # Binary Disease_Risk target (separate from multi-label)
        # Disease_Risk=0 means NORMAL (no diseases present)
        # Disease_Risk=1 means ABNORMAL (at least one disease present)
        if self.has_disease_risk:
            disease_risk_label = torch.tensor(float(row[self.disease_risk_col]), dtype=torch.float32)
        else:
            # If Disease_Risk column doesn't exist, compute from other labels
            # 0.0 = normal (no diseases), 1.0 = abnormal (any disease present)
            disease_risk_label = torch.tensor(1.0 if labels.sum() > 0 else 0.0, dtype=torch.float32)
        
        return image, labels, disease_risk_label

# ----------------------
# Cosine with warmup scheduler for ViTs
# ----------------------
def cosine_with_warmup(optimizer, warmup_epochs=5, total_epochs=20):
    """Cosine decay with warmup scheduler for ViT training"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ----------------------
# Per-model pretrained transforms using timm configs
# ----------------------
def vit_transforms(model_name: str, train: bool):
    """Get transforms for ViT models using timm's pretrained configurations"""
    model_name = model_name.lower()
    
    # Map model names to timm model identifiers
    model_map = {
        "swin_tiny": "swin_tiny_patch4_window7_224",
        "vit_small": "vit_small_patch16_224", 
        "deit_small": "deit_small_patch16_224",
        "crossvit_small": "crossvit_15_240",
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown ViT model name: {model_name}")
    
    # Create model to get its data config
    m = timm.create_model(model_map[model_name], pretrained=True, num_classes=1)
    cfg = resolve_data_config({}, model=m)
    
    # Create transforms using timm's factory
    return create_transform(**cfg, is_training=train)

# ----------------------
# Multi-ViT model builder with binary head
# ----------------------
def build_model(model_name, num_classes, include_binary_head=True):
    """
    Return ViT backbone with:
    1. Multi-label classifier head (for all diseases - num_classes outputs)
    2. Binary classifier head (for Disease_Risk - any abnormal vs normal, 1 output)
    
    Args:
        num_classes: Number of disease classes (e.g., 29 or 45 depending on dataset)
    """
    model_name = model_name.lower()

    # Increased drop_path_rate for all models to reduce overfitting
    if model_name == "swin_tiny":
        # Swin-Tiny: swin_tiny_patch4_window7_224 (~28M params)
        backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0, drop_path_rate=0.25)
        in_f = backbone.num_features
    elif model_name == "vit_small":
        # ViT-Small/16: vit_small_patch16_224 (~22M params)
        # Increased drop_path_rate to reduce overfitting
        backbone = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0, drop_path_rate=0.25)
        in_f = backbone.num_features
    elif model_name == "deit_small":
        # DeiT-Small/16: deit_small_patch16_224 (~22M params)
        backbone = timm.create_model('deit_small_patch16_224', pretrained=True, num_classes=0, drop_path_rate=0.25)
        in_f = backbone.num_features
    elif model_name == "crossvit_small":
        # CrossViT-Small: crossvit_15_240 (~27M params)
        backbone = timm.create_model('crossvit_15_240', pretrained=True, num_classes=0, drop_path_rate=0.25)
        in_f = backbone.num_features
    else:
        raise ValueError(f"Unknown ViT model name: {model_name}. Supported: swin_tiny, vit_small, deit_small, crossvit_small")

    # Multi-label classifier head (for all diseases)
    # Increased dropout to reduce overfitting
    multilabel_classifier = nn.Sequential(
        nn.Dropout(0.6),  # Increased from 0.5
        nn.Linear(in_f, 256),
        nn.ReLU(),
        nn.Dropout(0.4),  # Increased from 0.3
        nn.Linear(256, num_classes)
    )
    
    # Binary classifier head (for Disease_Risk - any abnormal vs normal)
    binary_classifier = None
    if include_binary_head:
        binary_classifier = nn.Sequential(
            nn.Dropout(0.6),  # Increased from 0.5
            nn.Linear(in_f, 256),
            nn.ReLU(),
            nn.Dropout(0.4),  # Increased from 0.3
            nn.Linear(256, 1)  # Binary output
        )
    
    # Create wrapper model
    class ViTMultiTaskWrapper(nn.Module):
        def __init__(self, backbone, multilabel_head, binary_head):
            super().__init__()
            self.backbone = backbone
            self.multilabel_classifier = multilabel_head
            self.binary_classifier = binary_head
            
        def forward(self, x):
            features = self.backbone(x)
            multilabel_logits = self.multilabel_classifier(features)
            
            if self.binary_classifier is not None:
                binary_logits = self.binary_classifier(features)
                return multilabel_logits, binary_logits
            else:
                return multilabel_logits
    
    return ViTMultiTaskWrapper(backbone, multilabel_classifier, binary_classifier)

# ----------------------
# Metric Helpers
# ----------------------
def compute_sens_spec(preds: torch.Tensor, labels: torch.Tensor, average="macro"):
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
    tp = (preds * labels).sum(dim=0).float()
    tn = ((1 - preds) * (1 - labels)).sum(dim=0).float()
    fp = (preds * (1 - labels)).sum(dim=0).float()
    fn = ((1 - preds) * labels).sum(dim=0).float()
    sens_per_class = tp / (tp + fn + 1e-6)
    spec_per_class = tn / (tn + fp + 1e-6)
    return sens_per_class.cpu().numpy(), spec_per_class.cpu().numpy()

# ----------------------
# Threshold Calibration
# ----------------------
def compute_optimal_thresholds(y_true, y_pred, target_spec=0.8):
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
# F1 Metrics at Thresholds
# ----------------------
def compute_f1_at_thresholds(all_labels, all_preds, thresholds=None):
    """Compute Macro and Micro F1 scores at chosen per-class thresholds"""
    from sklearn.metrics import f1_score
    
    if thresholds is None:
        thresholds = np.full(all_preds.shape[1], 0.5)
    
    # Apply thresholds to get binary predictions
    preds_binary = (all_preds > thresholds).astype(int)
    
    try:
        # Macro F1: average of per-class F1 scores
        macro_f1 = f1_score(all_labels, preds_binary, average='macro', zero_division=0)
        
        # Micro F1: treats all classes as one big binary problem
        micro_f1 = f1_score(all_labels, preds_binary, average='micro', zero_division=0)
        
        return macro_f1, micro_f1
    except Exception:
        return 0.0, 0.0

# ----------------------
# Train/Eval routines (updated for multi-task learning)
# ----------------------
def train_one_epoch(model, loader, multilabel_criterion, binary_criterion, optimizer, device, scaler):
    """Train for one epoch with multi-task learning (multi-label + binary Disease_Risk)"""
    model.train()
    running_loss = 0.0
    running_multilabel_loss = 0.0
    running_binary_loss = 0.0

    TP = TN = FP = FN = None
    TP_binary = TN_binary = FP_binary = FN_binary = None
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels, disease_risk_labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        disease_risk_labels = disease_risk_labels.to(device)
        
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            # Forward pass returns both outputs
            multilabel_logits, binary_logits = model(images)
            
            # Apply label smoothing if enabled
            if LABEL_SMOOTHING > 0:
                # Label smoothing: convert hard labels (0,1) to soft labels
                # 0 -> LABEL_SMOOTHING/num_classes, 1 -> 1 - LABEL_SMOOTHING
                labels_smooth = labels * (1.0 - LABEL_SMOOTHING) + LABEL_SMOOTHING / labels.shape[1]
                disease_risk_labels_smooth = disease_risk_labels * (1.0 - LABEL_SMOOTHING) + LABEL_SMOOTHING / 2.0
            else:
                labels_smooth = labels
                disease_risk_labels_smooth = disease_risk_labels
            
            # Compute losses
            multilabel_loss = multilabel_criterion(multilabel_logits, labels_smooth)
            binary_loss = binary_criterion(binary_logits.squeeze(), disease_risk_labels_smooth)
            
            # Combined loss
            total_loss = MULTILABEL_LOSS_WEIGHT * multilabel_loss + BINARY_LOSS_WEIGHT * binary_loss

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += total_loss.item()
        running_multilabel_loss += multilabel_loss.item()
        running_binary_loss += binary_loss.item()

        # Metrics for multi-label
        preds = (torch.sigmoid(multilabel_logits) > 0.5).float()
        tp = (preds * labels).sum(dim=0)
        tn = ((1 - preds) * (1 - labels)).sum(dim=0)
        fp = (preds * (1 - labels)).sum(dim=0)
        fn = ((1 - preds) * labels).sum(dim=0)

        if TP is None:
            TP, TN, FP, FN = tp, tn, fp, fn
        else:
            TP += tp; TN += tn; FP += fp; FN += fn
        
        # Metrics for binary Disease_Risk
        binary_preds = (torch.sigmoid(binary_logits.squeeze()) > 0.5).float()
        tp_b = ((binary_preds == 1) & (disease_risk_labels == 1)).sum().float()
        tn_b = ((binary_preds == 0) & (disease_risk_labels == 0)).sum().float()
        fp_b = ((binary_preds == 1) & (disease_risk_labels == 0)).sum().float()
        fn_b = ((binary_preds == 0) & (disease_risk_labels == 1)).sum().float()
        
        if TP_binary is None:
            TP_binary, TN_binary, FP_binary, FN_binary = tp_b, tn_b, fp_b, fn_b
        else:
            TP_binary += tp_b; TN_binary += tn_b; FP_binary += fp_b; FN_binary += fn_b

    sens = (TP.sum() / (TP.sum() + FN.sum() + 1e-6)).item()
    spec = (TN.sum() / (TN.sum() + FP.sum() + 1e-6)).item()
    bal_acc = 0.5 * (sens + spec)
    
    # Binary Disease_Risk metrics
    sens_binary = (TP_binary / (TP_binary + FN_binary + 1e-6)).item()
    spec_binary = (TN_binary / (TN_binary + FP_binary + 1e-6)).item()
    bal_acc_binary = 0.5 * (sens_binary + spec_binary)
    
    avg_loss = running_loss / len(loader)
    avg_multilabel_loss = running_multilabel_loss / len(loader)
    avg_binary_loss = running_binary_loss / len(loader)
    
    return avg_loss, avg_multilabel_loss, avg_binary_loss, bal_acc, sens, spec, bal_acc_binary, sens_binary, spec_binary

@torch.no_grad()
def evaluate_model(model, loader, multilabel_criterion, binary_criterion, device, thresholds=None):
    """Evaluate with multi-task outputs (multi-label + binary Disease_Risk)"""
    model.eval()
    running_loss = 0.0
    running_multilabel_loss = 0.0
    running_binary_loss = 0.0
    
    TP = TN = FP = FN = None
    TP_binary = TN_binary = FP_binary = FN_binary = None
    all_preds, all_labels = [], []
    all_binary_preds, all_binary_labels = [], []

    pbar = tqdm(loader, desc="Evaluating", leave=False)
    for images, labels, disease_risk_labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        disease_risk_labels = disease_risk_labels.to(device)
        
        multilabel_logits, binary_logits = model(images)
        
        # Apply label smoothing if enabled (same as training)
        if LABEL_SMOOTHING > 0:
            labels_smooth = labels * (1.0 - LABEL_SMOOTHING) + LABEL_SMOOTHING / labels.shape[1]
            disease_risk_labels_smooth = disease_risk_labels * (1.0 - LABEL_SMOOTHING) + LABEL_SMOOTHING / 2.0
        else:
            labels_smooth = labels
            disease_risk_labels_smooth = disease_risk_labels
        
        # Compute losses
        multilabel_loss = multilabel_criterion(multilabel_logits, labels_smooth)
        binary_loss = binary_criterion(binary_logits.squeeze(), disease_risk_labels_smooth)
        total_loss = MULTILABEL_LOSS_WEIGHT * multilabel_loss + BINARY_LOSS_WEIGHT * binary_loss
        
        running_loss += total_loss.item()
        running_multilabel_loss += multilabel_loss.item()
        running_binary_loss += binary_loss.item()

        probs = torch.sigmoid(multilabel_logits).cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        binary_probs = torch.sigmoid(binary_logits.squeeze()).cpu().numpy()
        binary_labels_np = disease_risk_labels.cpu().numpy()

        if thresholds is None:
            preds = (probs > 0.5).astype(float)
        else:
            preds = (probs > thresholds).astype(float)

        preds_t = torch.tensor(preds)
        labels_t = torch.tensor(labels_np)

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
        
        # Binary Disease_Risk metrics
        binary_preds = (binary_probs > 0.5).astype(float)
        tp_b = ((binary_preds == 1) & (binary_labels_np == 1)).sum()
        tn_b = ((binary_preds == 0) & (binary_labels_np == 0)).sum()
        fp_b = ((binary_preds == 1) & (binary_labels_np == 0)).sum()
        fn_b = ((binary_preds == 0) & (binary_labels_np == 1)).sum()
        
        if TP_binary is None:
            TP_binary, TN_binary, FP_binary, FN_binary = tp_b, tn_b, fp_b, fn_b
        else:
            TP_binary += tp_b; TN_binary += tn_b; FP_binary += fp_b; FN_binary += fn_b
        
        all_binary_preds.append(binary_probs)
        all_binary_labels.append(binary_labels_np)

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_binary_preds = np.concatenate(all_binary_preds)
    all_binary_labels = np.concatenate(all_binary_labels)

    sens = (TP.sum() / (TP.sum() + FN.sum() + 1e-6)).item()
    spec = (TN.sum() / (TN.sum() + FP.sum() + 1e-6)).item()
    bal_acc = 0.5 * (sens + spec)

    sens_per_class = (TP / (TP + FN + 1e-6)).cpu().numpy()
    spec_per_class = (TN / (TN + FP + 1e-6)).cpu().numpy()
    
    # Binary Disease_Risk metrics
    sens_binary = (TP_binary / (TP_binary + FN_binary + 1e-6)).item()
    spec_binary = (TN_binary / (TN_binary + FP_binary + 1e-6)).item()
    bal_acc_binary = 0.5 * (sens_binary + spec_binary)

    try:
        valid_cols = (np.sum(all_labels, axis=0) > 0) & (np.sum(all_labels == 0, axis=0) > 0)
        if np.any(valid_cols):
            # Macro AUROC (existing)
            auc_score = roc_auc_score(all_labels[:, valid_cols], all_preds[:, valid_cols], average='macro')
            # Micro AUROC (handles prevalence/imbalance by pooling)
            auc_micro_score = roc_auc_score(all_labels[:, valid_cols], all_preds[:, valid_cols], average='micro')
        else:
            auc_score = 0.0
            auc_micro_score = 0.0
    except Exception:
        auc_score = 0.0
        auc_micro_score = 0.0
    
    # Binary Disease_Risk AUC
    try:
        if len(np.unique(all_binary_labels)) >= 2:
            auc_binary = roc_auc_score(all_binary_labels, all_binary_preds)
        else:
            auc_binary = float("nan")
    except Exception:
        auc_binary = float("nan")

    # Calculate Macro and Micro F1 at chosen per-class thresholds
    macro_f1, micro_f1 = compute_f1_at_thresholds(all_labels, all_preds, thresholds)

    avg_loss = running_loss / len(loader)
    avg_multilabel_loss = running_multilabel_loss / len(loader)
    avg_binary_loss = running_binary_loss / len(loader)
    
    return (avg_loss, avg_multilabel_loss, avg_binary_loss, bal_acc, sens, spec, auc_score, auc_micro_score, 
            macro_f1, micro_f1, all_labels, all_preds, sens_per_class, spec_per_class,
            bal_acc_binary, sens_binary, spec_binary, auc_binary, all_binary_labels, all_binary_preds)

def overall_confusion_from_batches(all_labels, all_preds, thresholds=None):
    """Calculate overall TP, TN, FP, FN from batched predictions and labels"""
    thr = 0.5 if thresholds is None else thresholds
    preds = (all_preds > thr).astype(float)
    TP = (preds * all_labels).sum()
    TN = ((1 - preds) * (1 - all_labels)).sum()
    FP = (preds * (1 - all_labels)).sum()
    FN = ((1 - preds) * all_labels).sum()
    return int(TP), int(TN), int(FP), int(FN)

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
    try:
        epochs = np.arange(0, len(train_sens))
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_sens, label="Training Sensitivity", color='blue', linewidth=2, marker='o', markersize=4)
        plt.plot(epochs, train_spec, label="Training Specificity", color='lightblue', linewidth=2, marker='^', markersize=4)
        plt.plot(epochs, val_sens, label="Validation Sensitivity", color='red', linewidth=2, marker='s', markersize=4)
        plt.plot(epochs, val_spec, label="Validation Specificity", color='lightcoral', linewidth=2, marker='d', markersize=4)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Sensitivity / Specificity", fontsize=12)
        plt.title("Training and Validation Sensitivity & Specificity Over Time", fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ Sensitivity/Specificity curves saved to: {out_path}")
    except Exception as e:
        print(f"[WARN] Failed to plot sensitivity/specificity curves: {e}")

# ----------------------
# Core training pipeline — runs for one model
# ----------------------
def run_for_model(model_name: str):
    global RESULTS_DIR, SAVE_PATH, METRICS_CSV, THRESHOLDS_PATH, SAVE_PATH_ANY

    pretty = {
        "swin_tiny": "SwinTiny",
        "vit_small": "ViTSmall",
        "deit_small": "DeiTSmall",
        "crossvit_small": "CrossViTSmall",
    }[model_name.lower()]

    RESULTS_DIR = ROOT_DIR / "results" / "ViT" / pretty / "Updated"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_PATH = RESULTS_DIR / f"{model_name.lower()}_rfmid_best.pth"
    METRICS_CSV = RESULTS_DIR / f"{model_name.lower()}_metrics.csv"
    THRESHOLDS_PATH = RESULTS_DIR / "optimal_thresholds.npy"
    SAVE_PATH_ANY = RESULTS_DIR / f"{model_name.lower()}_rfmid_best_any_abnormal.pth"

    print(f"🚀 Starting {pretty} training with BINARY Disease_Risk head (multi-task learning)")
    print(f"   Multi-label loss weight: {MULTILABEL_LOSS_WEIGHT}")
    print(f"   Binary loss weight: {BINARY_LOSS_WEIGHT}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_labels = pd.read_csv(DATA_DIR / "2. Groundtruths" / "a. RFMiD_Training_Labels.csv")
    val_labels = pd.read_csv(DATA_DIR / "2. Groundtruths" / "b. RFMiD_Validation_Labels.csv")
    test_labels = pd.read_csv(DATA_DIR / "2. Groundtruths" / "c. RFMiD_Testing_Labels.csv")

    # CRITICAL: Disease_Risk is separate; do NOT include it in multi-label head
    # Multi-label head = only disease classes (excludes Disease_Risk)
    # Binary head = Disease_Risk only (any abnormal vs normal)
    all_cols = train_labels.columns.tolist()
    disease_label_columns = [c for c in all_cols if c not in ["ID", "Disease_Risk"]]
    num_classes = len(disease_label_columns)
    
    print(f"📊 Found {num_classes} disease classes (for multi-label head)")
    print(f"   Disease columns: {disease_label_columns[:5]}..." if len(disease_label_columns) > 5 else f"   Disease columns: {disease_label_columns}")
    
    # Check if Disease_Risk exists
    has_disease_risk = "Disease_Risk" in train_labels.columns
    if has_disease_risk:
        print(f"✅ Found Disease_Risk column - will train binary head separately (NOT in multi-label head)")
        
        # Reindex VAL/TEST to include Disease_Risk separately
        val_labels = val_labels.reindex(columns=["ID"] + disease_label_columns + (["Disease_Risk"] if "Disease_Risk" in val_labels.columns else []), fill_value=0)
        test_labels = test_labels.reindex(columns=["ID"] + disease_label_columns + (["Disease_Risk"] if "Disease_Risk" in test_labels.columns else []), fill_value=0)
        
        # VALIDATION: Verify that Disease_Risk=0 means truly normal (no diseases)
        # and Disease_Risk=1 means abnormal (at least one disease)
        y_disease = train_labels[disease_label_columns].values
        drisk_values = train_labels["Disease_Risk"].values
        
        # Check consistency: Disease_Risk=0 should mean no diseases
        normal_mask = (drisk_values == 0)
        abnormal_mask = (drisk_values == 1)
        
        # For normal images (Disease_Risk=0), check if any have diseases
        normal_with_diseases = (y_disease[normal_mask].sum(axis=1) > 0).sum()
        # For abnormal images (Disease_Risk=1), check if any have no diseases
        abnormal_without_diseases = (y_disease[abnormal_mask].sum(axis=1) == 0).sum()
        
        print(f"   Validation: {normal_with_diseases} normal images (Disease_Risk=0) have diseases (should be 0)")
        print(f"   Validation: {abnormal_without_diseases} abnormal images (Disease_Risk=1) have no diseases (should be 0)")
        
        if normal_with_diseases > 0:
            print(f"   ⚠️ WARNING: {normal_with_diseases} images marked as normal (Disease_Risk=0) actually have diseases!")
            print(f"   This may indicate label inconsistency. The model will still learn, but may be confused.")
        
        if abnormal_without_diseases > 0:
            print(f"   ⚠️ WARNING: {abnormal_without_diseases} images marked as abnormal (Disease_Risk=1) have no diseases!")
            print(f"   This may indicate label inconsistency. The model will still learn, but may be confused.")
        
        # Count normal vs abnormal
        drisk_pos = drisk_values.sum()
        drisk_neg = len(drisk_values) - drisk_pos
        print(f"   Disease_Risk distribution: {drisk_neg} normal (0), {drisk_pos} abnormal (1)")
        print(f"   Normal images: {drisk_neg}/{len(drisk_values)} ({drisk_neg/len(drisk_values)*100:.1f}%)")
        print(f"   Abnormal images: {drisk_pos}/{len(drisk_values)} ({drisk_pos/len(drisk_values)*100:.1f}%)")
        
        drisk_pos_weight = torch.tensor((drisk_neg / (drisk_pos + 1e-6)), dtype=torch.float32).to(device)
    else:
        print(f"⚠️ Disease_Risk column not found - will compute from disease labels")
        # Reindex VAL/TEST without Disease_Risk
        val_labels = val_labels.reindex(columns=["ID"] + disease_label_columns, fill_value=0)
        test_labels = test_labels.reindex(columns=["ID"] + disease_label_columns, fill_value=0)
        
        y_disease = train_labels[disease_label_columns].values
        # Compute from disease labels: any disease present = abnormal
        drisk_values = (y_disease.sum(axis=1) > 0).astype(float)
        drisk_pos = drisk_values.sum()
        drisk_neg = len(drisk_values) - drisk_pos
        print(f"   Computed Disease_Risk from disease labels: {drisk_neg} normal (0), {drisk_pos} abnormal (1)")
        drisk_pos_weight = torch.tensor((drisk_neg / (drisk_pos + 1e-6)), dtype=torch.float32).to(device)

    # Compute per-class pos_weight from TRAIN ONLY (for BCEWithLogitsLoss)
    # IMPORTANT: Only use disease columns, NOT Disease_Risk
    pos = y_disease.sum(axis=0)
    neg = y_disease.shape[0] - pos
    pos_weight = torch.tensor((neg / (pos + 1e-6)).astype(np.float32)).to(device)

    print(f"Training samples: {len(train_labels)}, Validation: {len(val_labels)}, Test: {len(test_labels)}")

    # Per-model transforms from pretrained weights
    train_transform    = vit_transforms(model_name, train=True)
    val_test_transform = vit_transforms(model_name, train=False)

    # Use disease_label_columns for datasets (excludes Disease_Risk from multi-label)
    train_dataset = RFMiDDataset(DATA_DIR / "1. Original Images" / "a. Training Set", train_labels, disease_label_columns, train_transform)
    val_dataset   = RFMiDDataset(DATA_DIR / "1. Original Images" / "b. Validation Set", val_labels, disease_label_columns, val_test_transform)
    test_dataset  = RFMiDDataset(DATA_DIR / "1. Original Images" / "c. Testing Set", test_labels, disease_label_columns, val_test_transform)

    train_loader = DataLoader(train_dataset, BATCH_SIZE, True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_dataset,   BATCH_SIZE, False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_dataset,  BATCH_SIZE, False, num_workers=NUM_WORKERS)

    # Build model with binary head
    # num_classes = number of disease classes (excludes Disease_Risk)
    model = build_model(model_name, num_classes=num_classes, include_binary_head=True).to(device)
    
    # Two separate criteria
    multilabel_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    binary_criterion = nn.BCEWithLogitsLoss(pos_weight=drisk_pos_weight)

    # Optimizer/scheduler for ViT models (AdamW + cosine with warmup)
    backbone_params = list(model.backbone.parameters())
    multilabel_params = list(model.multilabel_classifier.parameters())
    binary_params = list(model.binary_classifier.parameters())

    optimizer = torch.optim.AdamW([
        {'params': backbone_params,      'lr': 5e-5, 'weight_decay': 0.05},  # Gentle LR for backbone
        {'params': multilabel_params,     'lr': 4e-4, 'weight_decay': 0.01},  # Reduced LR, added WD to reduce overfitting
        {'params': binary_params,         'lr': 4e-4, 'weight_decay': 0.01},  # Reduced LR, added WD to reduce overfitting
    ])
    scheduler = cosine_with_warmup(optimizer, warmup_epochs=5, total_epochs=EPOCHS)

    # CSV header with per-class columns + binary Disease_Risk metrics
    # class_names = disease classes only (excludes Disease_Risk)
    class_names = disease_label_columns
    header = "epoch,train_loss,train_multilabel_loss,train_binary_loss,train_bal_acc,train_sens,train_spec,"
    header += "train_binary_bal_acc,train_binary_sens,train_binary_spec,"
    header += "val_loss,val_multilabel_loss,val_binary_loss,val_bal_acc,val_sens,val_spec,val_auc,"
    header += "val_binary_bal_acc,val_binary_sens,val_binary_spec,val_binary_auc"
    for class_name in class_names:
        header += f",val_sens_{class_name},val_spec_{class_name}"
    with open(METRICS_CSV, "w") as f:
        f.write(header + "\n")

    best_val_auc = 0.0
    best_binary_auc = 0.0
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    train_sens_list, train_spec_list, val_sens_list, val_spec_list = [], [], [], []

    # Epoch 0: initial evaluation
    print("\n📊 Epoch 0: Evaluating initial model performance...")
    train_loss_0, train_multilabel_loss_0, train_binary_loss_0, train_bal_acc_0, train_sens_0, train_spec_0, train_binary_bal_acc_0, train_binary_sens_0, train_binary_spec_0 = train_one_epoch(model, train_loader, multilabel_criterion, binary_criterion, optimizer, device, torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available()))
    
    eval_results = evaluate_model(model, val_loader, multilabel_criterion, binary_criterion, device)
    (val_loss_0, val_multilabel_loss_0, val_binary_loss_0, val_bal_acc_0, val_sens_0, val_spec_0, 
     val_auc_0, _, _, _, _, _, val_sens_per_class_0, val_spec_per_class_0,
     val_binary_bal_acc_0, val_binary_sens_0, val_binary_spec_0, val_binary_auc_0, _, _) = eval_results

    train_losses.append(train_loss_0); val_losses.append(val_loss_0)
    train_accs.append(train_bal_acc_0); val_accs.append(val_bal_acc_0)
    train_sens_list.append(train_sens_0); train_spec_list.append(train_spec_0)
    val_sens_list.append(val_sens_0); val_spec_list.append(val_spec_0)

    print(f"Initial Train Balanced Acc: {train_bal_acc_0:.4f} | Sens: {train_sens_0:.4f} | Spec: {train_spec_0:.4f}")
    print(f"Initial Train Binary (Disease_Risk) - Bal Acc: {train_binary_bal_acc_0:.4f} | Sens: {train_binary_sens_0:.4f} | Spec: {train_binary_spec_0:.4f}")
    print(f"Initial Val Balanced Acc: {val_bal_acc_0:.4f} | Sens: {val_sens_0:.4f} | Spec: {val_spec_0:.4f} | AUC: {val_auc_0:.4f}")
    print(f"Initial Val Binary (Disease_Risk) - Bal Acc: {val_binary_bal_acc_0:.4f} | Sens: {val_binary_sens_0:.4f} | Spec: {val_binary_spec_0:.4f} | AUC: {val_binary_auc_0:.4f}")

    csv_line = (f"0,{train_loss_0:.6f},{train_multilabel_loss_0:.6f},{train_binary_loss_0:.6f},"
                f"{train_bal_acc_0:.6f},{train_sens_0:.6f},{train_spec_0:.6f},"
                f"{train_binary_bal_acc_0:.6f},{train_binary_sens_0:.6f},{train_binary_spec_0:.6f},"
                f"{val_loss_0:.6f},{val_multilabel_loss_0:.6f},{val_binary_loss_0:.6f},"
                f"{val_bal_acc_0:.6f},{val_sens_0:.6f},{val_spec_0:.6f},{val_auc_0:.6f},"
                f"{val_binary_bal_acc_0:.6f},{val_binary_sens_0:.6f},{val_binary_spec_0:.6f},{val_binary_auc_0:.6f}")
    for i in range(len(class_names)):
        csv_line += f",{val_sens_per_class_0[i]:.6f},{val_spec_per_class_0[i]:.6f}"
    with open(METRICS_CSV, "a") as f:
        f.write(csv_line + "\n")

    # Early stopping state (on validation AUC)
    best_val_auc_es = -1.0
    epochs_no_improve = 0
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss, train_multilabel_loss, train_binary_loss, train_bal_acc, train_sens, train_spec, train_binary_bal_acc, train_binary_sens, train_binary_spec = train_one_epoch(model, train_loader, multilabel_criterion, binary_criterion, optimizer, device, scaler)
        
        eval_results = evaluate_model(model, val_loader, multilabel_criterion, binary_criterion, device)
        (val_loss, val_multilabel_loss, val_binary_loss, val_bal_acc, val_sens, val_spec, 
         val_auc, val_auc_micro, val_macro_f1, val_micro_f1, y_true_val_all, y_pred_val_all, 
         val_sens_per_class, val_spec_per_class,
         val_binary_bal_acc, val_binary_sens, val_binary_spec, val_binary_auc, 
         y_true_binary_val, y_pred_binary_val) = eval_results

        scheduler.step()
        train_losses.append(train_loss); val_losses.append(val_loss)
        train_accs.append(train_bal_acc); val_accs.append(val_bal_acc)
        train_sens_list.append(train_sens); train_spec_list.append(train_spec)
        val_sens_list.append(val_sens); val_spec_list.append(val_spec)

        print(f"Train Balanced Acc: {train_bal_acc:.4f} | Sens: {train_sens:.4f} | Spec: {train_spec:.4f}")
        print(f"Train Binary (Disease_Risk) - Bal Acc: {train_binary_bal_acc:.4f} | Sens: {train_binary_sens:.4f} | Spec: {train_binary_spec:.4f}")
        print(f"Val Balanced Acc: {val_bal_acc:.4f} | Sens: {val_sens:.4f} | Spec: {val_spec:.4f} | AUC: {val_auc:.4f} | Micro AUC: {val_auc_micro:.4f} | Macro F1: {val_macro_f1:.4f} | Micro F1: {val_micro_f1:.4f}")
        print(f"Val Binary (Disease_Risk) - Bal Acc: {val_binary_bal_acc:.4f} | Sens: {val_binary_sens:.4f} | Spec: {val_binary_spec:.4f} | AUC: {val_binary_auc:.4f}")

        # Checkpoint based on binary Disease_Risk AUC (this is the "any abnormal" task)
        if not np.isnan(val_binary_auc) and val_binary_auc > best_binary_auc:
            best_binary_auc = val_binary_auc
            torch.save({'model_state_dict': model.state_dict()}, SAVE_PATH_ANY)
            print(f"💾 Best BINARY Disease_Risk model saved! (val AUC={val_binary_auc:.4f})")

        # Write metrics to CSV
        csv_line = (f"{epoch},{train_loss:.6f},{train_multilabel_loss:.6f},{train_binary_loss:.6f},"
                    f"{train_bal_acc:.6f},{train_sens:.6f},{train_spec:.6f},"
                    f"{train_binary_bal_acc:.6f},{train_binary_sens:.6f},{train_binary_spec:.6f},"
                    f"{val_loss:.6f},{val_multilabel_loss:.6f},{val_binary_loss:.6f},"
                    f"{val_bal_acc:.6f},{val_sens:.6f},{val_spec:.6f},{val_auc:.6f},"
                    f"{val_binary_bal_acc:.6f},{val_binary_sens:.6f},{val_binary_spec:.6f},{val_binary_auc:.6f}")
        for i in range(len(class_names)):
            csv_line += f",{val_sens_per_class[i]:.6f},{val_spec_per_class[i]:.6f}"
        with open(METRICS_CSV, "a") as f:
            f.write(csv_line + "\n")

        # Early stopping check (val AUC)
        if val_auc > best_val_auc_es + MIN_DELTA:
            best_val_auc_es = val_auc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"[ES] No val-AUC improvement for {epochs_no_improve}/{PATIENCE} epoch(s).")

        if epochs_no_improve >= PATIENCE:
            print(f"[ES] Early stopping triggered (patience={PATIENCE}).")
            plot_training_curves(train_losses, train_accs, val_losses, val_accs, RESULTS_DIR / "training_curves.png")
            plot_loss_curves(train_losses, val_losses, RESULTS_DIR / "loss_curves.png")
            plot_sensitivity_specificity_curves(train_sens_list, train_spec_list, val_sens_list, val_spec_list, RESULTS_DIR / "sensitivity_specificity_curves.png")
            break

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({'model_state_dict': model.state_dict()}, SAVE_PATH)
            print(f"💾 Best model saved! (AUC={val_auc:.4f})")

        # Update plots
        plot_training_curves(train_losses, train_accs, val_losses, val_accs, RESULTS_DIR / "training_curves.png")
        plot_loss_curves(train_losses, val_losses, RESULTS_DIR / "loss_curves.png")
        plot_sensitivity_specificity_curves(train_sens_list, train_spec_list, val_sens_list, val_spec_list, RESULTS_DIR / "sensitivity_specificity_curves.png")

    # ----------------------
    # Threshold calibration
    # ----------------------
    print("\n📊 Calibrating thresholds (target specificity=0.8)...")
    if os.path.exists(SAVE_PATH):
        checkpoint = torch.load(SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Loaded best saved model for calibration.")
    else:
        print("⚠️ No best model saved yet (AUC=nan or interrupted training). Using last trained model instead.")

    eval_results = evaluate_model(model, val_loader, multilabel_criterion, binary_criterion, device)
    (_, _, _, _, _, _, val_auc, _, _, _, y_true_val, y_pred_val, _, _,
     _, _, _, _, y_true_binary_val, y_pred_binary_val) = eval_results
    
    thresholds = compute_optimal_thresholds(np.array(y_true_val), np.array(y_pred_val), target_spec=0.8)
    np.save(THRESHOLDS_PATH, thresholds)
    print(f"Optimal thresholds saved to: {THRESHOLDS_PATH}")

    # =============== Binary Disease_Risk (Any Abnormal vs Normal) metrics ===============
    # Use the binary head directly (no pooling needed!)
    thr_any, spec_any_val, sens_any_val = _pick_threshold_for_specificity(y_true_binary_val, y_pred_binary_val, target_spec=0.8)
    print(f"🔧 Binary Disease_Risk validation operating point @~0.80 specificity: thr={thr_any:.4f}, spec={spec_any_val:.4f}, sens={sens_any_val:.4f}")

    if os.path.exists(SAVE_PATH_ANY):
        checkpoint_any = torch.load(SAVE_PATH_ANY, map_location=device)
        model.load_state_dict(checkpoint_any['model_state_dict'])
        print("✅ Loaded best binary Disease_Risk model for overall metrics.")
        eval_results = evaluate_model(model, val_loader, multilabel_criterion, binary_criterion, device)
        (_, _, _, _, _, _, _, _, _, _, _, _, _, _,
         _, _, _, _, y_true_binary_val_ckpt, y_pred_binary_val_ckpt) = eval_results
        thr_any, spec_any_val, sens_any_val = _pick_threshold_for_specificity(y_true_binary_val_ckpt, y_pred_binary_val_ckpt, target_spec=0.8)
        print(f"🔧 Recomputed binary Disease_Risk val operating point: thr={thr_any:.4f}, spec={spec_any_val:.4f}, sens={sens_any_val:.4f}")
    else:
        print("⚠️ No binary Disease_Risk checkpoint found; using current model for overall metrics.")

    eval_results = evaluate_model(model, test_loader, multilabel_criterion, binary_criterion, device)
    (_, _, _, _, _, _, _, _, _, _, _, _, _, _,
     _, _, _, _, y_true_binary_test, y_pred_binary_test) = eval_results

    auc_any = roc_auc_score(y_true_binary_test, y_pred_binary_test)
    y_pred_any_test = (y_pred_binary_test >= thr_any).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true_binary_test, y_pred_any_test, labels=[0,1]).ravel()
    precision_at_thr = tp / (tp + fp + 1e-8)
    recall_at_thr = tp / (tp + fn + 1e-8)

    f1max, thr_f1, prec_f1, rec_f1 = _compute_f1max(y_true_binary_test, y_pred_binary_test)

    overall_csv = RESULTS_DIR / "overall_any_abnormal_metrics.csv"
    with open(overall_csv, "w") as f:
        f.write("Metric,Value\n")
        f.write(f"AUC (%),{auc_any*100:.4f}\n")
        f.write(f"Threshold@0.80spec,{thr_any:.6f}\n")
        f.write(f"Precision@Thr,{precision_at_thr*100:.4f}\n")
        f.write(f"Recall@Thr (%),{recall_at_thr*100:.4f}\n")
        f.write(f"TP,{int(tp)}\n")
        f.write(f"TN,{int(tn)}\n")
        f.write(f"FP,{int(fp)}\n")
        f.write(f"FN,{int(fn)}\n")
        f.write(f"F1max,{f1max:.6f}\n")
        f.write(f"F1max_Threshold,{thr_f1:.6f}\n")
        f.write(f"F1max_Precision,{prec_f1*100:.4f}\n")
        f.write(f"F1max_Recall (%),{rec_f1*100:.4f}\n")
    print(f"🧾 Wrote overall any-abnormal metrics to: {overall_csv}")

    # ================== SAVE PER-IMAGE OUTPUTS FOR LATER STATS (DeLong & McNemar) ==================
    val_stats_npz = RESULTS_DIR / "vit_anyabnormal_val_outputs.npz"
    np.savez(val_stats_npz,
             ids=val_labels["ID"].values,
             y_true=y_true_binary_val.astype(np.int8),
             y_score=y_pred_binary_val.astype(np.float32))
    print(f"💾 Saved validation per-image any-abnormal outputs to: {val_stats_npz}")

    test_stats_npz = RESULTS_DIR / "vit_anyabnormal_test_outputs.npz"
    np.savez(test_stats_npz,
             ids=test_labels["ID"].values,
             y_true=y_true_binary_test.astype(np.int8),
             y_score=y_pred_binary_test.astype(np.float32),
             y_pred_at_spec80=(y_pred_binary_test >= thr_any).astype(np.int8),
             thr_spec80=float(thr_any))
    print(f"💾 Saved test per-image any-abnormal outputs to: {test_stats_npz}")

    # 🔁 Restore best-AUC checkpoint for per-class final evaluation
    if os.path.exists(SAVE_PATH):
        checkpoint = torch.load(SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Restored best-AUC checkpoint for per-class thresholded evaluation.")
    else:
        print("⚠️ Expected best-AUC checkpoint not found; proceeding with current weights.")

    print("\n🧪 Final evaluation on test set (using calibrated thresholds)...")
    eval_results = evaluate_model(model, test_loader, multilabel_criterion, binary_criterion, device, thresholds)
    (test_loss, _, _, test_bal_acc, test_sens, test_spec, test_auc, test_auc_micro, test_macro_f1, test_micro_f1, 
     test_all_labels, test_all_preds, test_sens_per_class, test_spec_per_class,
     _, _, _, _, _, _) = eval_results
    print(f"Test Balanced Acc: {test_bal_acc:.4f} | Sens: {test_sens:.4f} | Spec: {test_spec:.4f} | AUC: {test_auc:.4f} | Micro AUC: {test_auc_micro:.4f} | Macro F1: {test_macro_f1:.4f} | Micro F1: {test_micro_f1:.4f}")

    test_results_csv = RESULTS_DIR / "final_test_results.csv"
    with open(test_results_csv, "w") as f:
        f.write("class_name,test_sensitivity,test_specificity\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name},{test_sens_per_class[i]:.6f},{test_spec_per_class[i]:.6f}\n")

    # Calculate additional metrics for overall test results
    test_tp, test_tn, test_fp, test_fn = overall_confusion_from_batches(test_all_labels, test_all_preds, thresholds)
    precision_overall = test_tp / (test_tp + test_fp + 1e-8)
    recall_overall = test_tp / (test_tp + test_fn + 1e-8)
    
    # Calculate F1max for overall results
    f1max_overall, thr_f1_overall, prec_f1_overall, rec_f1_overall = _compute_overall_f1max(test_all_labels, test_all_preds)
    
    overall_results_csv = RESULTS_DIR / "overall_test_results.csv"
    with open(overall_results_csv, "w") as f:
        f.write("metric,value\n")
        f.write(f"test_loss,{test_loss:.6f}\n")
        f.write(f"test_balanced_accuracy,{test_bal_acc:.6f}\n")
        f.write(f"test_sensitivity,{test_sens:.6f}\n")
        f.write(f"test_specificity,{test_spec:.6f}\n")
        f.write(f"test_auc,{test_auc:.6f}\n")
        f.write(f"test_auc_micro,{test_auc_micro:.6f}\n")
        f.write(f"test_macro_f1,{test_macro_f1:.6f}\n")
        f.write(f"test_micro_f1,{test_micro_f1:.6f}\n")
        f.write(f"best_validation_auc,{best_val_auc:.6f}\n")
        f.write(f"test_precision,{precision_overall:.6f}\n")
        f.write(f"test_recall,{recall_overall:.6f}\n")
        f.write(f"test_tp,{int(test_tp)}\n")
        f.write(f"test_tn,{int(test_tn)}\n")
        f.write(f"test_fp,{int(test_fp)}\n")
        f.write(f"test_fn,{int(test_fn)}\n")
        f.write(f"test_f1max,{f1max_overall:.6f}\n")
        f.write(f"test_f1max_threshold,{thr_f1_overall:.6f}\n")
        f.write(f"test_f1max_precision,{prec_f1_overall:.6f}\n")
        f.write(f"test_f1max_recall,{rec_f1_overall:.6f}\n")

    print(f"\n🎉 Training completed for {pretty}!")
    print(f"Best validation AUC: {best_val_auc:.4f}")
    print(f"Best binary Disease_Risk AUC: {best_binary_auc:.4f}")
    print(f"Model saved to: {SAVE_PATH}")
    print(f"Binary Disease_Risk model saved to: {SAVE_PATH_ANY}")
    print(f"Thresholds saved to: {THRESHOLDS_PATH}")
    print(f"Training metrics saved to: {METRICS_CSV}")
    print(f"Final test per-class results saved to: {test_results_csv}")
    print(f"Overall test results saved to: {overall_results_csv}")

# ----------------------
# Helpers for any-abnormal operating point
# ----------------------
def _pick_threshold_for_specificity(y_true_binary, y_score, target_spec=0.8):
    fpr, tpr, thr = roc_curve(y_true_binary, y_score)
    spec = 1 - fpr
    idx = np.argmin(np.abs(spec - target_spec))
    return float(thr[idx]), float(spec[idx]), float(tpr[idx])

def _compute_f1max(y_true_binary, y_score):
    precision, recall, thr = precision_recall_curve(y_true_binary, y_score)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    f1_use = f1[:-1]
    best_idx = int(np.nanargmax(f1_use))
    return float(f1_use[best_idx]), float(thr[best_idx]), float(precision[best_idx]), float(recall[best_idx])

def _compute_overall_f1max(all_labels, all_preds):
    """Compute F1max for overall multi-class classification using micro-averaging"""
    # Flatten all predictions and labels for micro-averaging
    y_true_flat = all_labels.flatten()
    y_score_flat = all_preds.flatten()
    
    # Only consider valid predictions (where there are both positive and negative samples)
    valid_mask = np.isfinite(y_score_flat)
    y_true_valid = y_true_flat[valid_mask]
    y_score_valid = y_score_flat[valid_mask]
    
    if len(y_true_valid) == 0 or len(np.unique(y_true_valid)) < 2:
        return 0.0, 0.5, 0.0, 0.0
    
    # Use precision_recall_curve for micro-averaged F1max
    precision, recall, thr = precision_recall_curve(y_true_valid, y_score_valid)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    f1_use = f1[:-1]  # Remove last element as it's always 1.0
    best_idx = int(np.nanargmax(f1_use))
    return float(f1_use[best_idx]), float(thr[best_idx]), float(precision[best_idx]), float(recall[best_idx])

# ----------------------
# Entry point
# ----------------------
if __name__ == "__main__":
    # Train all 4 ViT models with binary Disease_Risk head
    model_names = ["swin_tiny", "vit_small", "deit_small", "crossvit_small"]
    for model_name in model_names:
        print(f"\n{'='*70}")
        print(f"{'='*70}")
        print(f"Training {model_name.upper()} (WITH BINARY HEAD + LABEL SMOOTHING)")
        print(f"{'='*70}")
        print(f"{'='*70}\n")
        run_for_model(model_name)

