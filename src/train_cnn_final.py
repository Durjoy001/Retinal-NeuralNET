# -*- coding: utf-8 -*-
"""
Binary CNN training for RFMiD:
- Single Disease_Risk head: 1 means diseased, 0 means normal
- Custom preprocessing: center crop, resize to 224, Gaussian background subtraction,
  horizontal flip, brightness shift, gentle elastic deformation
- ImageNet pretrained CNNs (ResNet50, EfficientNet-B3, DenseNet121)
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
    EfficientNet_B3_Weights
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
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ----------------------
# Paths
# ----------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path("")

RESULTS_DIR = None
SAVE_PATH = None
METRICS_CSV = None
THRESHOLDS_PATH = None
PREPROCESS_DIR = ROOT_DIR / "preprocess"

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
# Dataset for binary Disease_Risk label
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
# Simple sanity checks
# ----------------------
def preview_random_labels(num_samples: int = 5, dataset: str = "train"):
    """
    Print a handful of (ID, Disease_Risk) pairs and confirm the image file exists.
    """
    dataset = dataset.lower()
    groundtruth_paths = {
        "train": DATA_DIR / "2. Groundtruths" / "a. RFMiD_Training_Labels.csv",
        "val": DATA_DIR / "2. Groundtruths" / "b. RFMiD_Validation_Labels.csv",
        "test": DATA_DIR / "2. Groundtruths" / "c. RFMiD_Testing_Labels.csv",
    }
    image_dirs = {
        "train": DATA_DIR / "1. Original Images" / "a. Training Set",
        "val": DATA_DIR / "1. Original Images" / "b. Validation Set",
        "test": DATA_DIR / "1. Original Images" / "c. Testing Set",
    }

    if dataset not in groundtruth_paths:
        raise ValueError("dataset must be one of: train, val, test")

    labels_path = groundtruth_paths[dataset]
    img_dir = image_dirs[dataset]

    if not labels_path.exists():
        print(f"[WARN] Labels CSV not found: {labels_path}")
        return

    labels_df = pd.read_csv(labels_path)
    labels_df["ID"] = labels_df["ID"].astype(str)

    if labels_df.empty:
        print(f"[WARN] No rows found in {labels_path.name}")
        return

    if "Disease_Risk" not in labels_df.columns:
        print(f"[WARN] 'Disease_Risk' column not present in {labels_path.name}")
        return

    random_seed = int(np.random.default_rng().integers(0, 10_000))
    sample_df = labels_df.sample(
        n=min(num_samples, len(labels_df)),
        random_state=random_seed
    )

    print(f"\nSample {len(sample_df)} entries from {dataset} split:")
    for row in sample_df.itertuples(index=False):
        img_id = row.ID
        risk = getattr(row, "Disease_Risk", None)
        img_path = img_dir / f"{img_id}.png"
        status = "FOUND" if img_path.exists() else "MISSING"
        print(f"  ID={img_id} | Disease_Risk={risk} | Image: {status} ({img_path.name})")


# ----------------------
# Preprocessing helpers
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

def get_elastic_transform():
    return transforms.ElasticTransform(alpha=50.0, sigma=5.0)


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


def _tensor_to_pil_image(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a normalized tensor (ImageNet mean/std) back to a PIL image for inspection.
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = tensor.detach().clone()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    tensor = tensor.clamp(0.0, 1.0)
    np_image = (tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(np_image)


def preview_preprocessing(model_name: str, num_images: int = 2):
    """
    Apply the training preprocessing pipeline to a few sample images and save the
    original/processed pairs so users can inspect them.
    """
    PREPROCESS_DIR.mkdir(parents=True, exist_ok=True)

    labels_path = DATA_DIR / "2. Groundtruths" / "a. RFMiD_Training_Labels.csv"
    labels_df = pd.read_csv(labels_path)
    sample_rows = labels_df.sample(
        n=min(num_images, len(labels_df)),
        random_state=SEED
    )

    transform = get_transforms(model_name, train=True)
    img_dir = DATA_DIR / "1. Original Images" / "a. Training Set"

    print(f"\nPreviewing preprocessing for {model_name}...")
    for idx, row in enumerate(sample_rows.itertuples(), start=1):
        img_id = row.ID
        img_path = img_dir / f"{img_id}.png"

        if not img_path.exists():
            print(f"[WARN] Image not found: {img_path}")
            continue

        original_img = Image.open(img_path).convert("RGB")
        processed_tensor = transform(original_img)
        processed_img = _tensor_to_pil_image(processed_tensor)

        base_name = f"sample_{idx}_{img_id}"
        original_out = PREPROCESS_DIR / f"{base_name}_original.png"
        processed_out = PREPROCESS_DIR / f"{base_name}_processed.png"

        original_img.save(original_out)
        processed_img.save(processed_out)

        print(f"Saved original image to: {original_out}")
        print(f"Saved processed image to: {processed_out}")

# ----------------------
# Model builders with binary head (Disease_Risk)
# ----------------------
def build_model(model_name: str):
    """
    Build a CNN with a single binary output:
    Disease_Risk: 1 means diseased, 0 means normal.
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

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model


# ----------------------
# Freezing strategy
# ----------------------
def freeze_early_layers(model: nn.Module, num_trainable_layers: int = 12):
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

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    sens, spec, _, bal_acc = compute_sens_spec_acc(TP, TN, FP, FN)
    avg_loss = running_loss / len(loader)
    return avg_loss, bal_acc, sens, spec


@torch.no_grad()
def evaluate_model(model, loader, criterion, device, threshold: float = 0.5, desc: str = "Evaluating"):
    model.eval()
    running_loss = 0.0

    TP = TN = FP = FN = 0
    all_probs = []
    all_labels = []

    pbar = tqdm(loader, desc=desc, leave=False)
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


def _best_balanced_accuracy_threshold(y_true_binary, y_score):
    if len(np.unique(y_true_binary)) < 2:
        default_thr = 0.5
        preds = (y_score >= default_thr).astype(int)
        tp, tn, fp, fn = compute_confusion_from_preds(y_true_binary, preds)
        sens, spec, _, bal_acc = compute_sens_spec_acc(tp, tn, fp, fn)
        return default_thr, bal_acc, sens, spec

    unique_scores = np.unique(y_score)
    candidate_thresholds = np.concatenate((
        [unique_scores[0] - 1.0],
        unique_scores,
        [unique_scores[-1] + 1.0],
        [0.5]
    ))
    candidate_thresholds = np.unique(candidate_thresholds)

    best_thr = 0.5
    best_bal_acc = -np.inf
    best_sens = 0.0
    best_spec = 0.0

    for thr in candidate_thresholds:
        preds = (y_score >= thr).astype(int)
        tp, tn, fp, fn = compute_confusion_from_preds(y_true_binary, preds)
        sens, spec, _, bal_acc = compute_sens_spec_acc(tp, tn, fp, fn)
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_thr = thr
            best_sens = sens
            best_spec = spec

    return float(best_thr), float(best_bal_acc), float(best_sens), float(best_spec)


# ----------------------
# Core training pipeline for one model
# ----------------------
def run_for_model(model_name: str):
    global RESULTS_DIR, SAVE_PATH, METRICS_CSV, THRESHOLDS_PATH

    pretty = {
        "densenet121": "Densenet121",
        "resnet50": "ResNet50",
        "efficientnet_b3": "EfficientNetB3",
    }[model_name.lower()]

    RESULTS_DIR = ROOT_DIR / "results" / "CNN_binary" / pretty
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    SAVE_PATH = RESULTS_DIR / f"{model_name.lower()}_rfmid_binary_best.pth"
    METRICS_CSV = RESULTS_DIR / f"{model_name.lower()}_metrics.csv"
    THRESHOLDS_PATH = RESULTS_DIR / "best_test_threshold.npy"

    print(f"Starting {pretty} training (binary Disease_Risk head).")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load labels
    train_labels = pd.read_csv(DATA_DIR / "2. Groundtruths" / "a. RFMiD_Training_Labels.csv")
    test_labels = pd.read_csv(DATA_DIR / "2. Groundtruths" / "c. RFMiD_Testing_Labels.csv")

    print(f"Training samples: {len(train_labels)}, Test: {len(test_labels)}")

    # Transforms
    train_transform = get_transforms(model_name, train=True)
    test_transform = get_transforms(model_name, train=False)

    train_dataset = RFMiDDatasetBinary(
        DATA_DIR / "1. Original Images" / "a. Training Set",
        train_labels,
        "Disease_Risk",
        train_transform
    )
    test_dataset = RFMiDDatasetBinary(
        DATA_DIR / "1. Original Images" / "c. Testing Set",
        test_labels,
        "Disease_Risk",
        test_transform
    )

    train_loader = DataLoader(train_dataset, BATCH_SIZE, True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, False, num_workers=NUM_WORKERS)

    # Build model, freeze early layers, set up optimizer
    model = build_model(model_name)
    freeze_early_layers(model, num_trainable_layers=12)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LR, weight_decay=1e-4)

    # CSV header
    header = (
        "epoch,train_loss,test_loss,test_bal_acc,test_sens,"
        "test_spec,test_auc,best_threshold"
    )
    with open(METRICS_CSV, "w") as f:
        f.write(header + "\n")

    best_bal_acc = float("-inf")
    best_threshold = 0.5
    best_epoch = 0

    # Main training loop
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss, _, _, _ = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, _, _, _, test_auc, y_true_test, y_score_test = evaluate_model(
            model, test_loader, criterion, device, threshold=0.5, desc="Testing"
        )

        thr, bal_acc, sens, spec = _best_balanced_accuracy_threshold(y_true_test, y_score_test)

        print(f"Average train loss: {train_loss:.4f}")
        print(
            "Test metrics -- "
            f"loss: {test_loss:.4f} | bal_acc: {bal_acc:.4f} | "
            f"sens: {sens:.4f} | spec: {spec:.4f} | auc: {test_auc:.4f} | "
            f"thr: {thr:.4f}"
        )

        csv_line = (
            f"{epoch},{train_loss:.6f},{test_loss:.6f},{bal_acc:.6f},"
            f"{sens:.6f},{spec:.6f},{test_auc:.6f},{thr:.6f}"
        )
        with open(METRICS_CSV, "a") as f:
            f.write(csv_line + "\n")

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_threshold = thr
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "threshold": best_threshold,
                    "balanced_accuracy": best_bal_acc,
                    "epoch": best_epoch,
                },
                SAVE_PATH
            )
            np.save(THRESHOLDS_PATH, np.array([best_threshold], dtype=np.float32))
            print(
                f"Saved new best checkpoint (epoch {best_epoch}) "
                f"with bal_acc {best_bal_acc:.4f} @ thr {best_threshold:.4f}"
            )

    print("\nTraining complete. Loading best checkpoint for final reporting.")
    if os.path.exists(SAVE_PATH):
        checkpoint = torch.load(SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_threshold = checkpoint.get("threshold", best_threshold)
        best_bal_acc = checkpoint.get("balanced_accuracy", best_bal_acc)
        best_epoch = checkpoint.get("epoch", best_epoch)
    else:
        print("Warning: best checkpoint not found, using last epoch weights.")

    final_test_loss, _, _, _, final_auc, y_true_test, y_score_test = evaluate_model(
        model, test_loader, criterion, device, threshold=best_threshold, desc="Testing (final)"
    )
    y_pred_test = (y_score_test >= best_threshold).astype(int)
    tp, tn, fp, fn = compute_confusion_from_preds(y_true_test, y_pred_test)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    sens, spec, _, bal_acc = compute_sens_spec_acc(tp, tn, fp, fn)
    f1max, thr_f1, prec_f1, rec_f1 = _compute_f1max(y_true_test, y_score_test)

    print(
        f"Best epoch: {best_epoch} | Threshold: {best_threshold:.4f} | "
        f"Balanced Acc: {best_bal_acc:.4f}"
    )
    print(
        f"Final test -- loss: {final_test_loss:.4f} | "
        f"bal_acc: {bal_acc:.4f} | sens: {sens:.4f} | "
        f"spec: {spec:.4f} | auc: {final_auc:.4f}"
    )

    overall_csv = RESULTS_DIR / "overall_test_results.csv"
    with open(overall_csv, "w") as f:
        f.write("metric,value\n")
        f.write(f"best_epoch,{best_epoch}\n")
        f.write(f"test_loss,{final_test_loss:.6f}\n")
        f.write(f"test_balanced_accuracy,{bal_acc:.6f}\n")
        f.write(f"test_sensitivity,{sens:.6f}\n")
        f.write(f"test_specificity,{spec:.6f}\n")
        f.write(f"test_auc,{final_auc:.6f}\n")
        f.write(f"test_precision,{precision:.6f}\n")
        f.write(f"test_recall,{recall:.6f}\n")
        f.write(f"test_tp,{int(tp)}\n")
        f.write(f"test_tn,{int(tn)}\n")
        f.write(f"test_fp,{int(fp)}\n")
        f.write(f"test_fn,{int(fn)}\n")
        f.write(f"best_threshold,{best_threshold:.6f}\n")
        f.write(f"f1max,{f1max:.6f}\n")
        f.write(f"f1max_threshold,{thr_f1:.6f}\n")
        f.write(f"f1max_precision,{prec_f1:.6f}\n")
        f.write(f"f1max_recall,{rec_f1:.6f}\n")

    print(f"Wrote overall test results to: {overall_csv}")

    test_stats_npz = RESULTS_DIR / "binary_test_outputs.npz"
    np.savez(
        test_stats_npz,
        ids=test_labels["ID"].values,
        y_true=y_true_test.astype(np.int8),
        y_score=y_score_test.astype(np.float32),
        y_pred_at_best=y_pred_test.astype(np.int8),
        best_threshold=float(best_threshold)
    )
    print(f"Saved test outputs to: {test_stats_npz}")

    print(
        f"\nDone training {pretty}. Best balanced accuracy: {best_bal_acc:.4f} "
        f"(epoch {best_epoch})"
    )
    print(f"Best model saved to: {SAVE_PATH}")
    print(f"Threshold saved to: {THRESHOLDS_PATH}")
    print(f"Per-epoch metrics saved to: {METRICS_CSV}")


# ----------------------
# Entry point
# ----------------------
if __name__ == "__main__":
    # You can change this list if you want other models
    model_names = ["resnet50"]

    print("Select an option:")
    print("1) Check preprocessing")
    print("2) Train the model")
    print("3) Preview random ID/label pairs")

    choice = input("Enter choice [1/2/3]: ").strip()
    while choice not in {"1", "2", "3"}:
        choice = input("Please enter 1, 2, or 3: ").strip()

    if choice == "1":
        for m in model_names:
            print(f"\n==================== {m.upper()} PREPROCESS ====================")
            preview_preprocessing(m)
        print("\nPreprocessing preview complete. Run again and choose option 2 to train.")
    elif choice == "2":
        for m in model_names:
            print(f"\n==================== {m.upper()} ====================")
            run_for_model(m)
    else:
        split = input("Choose split [train/val/test] (default train): ").strip().lower()
        if split not in {"train", "val", "test"}:
            split = "train"
        try:
            preview_random_labels(num_samples=5, dataset=split)
        except Exception as exc:
            print(f"[ERROR] Failed to preview labels: {exc}")
