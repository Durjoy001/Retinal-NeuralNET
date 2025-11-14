# -*- coding: utf-8 -*-
"""
Generic external evaluation on Messidor-2 for models trained on RFMiD.

Works for CNNs, ViTs, Hybrids, and (optionally) VLMs via a simple adapter registry.

Endpoints (on gradable images only):
  1) Any-abnormal (primary): diagnosis > 0  [DR-only, ontology-matched; any DR grade 1-3]
  2) Per-class DR metric: same target (diagnosis > 0) for the single "DR" class
  3) Referable DR: (diagnosis >= 2)  -- enabled by default, disable with --no_referable_dr

Operating points:
  - Per-class thresholds: loaded from RFMiD validation ("optimal_thresholds.npy")
  - Any-abnormal threshold: defaults to DR threshold from RFMiD; override with --any_thr.

Usage example:
  python -m src.external_evaluation.eval_external_messidor2_generic \
    --model_name resnet50 \
    --checkpoint results/CNN/ResNet50/resnet50_rfmid_best.pth \
    --thresholds results/CNN/ResNet50/optimal_thresholds.npy \
    --rfmid_train_csv data/RFMiD_Challenge_Dataset/2. Groundtruths/a. RFMiD_Training_Labels.csv \
    --messidor_csv /path/to/messidor2_labels.csv \
    --images_dir /path/to/messidor2/images \
    --results_dir results/External/Messidor2/ResNet50

Notes:
- This script expects your checkpoints to contain a key "model_state_dict" compatible with the chosen builder.
- If your VLM checkpoints are organized differently, write a tiny adapter and register it below (see CLIP example).
"""

import warnings, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

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
from torchvision.transforms import InterpolationMode

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

try:
    import open_clip
    HAS_OPENCLIP = True
except ImportError:
    HAS_OPENCLIP = False

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split, GroupShuffleSplit
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# ============================
# Enhanced Preprocessing for External Evaluation
# ============================
def pad_to_square(img, fill=0):
    """Pad image to square with fill color."""
    w, h = img.size
    if w == h:
        return img
    s = max(w, h)
    new = Image.new("RGB", (s, s), color=(fill, fill, fill))
    new.paste(img, ((s - w) // 2, (s - h) // 2))
    return new

class CLAHETransform:
    """Contrast Limited Adaptive Histogram Equalization transform."""
    def __call__(self, img):
        if HAS_CV2:
            x = np.array(img)
            lab = cv2.cvtColor(x, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l2 = clahe.apply(l)
            lab2 = cv2.merge((l2, a, b))
            x2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
            return Image.fromarray(x2)
        else:
            # Simple fallback if cv2 is not present
            return ImageOps.equalize(img)

class BackgroundSubtraction(object):
    """
    Approximate background subtraction (Hemelings-style):
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

def build_eval_transform(img_size=448, use_clahe=True):
    """Build enhanced evaluation transform with square padding, optional CLAHE, and larger input."""
    transform_list = [
        transforms.Lambda(pad_to_square),
    ]
    if use_clahe:
        transform_list.append(CLAHETransform())
    transform_list.extend([
        transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    return transforms.Compose(transform_list)

def threshold_for_tpr(y_true, y_score, target_tpr=0.85):
    """Pick threshold that achieves target sensitivity (TPR)."""
    fpr, tpr, thr = roc_curve(y_true, y_score)
    i = np.argmin(np.abs(tpr - target_tpr))
    return float(thr[i]), float(tpr[i]), float(1 - fpr[i])

def threshold_for_youden_j(y_true, y_score):
    """Pick threshold that maximizes Youden's J (sensitivity + specificity - 1)."""
    fpr, tpr, thr = roc_curve(y_true, y_score)
    youden_j = tpr - fpr  # sensitivity + specificity - 1
    i = np.argmax(youden_j)
    return float(thr[i]), float(tpr[i]), float(1 - fpr[i]), float(youden_j[i])

def threshold_for_f1max(y_true, y_score):
    """Pick threshold that maximizes F1 score."""
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    f1_use = f1[:-1]
    if len(f1_use) == 0:
        return 0.5, 0.0, 0.0, 0.0
    i = int(np.nanargmax(f1_use))
    return float(thr[i]), float(f1_use[i]), float(prec[i]), float(rec[i])

# ============================
# Registry / Adapters
# ============================
class ModelAdapter:
    """Base adapter: build() returns nn.Module, transforms(is_train) returns a torchvision transform.
    Forward must output logits [B, C]."""
    def __init__(self, name: str, num_classes: int):
        self.name = name
        self.num_classes = num_classes

    def build(self) -> nn.Module:
        raise NotImplementedError

    def transforms(self, is_train: bool):
        raise NotImplementedError

class TimmBackboneWithHead(ModelAdapter):
    """Generic adapter for timm models with a standard head. Supports binary head for new architecture."""
    def __init__(self, timm_name: str, num_classes: int, drop_path: float = 0.2, head_dim: int = 256, fallback_names: list = None, include_binary_head: bool = False):
        super().__init__(timm_name, num_classes)
        self.timm_name = timm_name
        self.fallback_names = fallback_names or []
        self.drop_path = drop_path
        self.head_dim = head_dim
        self.include_binary_head = include_binary_head

    def build(self) -> nn.Module:
        # Try primary name, then fallbacks if provided
        candidates = [self.timm_name] + self.fallback_names
        backbone = None
        used_name = None
        for name in candidates:
            try:
                # Use updated drop_path_rate for vit_small if binary head
                drop_path_rate = 0.25 if (self.include_binary_head and "vit_small" in name) else self.drop_path
                backbone = timm.create_model(name, pretrained=True, num_classes=0, drop_path_rate=drop_path_rate)
                used_name = name
                break
            except Exception as e:
                if name == candidates[-1]:  # Last candidate failed
                    raise RuntimeError(f"Could not create timm model from any candidate: {candidates}. Last error: {e}")
                continue
        
        if used_name != self.timm_name:
            print(f"⚠️  Note: Using fallback timm model '{used_name}' instead of '{self.timm_name}'")
        
        in_f = backbone.num_features
        
        # Multi-label classifier head
        # Use updated dropout rates for new architecture
        dropout1 = 0.6 if self.include_binary_head else 0.5
        dropout2 = 0.4 if self.include_binary_head else 0.3
        multilabel_head = nn.Sequential(
            nn.Dropout(dropout1),
            nn.Linear(in_f, self.head_dim),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(self.head_dim, self.num_classes),
        )
        
        # Binary classifier head (for Disease_Risk - any abnormal vs normal)
        binary_head = None
        if self.include_binary_head:
            binary_head = nn.Sequential(
                nn.Dropout(dropout1),
                nn.Linear(in_f, self.head_dim),
                nn.ReLU(),
                nn.Dropout(dropout2),
                nn.Linear(self.head_dim, 1)  # Binary output
            )
        
        class Wrapper(nn.Module):
            def __init__(self, backbone, multilabel_classifier, binary_classifier=None):
                super().__init__()
                self.backbone = backbone
                self.classifier = multilabel_classifier  # For backward compatibility
                self.multilabel_classifier = multilabel_classifier
                self.binary_classifier = binary_classifier
            def forward(self, x):
                z = self.backbone(x)
                multilabel_out = self.multilabel_classifier(z)
                
                if self.binary_classifier is not None:
                    binary_out = self.binary_classifier(z)
                    return multilabel_out, binary_out
                else:
                    # If a timm model returns tuple (rare in eval), take first
                    if isinstance(multilabel_out, tuple):
                        multilabel_out = multilabel_out[0]
                    return multilabel_out
        return Wrapper(backbone, multilabel_head, binary_head)

    def transforms(self, is_train: bool):
        # Use the same fallback logic as build() for consistency
        candidates = [self.timm_name] + self.fallback_names
        m = None
        for name in candidates:
            try:
                m = timm.create_model(name, pretrained=True, num_classes=1)
                break
            except Exception:
                continue
        if m is None:
            raise RuntimeError(f"Could not create timm model for transforms from: {candidates}")
        cfg = resolve_data_config({}, model=m)
        return create_transform(**cfg, is_training=is_train)

class CNNAdapter(ModelAdapter):
    """Adapter for torchvision CNN models."""
    def __init__(self, model_name: str, num_classes: int):
        super().__init__(model_name, num_classes)
        self.model_name = model_name.lower()

    def build(self) -> nn.Module:
        """
        Build CNN model exactly as train_cnn_final.py does.
        For binary-only models (num_classes=1), architecture matches training exactly.
        """
        if self.model_name == "densenet121":
            model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            in_f = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_f, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, self.num_classes)
            )
        elif self.model_name == "resnet50":
            model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            in_f = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_f, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, self.num_classes)
            )
        elif self.model_name == "efficientnet_b3":
            model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
            in_f = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_f, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, self.num_classes)
            )
        elif self.model_name == "inception_v3":
            model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=False)
            in_f = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_f, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, self.num_classes)
            )
        else:
            raise ValueError(f"Unknown CNN model: {self.model_name}")
        
        # Note: CNNs are returned directly (no wrapper) to match training script checkpoint structure
        # The training script saves models without wrappers, so we must match that structure
        return model

    def transforms(self, is_train: bool):
        """
        Use Hemelings-style preprocessing to match train_cnn_final.py:
        1) Center crop to 1016 x 1016
        2) Resize to 224 x 224
        3) Gaussian background subtraction
        4) ToTensor + ImageNet normalization
        """
        return transforms.Compose([
            transforms.CenterCrop(1016),
            transforms.Resize((224, 224)),
            BackgroundSubtraction(kernel_size=31),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

class CLIPAdapter(ModelAdapter):
    """Adapter for CLIP models (requires open_clip)."""
    def __init__(self, model_name: str, num_classes: int, class_names: list):
        super().__init__(model_name, num_classes)
        self.class_names = class_names
        if not HAS_OPENCLIP:
            raise ImportError("open_clip required for CLIP models")

    def build(self) -> nn.Module:
        if "biomedclip" in self.name:
            clip_name = "ViT-L-14"
            base, _, _ = open_clip.create_model_and_transforms(clip_name, pretrained="biomedclip")
        else:
            if "336" in self.name:
                clip_name = "ViT-L-14-336"
            elif "b16" in self.name or "vit-b/16" in self.name:
                clip_name = "ViT-B-16"
            else:
                clip_name = "ViT-B-16"
            base, _, _ = open_clip.create_model_and_transforms(clip_name, pretrained="openai")
        
        for p in base.parameters():
            p.requires_grad = False
        for p in base.visual.parameters():
            p.requires_grad = True
        
        class CLIPWrapper(nn.Module):
            def __init__(self, base_model, class_names):
                super().__init__()
                self.base_model = base_model
                with torch.no_grad():
                    init = base_model.logit_scale.exp().clamp(1.0, 100.0).log()
                self.logit_scale = nn.Parameter(init.clone())
                self.class_bias = nn.Parameter(torch.zeros(len(class_names), dtype=torch.float32))
                self.class_names = class_names
                self.register_buffer("cached_text_embeds", torch.empty(0), persistent=True)
                self._prompts_cached = False

            def cache_prompts(self, device):
                if self._prompts_cached:
                    return
                prompts = [f"a fundus photograph with {name}" for name in self.class_names]
                with torch.no_grad():
                    text_tokens = open_clip.tokenize(prompts).to(device)
                    text_embeds = self.base_model.encode_text(text_tokens)
                    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                self.cached_text_embeds.resize_(text_embeds.shape)
                self.cached_text_embeds.copy_(text_embeds)
                self._prompts_cached = True

            def forward(self, x):
                self.cache_prompts(x.device)
                image_features = self.base_model.encode_image(x)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = image_features @ self.cached_text_embeds.T
                logits = logits * self.logit_scale.exp()
                logits = logits + self.class_bias
                return logits
        
        return CLIPWrapper(base, self.class_names)

    def transforms(self, is_train: bool):
        size = 336 if "336" in self.name else 224
        eval_tf = [
            transforms.Resize(size, interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                              std=(0.26862954, 0.26130258, 0.27577711)),
        ]
        return transforms.Compose(eval_tf)

class SigLIPAdapter(ModelAdapter):
    """Adapter for SigLIP models (timm-based)."""
    def __init__(self, model_name: str, num_classes: int, class_names: list):
        super().__init__(model_name, num_classes)
        self.class_names = class_names
        self.timm_name = None  # Resolved in build()

    def build(self) -> nn.Module:
        # Determine candidates based on model name
        if "large" in self.name:
            candidates = [
                "vit_large_patch16_siglip_384",
                "siglip_large_patch16_384",
                "vit_large_patch16_siglip_224",
                "siglip_large_patch16_224",
            ]
        else:
            candidates = [
                "vit_base_patch16_siglip_384",
                "siglip_base_patch16_384",
                "vit_base_patch16_siglip_224",
                "siglip_base_patch16_224",
            ]
        
        backbone = None
        for name in candidates:
            try:
                backbone = timm.create_model(name, pretrained=True, num_classes=0)
                self.timm_name = name
                break
            except Exception:
                continue
        
        if backbone is None:
            available = [m for m in timm.list_models("*siglip*")]
            raise RuntimeError(f"Could not create SigLIP model from {candidates}. Available: {available}")
        
        # Freeze backbone
        for p in backbone.parameters():
            p.requires_grad = False
        
        # Add classifier head
        in_features = backbone.num_features
        classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )
        
        class SigLIPWrapper(nn.Module):
            def __init__(self, b, c):
                super().__init__()
                self.backbone = b
                self.classifier = c
            def forward(self, x):
                features = self.backbone(x)
                return self.classifier(features)
        
        return SigLIPWrapper(backbone, classifier)

    def transforms(self, is_train: bool):
        if self.timm_name is None:
            # If build() hasn't been called, try to resolve it
            if "large" in self.name:
                candidates = ["vit_large_patch16_siglip_384", "siglip_large_patch16_384",
                             "vit_large_patch16_siglip_224", "siglip_large_patch16_224"]
            else:
                candidates = ["vit_base_patch16_siglip_384", "siglip_base_patch16_384",
                             "vit_base_patch16_siglip_224", "siglip_base_patch16_224"]
            for name in candidates:
                try:
                    tmp = timm.create_model(name, pretrained=True, num_classes=1)
                    self.timm_name = name
                    break
                except Exception:
                    continue
            if self.timm_name is None:
                raise RuntimeError(f"Could not resolve SigLIP model name for transforms from: {candidates}")
        else:
            tmp = timm.create_model(self.timm_name, pretrained=True, num_classes=1)
        
        cfg = resolve_data_config({}, model=tmp)
        return create_transform(**cfg, is_training=is_train)

# Registry: map model_name -> adapter factory
REGISTRY = {
    # CNNs
    "densenet121": lambda nc: CNNAdapter("densenet121", nc),
    "resnet50": lambda nc: CNNAdapter("resnet50", nc),
    "efficientnet_b3": lambda nc: CNNAdapter("efficientnet_b3", nc),
    "inception_v3": lambda nc: CNNAdapter("inception_v3", nc),
    
    # ViTs (timm)
    "swin_tiny": lambda nc: TimmBackboneWithHead("swin_tiny_patch4_window7_224", nc),
    "vit_small": lambda nc: TimmBackboneWithHead("vit_small_patch16_224", nc),
    "deit_small": lambda nc: TimmBackboneWithHead("deit_small_patch16_224", nc),
    "crossvit_small": lambda nc: TimmBackboneWithHead("crossvit_15_240", nc),
    
    # Hybrids (timm)
    "coatnet0": lambda nc: TimmBackboneWithHead("coatnet_0_rw_224", nc),
    "maxvit_tiny": lambda nc: TimmBackboneWithHead("maxvit_tiny_rw_224", nc, fallback_names=["maxvit_tiny_tf_224"]),  # try rw first, fallback to tf
}

# Note: CLIP and SigLIP need class_names, so they're registered differently in main()

# ============================
# Helpers
# ============================
DR_ALIASES = ["dr", "diabetic_retinopathy", "diabetic retinopathy", "retinopathy"]
DME_ALIASES = ["dme", "diabetic_macular_edema", "diabetic macular edema", "macular_edema", "macular edema"]

def _find_label_index(label_columns, aliases):
    cols_lower = [c.lower() for c in label_columns]
    for alias in aliases:
        alias = alias.lower()
        for i, c in enumerate(cols_lower):
            if c == alias or alias in c:
                return i
    return None

def _best_existing_path(stem, exts=(".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")):
    # First check if the path already exists (id_code might already include extension)
    if stem.exists():
        return stem
    
    # If not, try adding each extension
    for ext in exts:
        p = stem.with_suffix(ext)
        if p.exists():
            return p
    
    # Also try without any extension changes (in case stem already has extension)
    # Check if any file with similar name exists
    parent = stem.parent
    name = stem.name
    for ext in exts:
        # Try exact name with extension
        p = parent / (name + ext)
        if p.exists():
            return p
        # Try name without current extension + new extension
        name_no_ext = stem.stem
        p = parent / (name_no_ext + ext)
        if p.exists():
            return p
    
    # If still not found, search in subdirectories (common for Messidor-2)
    import glob
    parent = stem.parent
    name_pattern = stem.stem  # Name without extension
    for ext in exts:
        # Search recursively for files matching the pattern
        pattern = str(parent / "**" / f"{name_pattern}*{ext}")
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return Path(matches[0])
        # Also try with full name (in case it already has extension)
        pattern = str(parent / "**" / f"{name}*{ext}")
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return Path(matches[0])
    
    raise FileNotFoundError(f"Image not found for ID: {stem.name}. Tried path: {stem}. Tried extensions: {exts}. Searched in: {parent}")

# ============================
# Metrics
# ============================
def safe_auc(y_true, y_score):
    """Compute AUC with safety check for single-class cases."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")

def safe_confusion(y_true, y_pred):
    """Compute confusion matrix with safety checks using boolean masks."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp, tn, fp, fn

def f1_at_threshold(y_true, y_score, thr):
    y_pred = (y_score >= thr).astype(np.int32)
    tp, tn, fp, fn = safe_confusion(y_true, y_pred)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1, tp, tn, fp, fn

def f1_max(y_true, y_score):
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    f1_use = f1[:-1]
    if len(f1_use) == 0:
        return 0.0, 0.5, 0.0, 0.0
    i = int(np.nanargmax(f1_use))
    return float(f1_use[i]), float(thr[i]), float(prec[i]), float(rec[i])

def pick_threshold_for_specificity(y_true, y_score, target_spec=0.8):
    """Pick threshold that achieves target specificity (or closest)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_true, y_score)
    spec = 1 - fpr
    idx = int(np.argmin(np.abs(spec - target_spec)))
    return float(thr[idx]), float(spec[idx]), float(tpr[idx])

# ============================
# Dataset
# ============================
class Messidor2Dataset(Dataset):
    """Expects CSV with columns: id_code, diagnosis, adjudicated_dme, adjudicated_gradable"""
    def __init__(self, images_dir: Path, df: pd.DataFrame, transform):
        self.images_dir = Path(images_dir)
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        id_code = str(row["id_code"])
        # id_code might already include extension, or might not
        # Try direct path first, then try with extensions
        stem = self.images_dir / id_code
        img_path = _best_existing_path(stem)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image at {img_path} for ID {id_code}: {e}")
        if self.transform:
            img = self.transform(img)
        return img, id_code

# ============================
# Main
# ============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True, help="Model name from registry")
    ap.add_argument("--checkpoint", required=True, help="Path to RFMiD best-AUC checkpoint .pth")
    ap.add_argument("--thresholds", required=False, default=None, help="Path to RFMiD per-class thresholds .npy (optional for binary-only CNN with threshold selection)")
    ap.add_argument("--rfmid_train_csv", required=True, help="RFMiD training labels CSV (to recover label order)")
    ap.add_argument("--messidor_csv", required=True, help="Messidor-2 labels CSV")
    ap.add_argument("--images_dir", required=True, help="Messidor-2 images directory")
    ap.add_argument("--results_dir", required=True, help="Output directory for metrics")
    ap.add_argument("--any_thr", type=float, default=None, help="Override any-abnormal threshold (defaults to DR threshold)")
    ap.add_argument("--use_f1max_threshold", action="store_true", help="Use F1max threshold instead of RFMiD threshold (better for domain shift)")
    ap.add_argument("--use_dr_head", action="store_true", help="Force use of DR head from multilabel classifier instead of binary head (for comparison). Default: use binary head if available.")
    ap.add_argument("--target_specificity", type=float, default=None, help="Use threshold that achieves target specificity (e.g., 0.8 for 80% specificity). Overrides --use_f1max_threshold if set.")
    ap.add_argument("--target_sensitivity", type=float, default=None, help="Use threshold that achieves target sensitivity on Messidor-2 validation split (e.g., 0.85 for 85% sensitivity). Best for domain adaptation.")
    ap.add_argument("--threshold_method", type=str, default="youden", choices=["youden", "f1max", "sensitivity", "specificity"], 
                    help="Method for threshold selection on validation split: 'youden' (max J, default), 'f1max' (max F1), 'sensitivity' (target TPR), 'specificity' (target TNR)")
    ap.add_argument("--val_split", type=float, default=0.2, help="Fraction of Messidor-2 data to use for validation (for threshold selection). Default: 0.2")
    ap.add_argument("--eval_img_size", type=int, default=448, help="Input image size for evaluation (default: 448, larger than training)")
    ap.add_argument("--use_tta", action="store_true", default=False, help="Use test-time augmentation (average original + horizontal flip). Default: False")
    ap.add_argument("--no_tta", action="store_false", dest="use_tta", help="Disable test-time augmentation (faster evaluation)")
    ap.add_argument("--tta_num_augments", type=int, default=5, help="Number of TTA augmentations (default: 5, includes flip, rotations, scales)")
    ap.add_argument("--disable_clahe", action="store_true", help="Disable CLAHE preprocessing (use standard transforms)")
    ap.add_argument("--use_temperature_scaling", action="store_true", help="Apply temperature scaling calibration on validation split")
    ap.add_argument("--adapt_bn", action="store_true", help="Adapt BatchNorm statistics on Messidor-2 (no labels needed)")
    ap.add_argument("--temp_scaling_subset", type=int, default=None, help="Use subset of training data for temperature scaling (faster). Default: use all training data")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    ap.add_argument("--pin_memory", action="store_true", help="Pin memory for faster GPU transfer")
    ap.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility")
    ap.add_argument("--no_referable_dr", dest="with_referable_dr", action="store_false", default=True,
                    help="Disable referable DR metrics (default: enabled)")
    args = ap.parse_args()
    
    # Set reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load RFMiD schema and thresholds
    rfmid_train = pd.read_csv(args.rfmid_train_csv)
    label_columns = [c for c in rfmid_train.columns if c != "ID"]
    
    # Load thresholds (optional for binary-only CNN with threshold selection)
    if args.thresholds is None or not Path(args.thresholds).exists():
        if args.target_specificity is not None or args.target_sensitivity is not None or args.threshold_method in ["youden", "f1max"]:
            # Threshold selection will be used, so we can create a dummy threshold array
            print(f"⚠️  Threshold file not provided or doesn't exist: {args.thresholds}")
            print(f"   Using threshold selection on Messidor-2 validation split instead")
            thresholds = np.array([0.5])  # Dummy threshold, will be replaced by threshold selection
        else:
            raise ValueError(f"Threshold file required: {args.thresholds}. Provide --thresholds or use threshold selection (--target_specificity, --target_sensitivity, or --threshold_method)")
    else:
        thresholds = np.load(args.thresholds)

    
    # Check if checkpoint has binary head (affects expected threshold count)
    ckpt_preview = torch.load(args.checkpoint, map_location="cpu")
    state_preview = ckpt_preview.get("model_state_dict", ckpt_preview)
    has_binary_head_in_ckpt = any(k.startswith("binary_classifier.") for k in state_preview.keys())
    
    # Check if it's a binary-only CNN model (pure binary, not multi-label + binary)
    # Binary-only models have last layer with 1 output (e.g., fc.4.weight with shape [1, 256])
    is_binary_only_cnn = False
    if args.model_name.lower() in ["densenet121", "resnet50", "efficientnet_b3", "inception_v3"]:
        # Check last classifier layer
        for k in reversed(list(state_preview.keys())):
            if "weight" in k and ("fc" in k or "classifier" in k):
                weight_shape = state_preview[k].shape
                if weight_shape[0] == 1:  # Binary output
                    is_binary_only_cnn = True
                    print(f"✅ Detected binary-only CNN model (last layer: {k}, shape: {weight_shape})")
                break
    
    # Thresholds file handling:
    # - Binary-only CNN: single threshold (optimal_threshold_spec80.npy)
    # - Multi-label + binary head: thresholds for disease classes (excludes Disease_Risk)
    # - Multi-label only: thresholds for all classes
    if is_binary_only_cnn:
        # Binary-only model: expect single threshold
        if len(thresholds) != 1:
            print(f"⚠️  Warning: Binary-only CNN model expects 1 threshold, but got {len(thresholds)}. Using first threshold.")
            thresholds = np.array([thresholds[0] if len(thresholds) > 0 else 0.5])
        expected_threshold_count = 1
    else:
        expected_threshold_count = len(label_columns) - (1 if ("Disease_Risk" in label_columns and has_binary_head_in_ckpt) else 0)
        if len(thresholds) != expected_threshold_count:
            raise ValueError(f"Threshold length mismatch: expected {expected_threshold_count} thresholds (for {len(label_columns)} label columns, "
                            f"{'excluding Disease_Risk' if has_binary_head_in_ckpt and 'Disease_Risk' in label_columns else 'including all'}), "
                            f"but got {len(thresholds)} thresholds in file.")

    # For binary-only CNN, we don't need DR index (model outputs binary directly)
    if is_binary_only_cnn:
        dr_idx = None  # Not needed for binary-only model
        dr_idx_in_multilabel = None
        tau_dr = float(thresholds[0])  # Single threshold for binary model
        print(f"✅ Binary-only CNN: Using single threshold {tau_dr:.6f} from optimal_threshold_spec80.npy")
    else:
        dr_idx = _find_label_index(label_columns, DR_ALIASES)
        if dr_idx is None:
            raise ValueError(f"Could not find DR column in RFMiD labels. Checked aliases: {DR_ALIASES}")

        # Map dr_idx to thresholds array index and multilabel probabilities index
        # If Disease_Risk is excluded from multilabel head (binary head model), adjust index
        if "Disease_Risk" in label_columns and has_binary_head_in_ckpt:
            # Find Disease_Risk index in label_columns
            drisk_idx_in_labels = label_columns.index("Disease_Risk")
            # If DR comes after Disease_Risk, subtract 1 from dr_idx
            if dr_idx > drisk_idx_in_labels:
                dr_idx_in_multilabel = dr_idx - 1
            else:
                dr_idx_in_multilabel = dr_idx
        else:
            dr_idx_in_multilabel = dr_idx
        
        # For thresholds, use the same mapping
        dr_idx_in_thresholds = dr_idx_in_multilabel
        tau_dr = float(thresholds[dr_idx_in_thresholds])
    # Default: use threshold_method (Youden's J) on validation split
    # All threshold adaptation methods will compute threshold on validation split to prevent test leakage
    if args.any_thr is not None:
        tau_any = float(args.any_thr)
    else:
        # Will be set after computing threshold on validation split
        tau_any = None  # Placeholder, will be updated

    # Load Messidor-2 CSV and filter gradable
    mdf = pd.read_csv(args.messidor_csv)
    expected_cols = {"id_code", "diagnosis", "adjudicated_dme", "adjudicated_gradable"}
    missing = expected_cols - set(mdf.columns.str.lower())
    lower_map = {c.lower(): c for c in mdf.columns}
    def col(name): return lower_map[name] if name in lower_map else name
    if missing:
        raise ValueError(f"Messidor CSV missing columns: {missing}")

    mdf = mdf.rename(columns={
        col("id_code"): "id_code",
        col("diagnosis"): "diagnosis",
        col("adjudicated_dme"): "adjudicated_dme",
        col("adjudicated_gradable"): "adjudicated_gradable"
    })
    mdf = mdf[(mdf["adjudicated_gradable"] == 1) | (mdf["adjudicated_gradable"] == True)].copy()
    mdf.reset_index(drop=True, inplace=True)
    print(f"Using {len(mdf)} gradable Messidor-2 images.")

    # Split into train/val for threshold selection (ALWAYS use split if any threshold adaptation is requested)
    # This prevents test leakage - thresholds must be computed on validation split only
    use_val_split = (args.target_sensitivity is not None or 
                     args.threshold_method in ["youden", "f1max"] or
                     args.target_specificity is not None or
                     args.use_f1max_threshold or
                     args.use_temperature_scaling or
                     args.adapt_bn)
    
    if use_val_split:
        # Patient-level split: extract patient ID from id_code (assumes format like "patient_eye" or similar)
        # Try to extract patient ID - common patterns: "patient_001_left", "001_left", etc.
        def extract_patient_id(id_code):
            """Extract patient ID from image ID code."""
            id_str = str(id_code)
            # Try splitting by common delimiters
            for sep in ['_', '-', ' ']:
                parts = id_str.split(sep)
                if len(parts) > 1:
                    # Usually patient ID is the first part
                    return parts[0]
            # Fallback: use first 6-8 characters as patient ID
            return id_str[:8]
        
        mdf["patient_id"] = mdf["id_code"].apply(extract_patient_id)
        
        # Use GroupShuffleSplit to ensure patient-level grouping
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=args.val_split, random_state=args.seed)
        train_idx, test_idx = next(gss.split(mdf, groups=mdf["patient_id"], 
                                             y=(mdf["diagnosis"].astype(int) > 0)))  # Stratify by DR presence
        
        mdf_train = mdf.iloc[train_idx].reset_index(drop=True)
        mdf_test = mdf.iloc[test_idx].reset_index(drop=True)
        
        print(f"  Split (patient-level): {len(mdf_train)} train (for threshold/calibration), {len(mdf_test)} test (for final evaluation)")
        print(f"  Unique patients: {mdf_train['patient_id'].nunique()} train, {mdf_test['patient_id'].nunique()} test")
    else:
        mdf_train = None
        mdf_test = mdf  # Use all data for evaluation (only if no threshold adaptation)
    
    # Ground truths (for test set)
    y_any_abnormal = (mdf_test["diagnosis"].astype(int) > 0).astype(np.int32).values
    y_dr = (mdf_test["diagnosis"].astype(int) > 0).astype(np.int32).values
    y_dr_referr = (mdf_test["diagnosis"].astype(int) >= 2).astype(np.int32).values if args.with_referable_dr else None

    # Check checkpoint first to see if it has binary head
    # (Already loaded above for threshold check, but reload here for model building)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    
    # Check if checkpoint has binary_classifier (new model with binary head)
    # (Already computed above, but recompute here for clarity)
    has_binary_head_in_ckpt = any(k.startswith("binary_classifier.") for k in state.keys())
    
    # Build model via adapter
    # For binary-only CNN: num_classes = 1 (binary output)
    # For multi-label + binary head: exclude Disease_Risk from multilabel head
    # For multi-label only: use all classes
    if is_binary_only_cnn:
        num_classes = 1  # Binary-only model
        print(f"✅ Binary-only CNN model detected - building with num_classes=1")
    elif "Disease_Risk" in label_columns:
        num_classes = len(label_columns) - 1  # Exclude Disease_Risk from multilabel head
        if has_binary_head_in_ckpt:
            print(f"✅ Binary Disease_Risk head detected in checkpoint - will use for Messidor-2 (DR-only dataset)")
    else:
        num_classes = len(label_columns)
    
    model_name_lower = args.model_name.lower()
    
    # Special handling for CLIP and SigLIP (need class_names)
    if "clip" in model_name_lower or "biomedclip" in model_name_lower:
        if not HAS_OPENCLIP:
            raise ImportError("open_clip required for CLIP models")
        adapter = CLIPAdapter(model_name_lower, num_classes, label_columns)
    elif "siglip" in model_name_lower:
        adapter = SigLIPAdapter(model_name_lower, num_classes, label_columns)
    elif model_name_lower in REGISTRY:
        # For ViT models, check if we need binary head
        if model_name_lower in ["swin_tiny", "vit_small", "deit_small", "crossvit_small"] and has_binary_head_in_ckpt:
            # Map model names to timm identifiers
            timm_name_map = {
                "swin_tiny": "swin_tiny_patch4_window7_224",
                "vit_small": "vit_small_patch16_224",
                "deit_small": "deit_small_patch16_224",
                "crossvit_small": "crossvit_15_240",
            }
            adapter = TimmBackboneWithHead(
                timm_name_map[model_name_lower],
                num_classes,
                drop_path=0.25 if model_name_lower == "vit_small" else 0.2,
                include_binary_head=True
            )
        else:
            adapter = REGISTRY[model_name_lower](num_classes)
    else:
        raise ValueError(f"Unknown model: {args.model_name}. Available: {list(REGISTRY.keys())}")

    model = adapter.build().to(device)
    
    # Fix key mismatches (checkpoint may have different prefixes)
    model_keys = set(model.state_dict().keys())
    state_keys = set(state.keys())
    
    # If keys don't match, try remapping common patterns
    if not model_keys.intersection(state_keys):
        # For CNN models: checkpoint may have "backbone." or "m." prefix
        if model_name_lower in ["densenet121", "resnet50", "efficientnet_b3", "inception_v3"]:
            if any(k.startswith("backbone.") for k in state_keys):
                state = {k.replace("backbone.", ""): v for k, v in state.items()}
            elif any(k.startswith("m.") for k in state_keys):
                state = {k.replace("m.", ""): v for k, v in state.items()}
        # For ViT/Hybrid: checkpoint uses "backbone." and "classifier.", eval uses same (should match)
        # But if there's a mismatch with "b." and "h.", remap them
        elif model_name_lower in ["swin_tiny", "vit_small", "deit_small", "crossvit_small", "coatnet0", "maxvit_tiny"]:
            if any(k.startswith("b.") for k in state_keys):
                state = {k.replace("b.", "backbone."): v for k, v in state.items()}
            if any(k.startswith("h.") for k in state_keys):
                state = {k.replace("h.", "classifier."): v for k, v in state.items()}
    
    # CLIP and SigLIP adapters may not match RFMiD checkpoint format; handle gracefully
    if "clip" in model_name_lower or "biomedclip" in model_name_lower or "siglip" in model_name_lower:
        # VLM adapters may not load your RFMiD heads; use the adapter weights only
        vlm_type = "CLIP" if "clip" in model_name_lower or "biomedclip" in model_name_lower else "SigLIP"
        try:
            missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
            if missing_keys:
                print(f"⚠️  Warning: Missing keys in {vlm_type} checkpoint: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"⚠️  Warning: Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"⚠️  Warning: Unexpected keys in {vlm_type} checkpoint: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"⚠️  Warning: Unexpected keys: {unexpected_keys}")
        except Exception as e:
            print(f"⚠️  Skipping strict {vlm_type} state dict load (checkpoint format may not match); using adapter only.")
            print(f"   Error: {e}")
    else:
        missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
        if missing_keys:
            print(f"⚠️  Warning: Missing keys: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"⚠️  Warning: Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"⚠️  Warning: Unexpected keys: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"⚠️  Warning: Unexpected keys: {unexpected_keys}")
    
    model.eval()

    # Auto-adjust image size for ViT models (trained with 224x224)
    # ViT models require exact input size match
    eval_img_size = args.eval_img_size
    if model_name_lower in ["swin_tiny", "vit_small", "deit_small", "crossvit_small"]:
        if eval_img_size != 224:
            print(f"⚠️  Warning: ViT models require 224x224 input (trained size). Overriding eval_img_size from {eval_img_size} to 224.")
            eval_img_size = 224
    
    # Use adapter's transforms for CNN models (Hemelings-style preprocessing to match training)
    # For other models (ViT, Hybrid, VLM), use enhanced eval transform
    if model_name_lower in ["densenet121", "resnet50", "efficientnet_b3", "inception_v3"]:
        # CNN models: use Hemelings-style preprocessing (matches train_cnn_final.py)
        tfm = adapter.transforms(is_train=False)
        print(f"✅ Using Hemelings-style preprocessing for CNN model (matches training):")
        print(f"   Center crop 1016 → Resize 224 → Background subtraction → Normalize")
    else:
        # ViT, Hybrid, VLM models: use enhanced eval transform
        use_clahe = not args.disable_clahe
        tfm = build_eval_transform(img_size=eval_img_size, use_clahe=use_clahe)
        clahe_str = "with CLAHE" if use_clahe else "without CLAHE"
        print(f"Using enhanced eval transform: {eval_img_size}x{eval_img_size} {clahe_str} and square padding")
    
    # Create datasets
    ds_test = Messidor2Dataset(Path(args.images_dir), mdf_test, tfm)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=args.pin_memory)
    
    # Create train dataset if needed for threshold selection, BN adaptation, or temperature scaling
    if use_val_split:
        ds_train = Messidor2Dataset(Path(args.images_dir), mdf_train, tfm)
        dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=args.pin_memory)
    
    # Step 1: Adaptive BatchNorm (update BN stats on Messidor-2, no labels needed)
    if args.adapt_bn:
        print("\n📊 Adapting BatchNorm statistics on Messidor-2...")
        model.train()  # Set to train mode so BN uses batch statistics
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.track_running_stats = True
                module.momentum = 0.1  # Faster adaptation
        
        num_bn_batches = min(50, len(dl_train))
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(dl_train):
                if batch_idx >= num_bn_batches:
                    break
                images = images.to(device)
                _ = model(images)  # Forward pass updates BN stats
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {batch_idx + 1}/{num_bn_batches} batches...")
        model.eval()
        print("✅ BN statistics adapted")

    # Inference with TTA (test-time augmentation: average original + horizontal flip)
    use_amp = torch.cuda.is_available() or (torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False)
    dev_type = "cuda" if torch.cuda.is_available() else ("mps" if use_amp and hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    
    def get_tta_transforms(xb):
        """Generate gentle TTA transforms: small rotations (±5°) and horizontal flip."""
        from torchvision.transforms.functional import rotate
        transforms_list = []
        
        angles = [0, 5, -5]  # Small rotations only (±5°)
        flips = [False, True]  # Original and horizontal flip
        
        for angle in angles:
            if angle == 0:
                xa = xb
            else:
                # Rotate entire batch (more efficient)
                xa = rotate(xb, angle, interpolation=InterpolationMode.BILINEAR)
            
            for do_flip in flips:
                if do_flip:
                    xf = torch.flip(xa, dims=[-1])
                else:
                    xf = xa
                transforms_list.append(xf)
                
                if len(transforms_list) >= args.tta_num_augments:
                    break
            if len(transforms_list) >= args.tta_num_augments:
                break
        
        return transforms_list[:args.tta_num_augments]
    
    def run_inference(dataloader, dataset_name="test", temperature=1.0):
        """Run inference with optional TTA and temperature scaling."""
        all_logits = []
        all_binary_logits = [] if (has_binary_head_in_ckpt or is_binary_only_cnn) else None
        all_ids = []
        total_batches = len(dataloader)
        print(f"Running inference on {len(dataloader.dataset)} {dataset_name} images ({total_batches} batches)...")
        if args.use_tta:
            print(f"  Using TTA: {args.tta_num_augments} augmentations (flip, rotations, scales)")
        if temperature != 1.0:
            print(f"  Using temperature scaling: T={temperature:.3f}")
        
        with torch.inference_mode(), torch.autocast(device_type=dev_type, enabled=(dev_type != "cpu")):
            for batch_idx, (xb, ids) in enumerate(dataloader, 1):
                xb = xb.to(device)
                
                # TTA: average over multiple transforms
                if args.use_tta:
                    tta_transforms = get_tta_transforms(xb)
                    logits_list = []
                    binary_logits_list = [] if (has_binary_head_in_ckpt or is_binary_only_cnn) else None
                    for xb_aug in tta_transforms:
                        outputs = model(xb_aug)
                        # Handle binary-only CNN (single output [B, 1])
                        if is_binary_only_cnn:
                            binary_logits = outputs  # Binary-only CNN outputs [B, 1] directly
                            if binary_logits_list is not None:
                                binary_logits_list.append(binary_logits)
                            logits_list.append(binary_logits)  # For compatibility
                        # Handle tuple output (multi-label + binary head)
                        elif isinstance(outputs, tuple):
                            multilabel_logits, binary_logits = outputs
                            if binary_logits_list is not None:
                                binary_logits_list.append(binary_logits)
                            logits_list.append(multilabel_logits)
                        # Handle single output (multi-label only)
                        else:
                            multilabel_logits = outputs
                            if hasattr(multilabel_logits, "logits"):
                                multilabel_logits = multilabel_logits.logits
                            elif isinstance(multilabel_logits, (tuple, list)):
                                multilabel_logits = multilabel_logits[0]
                            logits_list.append(multilabel_logits)
                    logits_avg = torch.stack(logits_list).mean(dim=0)
                    if binary_logits_list is not None:
                        binary_logits_avg = torch.stack(binary_logits_list).mean(dim=0)
                    else:
                        binary_logits_avg = None
                else:
                    outputs = model(xb)
                    # Handle binary-only CNN (single output [B, 1])
                    if is_binary_only_cnn:
                        logits_avg = outputs  # Binary-only CNN outputs [B, 1] directly
                        binary_logits_avg = outputs
                    # Handle tuple output (multi-label + binary head)
                    elif isinstance(outputs, tuple):
                        logits_avg, binary_logits_avg = outputs
                    # Handle single output (multi-label only)
                    else:
                        logits_avg = outputs
                        binary_logits_avg = None
                        if hasattr(logits_avg, "logits"):
                            logits_avg = logits_avg.logits
                        elif isinstance(logits_avg, (tuple, list)):
                            logits_avg = logits_avg[0]
                
                # Apply temperature scaling
                logits_scaled = logits_avg / temperature
                all_logits.append(logits_scaled.cpu())
                
                if binary_logits_avg is not None:
                    binary_logits_scaled = binary_logits_avg / temperature
                    if all_binary_logits is not None:
                        all_binary_logits.append(binary_logits_scaled.cpu())
                
                all_ids.extend(ids)
                
                if batch_idx % 10 == 0 or batch_idx == total_batches:
                    print(f"  Processed {batch_idx}/{total_batches} batches ({len(all_ids)} images)...", flush=True)
        
        all_logits = torch.cat(all_logits, dim=0)
        all_probs = torch.sigmoid(all_logits).numpy()
        
        # If binary-only CNN or binary head available, use it for Messidor-2 (DR-only, so binary = any abnormal = DR present)
        if is_binary_only_cnn:
            # Binary-only CNN: output is already binary [N, 1]
            binary_probs = all_probs.squeeze()  # [N, 1] -> [N]
            print(f"✅ Inference complete: {len(binary_probs)} images, using binary-only CNN output for Messidor-2 (DR-only dataset).")
            return all_probs, all_ids, binary_probs
        elif has_binary_head_in_ckpt and all_binary_logits is not None:
            all_binary_logits = torch.cat(all_binary_logits, dim=0)
            binary_probs = torch.sigmoid(all_binary_logits).numpy().squeeze()
            print(f"✅ Inference complete: {all_probs.shape[0]} images, using binary Disease_Risk head for Messidor-2 (DR-only dataset).")
            return all_probs, all_ids, binary_probs
        else:
            print(f"✅ Inference complete: {all_probs.shape[0]} images, {all_probs.shape[1]} classes.")
            return all_probs, all_ids, None
    
    # Step 2: Temperature scaling (fit on validation split, apply to test)
    # True one-parameter temperature scaling: minimize NLL w.r.t. T
    # NOTE: Skip TTA during temperature scaling for speed (TTA not needed for calibration)
    temperature = 1.0
    if args.use_temperature_scaling and use_val_split:
        print("\n📊 Fitting temperature scaling on validation split...")
        print("  ⚡ Skipping TTA for speed (not needed for calibration)")
        # Get raw logits (without temperature scaling) for fitting
        # Optionally use subset for faster calibration
        subset_indices = None
        if args.temp_scaling_subset is not None and args.temp_scaling_subset < len(dl_train.dataset):
            print(f"  Using subset of {args.temp_scaling_subset} images for faster calibration...")
            from torch.utils.data import Subset
            np.random.seed(args.seed)  # For reproducibility
            subset_indices = np.random.choice(len(dl_train.dataset), args.temp_scaling_subset, replace=False)
            dl_train_subset = DataLoader(Subset(dl_train.dataset, subset_indices), 
                                        batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers, pin_memory=args.pin_memory)
            dl_train_for_temp = dl_train_subset
            total_batches = len(dl_train_subset)
        else:
            dl_train_for_temp = dl_train
            total_batches = len(dl_train)
        
        print(f"  Running inference on {len(dl_train_for_temp.dataset)} images ({total_batches} batches)...")
        all_logits_train_raw = []
        all_binary_logits_train_raw = [] if (has_binary_head_in_ckpt or is_binary_only_cnn) else None
        with torch.inference_mode(), torch.autocast(device_type=dev_type, enabled=(dev_type != "cpu")):
            for batch_idx, (xb, _) in enumerate(dl_train_for_temp, 1):
                xb = xb.to(device)
                # Skip TTA for temperature scaling - just need raw logits for calibration
                outputs = model(xb)
                # Handle binary-only CNN (single output [B, 1])
                if is_binary_only_cnn:
                    binary_logits = outputs  # Binary-only CNN outputs [B, 1] directly
                    if all_binary_logits_train_raw is not None:
                        all_binary_logits_train_raw.append(binary_logits.cpu())
                    all_logits_train_raw.append(binary_logits.cpu())
                # Handle tuple output (multi-label + binary head)
                elif isinstance(outputs, tuple):
                    multilabel_logits, binary_logits = outputs
                    if all_binary_logits_train_raw is not None:
                        all_binary_logits_train_raw.append(binary_logits.cpu())
                    all_logits_train_raw.append(multilabel_logits.cpu())
                # Handle single output (multi-label only)
                else:
                    logits = outputs
                    if hasattr(logits, "logits"):
                        logits = logits.logits
                    elif isinstance(logits, (tuple, list)):
                        logits = logits[0]
                    all_logits_train_raw.append(logits.cpu())
                
                if batch_idx % 20 == 0 or batch_idx == total_batches:
                    print(f"  Processed {batch_idx}/{total_batches} batches ({batch_idx * args.batch_size} images)...", flush=True)
        
        print("  Concatenating logits...", flush=True)
        all_logits_train_raw = torch.cat(all_logits_train_raw, dim=0)
        # Get corresponding labels for the subset used
        if subset_indices is not None:
            y_dr_train = (mdf_train.iloc[subset_indices]["diagnosis"].astype(int) > 0).astype(np.int32).values
        else:
            y_dr_train = (mdf_train["diagnosis"].astype(int) > 0).astype(np.int32).values
        
        # Use binary logits if available (better calibrated for any abnormal = DR)
        if is_binary_only_cnn:
            # Binary-only CNN: use the binary logits directly
            if all_binary_logits_train_raw is not None:
                all_binary_logits_train_raw = torch.cat(all_binary_logits_train_raw, dim=0)
                dr_logits_train = all_binary_logits_train_raw.squeeze()  # [N, 1] -> [N]
            else:
                # Fallback: use all_logits_train_raw (which is already binary for binary-only CNN)
                dr_logits_train = all_logits_train_raw.squeeze()  # [N, 1] -> [N]
            print(f"  Using binary-only CNN logits for temperature scaling")
        elif args.use_dr_head:
            # Force use of DR head logits
            dr_logits_train = all_logits_train_raw[:, dr_idx_in_multilabel]  # Keep as tensor
            print(f"  Using DR head logits for temperature scaling (user-specified)")
        elif has_binary_head_in_ckpt and all_binary_logits_train_raw is not None:
            all_binary_logits_train_raw = torch.cat(all_binary_logits_train_raw, dim=0)
            dr_logits_train = all_binary_logits_train_raw.squeeze()  # Binary head: [N, 1] -> [N]
            print(f"  Using binary Disease_Risk head logits for temperature scaling")
        else:
            dr_logits_train = all_logits_train_raw[:, dr_idx_in_multilabel]  # Keep as tensor
        
        print("  Optimizing temperature parameter...", flush=True)
        # True temperature scaling: optimize T by minimizing NLL
        T = torch.ones(1, requires_grad=True, device=device)
        y_val_t = torch.tensor(y_dr_train, dtype=torch.float32, device=device)
        logits_val_t = dr_logits_train.to(device)
        
        opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50, line_search_fn='strong_wolfe')
        
        def nll():
            opt.zero_grad()
            z = logits_val_t / T.clamp_min(1e-3)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y_val_t)
            loss.backward()
            return loss
        
        try:
            opt.step(nll)
            temperature = float(T.clamp(0.1, 10.0).item())
            print(f"  ✅ Fitted temperature: T={temperature:.3f}")
        except Exception as e:
            print(f"  ⚠️  Temperature scaling optimization failed: {e}")
            print(f"  Using default temperature: T=1.0")
            temperature = 1.0
    
    # Run inference on test set (with temperature scaling if fitted)
    all_probs, all_ids, binary_probs = run_inference(dl_test, "test", temperature=temperature)
    
    # If using threshold selection, also run inference on train set
    # CRITICAL: Always compute thresholds on validation split to prevent test leakage
    if use_val_split:
        all_probs_train, all_ids_train, binary_probs_train = run_inference(dl_train, "train (for threshold selection)", temperature=temperature)
        if is_binary_only_cnn:
            # Binary-only CNN: use binary probabilities directly
            dr_scores_train = binary_probs_train if binary_probs_train is not None else all_probs_train.squeeze()
        elif args.use_dr_head:
            # Force use of DR head
            dr_scores_train = all_probs_train[:, dr_idx_in_multilabel]
        elif has_binary_head_in_ckpt and binary_probs_train is not None:
            # Use binary head for threshold selection (better calibrated for any abnormal = DR)
            dr_scores_train = binary_probs_train
        else:
            dr_scores_train = all_probs_train[:, dr_idx_in_multilabel]
        y_dr_train = (mdf_train["diagnosis"].astype(int) > 0).astype(np.int32).values

    # Scores for endpoints (DR-only for Messidor-2)
    # For binary-only CNN models (trained with train_cnn_final.py):
    #   - Model architecture matches train_cnn_final.py exactly (Dropout(0.5) -> Linear(256) -> ReLU -> Dropout(0.3) -> Linear(1))
    #   - Model was trained for "any abnormal vs normal" binary classification task
    #   - Output is directly used for "any abnormal" evaluation (diagnosis > 0)
    # Option 1: Binary-only CNN (direct binary output) - matches train_cnn_final.py
    # Option 2: Binary head (recommended) - trained for "any abnormal vs normal"
    # Option 3: DR head from multilabel - trained specifically for DR
    # Default: use binary output if available (better calibrated for binary classification)
    if is_binary_only_cnn:
        # Binary-only CNN: use binary probabilities directly for "any abnormal" evaluation
        # This matches train_cnn_final.py: model trained for diseases_risk (any disease present = 1, normal = 0)
        dr_scores = binary_probs if binary_probs is not None else all_probs.squeeze()
        any_scores = dr_scores  # For binary-only CNN, "any abnormal" = binary output
        print(f"  ✅ Using binary-only CNN output for 'any abnormal' evaluation (matches train_cnn_final.py)")
        print(f"     Model architecture: ResNet50 with binary head (Dropout(0.5) -> Linear(256) -> ReLU -> Dropout(0.3) -> Linear(1))")
        print(f"     Training task: 'any abnormal vs normal' (diseases_risk label)")
        print(f"     Evaluation task: 'any abnormal' (diagnosis > 0 for Messidor-2)")
    elif args.use_dr_head:
        # Force use of DR head from multilabel classifier
        print(f"  Using DR head from multilabel classifier (user-specified)")
        dr_scores = all_probs[:, dr_idx_in_multilabel]
        any_scores = dr_scores
    elif has_binary_head_in_ckpt and binary_probs is not None:
        # Use binary head (recommended for Messidor-2)
        print(f"  Using binary Disease_Risk head for Messidor-2 (DR-only dataset)")
        print(f"    Rationale: Binary head is better calibrated for 'any abnormal vs normal' task")
        dr_scores = binary_probs
        any_scores = binary_probs  # Binary head = any abnormal = DR present
    else:
        # Fallback: use DR head if binary head not available
        print(f"  Using DR head from multilabel classifier (binary head not available)")
        dr_scores = all_probs[:, dr_idx_in_multilabel]
        any_scores = dr_scores  # DR-only, not max over all classes

    # Debug: Print threshold and score distribution
    print(f"\n[Debug] tau_dr (RFMiD): {tau_dr:.6f}")
    print(f"[Debug] dr_scores (test): min={dr_scores.min():.4f}  p25={np.percentile(dr_scores,25):.4f}  "
          f"median={np.median(dr_scores):.4f}  p75={np.percentile(dr_scores,75):.4f}  max={dr_scores.max():.4f}")
    print(f"[Debug] % predicted positive @tau_dr: {(dr_scores >= tau_dr).mean()*100:.2f}%")
    print(f"[Debug] Ground truth (test): {y_dr.sum()}/{len(y_dr)} positive ({y_dr.mean()*100:.2f}%)")

    # Threshold selection: ALWAYS use validation split to prevent test leakage
    # All threshold adaptation methods compute threshold on validation split only
    # Track which method was actually used for CSV reporting
    # PRIORITY: target_specificity and target_sensitivity override threshold_method
    threshold_method_used = "rfmid"  # Default
    if use_val_split and tau_any is None:
        if args.target_specificity is not None:
            tau_any, spec_at_thr, sens_at_thr = pick_threshold_for_specificity(y_dr_train, dr_scores_train, target_spec=args.target_specificity)
            threshold_method_used = f"target_spec_{args.target_specificity:.2f}"
            print(f"📊 Chosen threshold (target specificity) on Messidor-2 validation: {tau_any:.6f}")
            print(f"   Gives specificity: {spec_at_thr:.3f} and sensitivity: {sens_at_thr:.3f}")
        elif args.target_sensitivity is not None:
            tau_any, got_tpr, got_tnr = threshold_for_tpr(y_dr_train, dr_scores_train, target_tpr=args.target_sensitivity)
            threshold_method_used = f"target_sens_{args.target_sensitivity:.2f}"
            print(f"📊 Chosen threshold (target sensitivity) on Messidor-2 validation: {tau_any:.6f}")
            print(f"   Gives sensitivity: {got_tpr:.3f} and specificity: {got_tnr:.3f}")
        elif args.threshold_method == "youden":
            tau_any, got_tpr, got_tnr, youden_j = threshold_for_youden_j(y_dr_train, dr_scores_train)
            threshold_method_used = "youden"
            print(f"📊 Chosen threshold (Youden's J) on Messidor-2 validation: {tau_any:.6f}")
            print(f"   Gives sensitivity: {got_tpr:.3f}, specificity: {got_tnr:.3f}, Youden's J: {youden_j:.3f}")
        elif args.threshold_method == "f1max" or args.use_f1max_threshold:
            tau_any, f1_val, prec_val, rec_val = threshold_for_f1max(y_dr_train, dr_scores_train)
            threshold_method_used = "f1max"
            print(f"📊 Chosen threshold (F1max) on Messidor-2 validation: {tau_any:.6f}")
            print(f"   Gives F1: {f1_val:.3f}, precision: {prec_val:.3f}, recall: {rec_val:.3f}")
        else:
            # Fallback to RFMiD threshold if no method specified
            tau_any = tau_dr
            threshold_method_used = "rfmid"
        print(f"   (Applied to test set for final evaluation)")
    elif tau_any is None:
        # No validation split and no explicit threshold - use RFMiD threshold
        tau_any = tau_dr
        threshold_method_used = "rfmid"

    # Any-abnormal metrics
    auc_any = safe_auc(y_any_abnormal, any_scores)
    f1max_any, thr_f1_any, prec_f1_any, rec_f1_any = f1_max(y_any_abnormal, any_scores)
    
    prec_any, rec_any, f1_any, tp_any, tn_any, fp_any, fn_any = f1_at_threshold(y_any_abnormal, any_scores, tau_any)
    
    sensitivity_any = tp_any / (tp_any + fn_any + 1e-8)
    specificity_any = tn_any / (tn_any + fp_any + 1e-8)
    balanced_accuracy_any = 0.5 * (sensitivity_any + specificity_any)
    accuracy_any = (tp_any + tn_any) / (tp_any + tn_any + fp_any + fn_any + 1e-8)
    print(f"Any-abnormal - Sensitivity: {sensitivity_any:.4f} | Specificity: {specificity_any:.4f} | "
          f"Balanced Acc: {balanced_accuracy_any:.4f} | Accuracy: {accuracy_any:.4f}")

    # Per-class DR metrics
    auc_dr = safe_auc(y_dr, dr_scores)
    f1max_dr, thr_f1_dr, prec_f1_dr, rec_f1_dr = f1_max(y_dr, dr_scores)
    
    # Debug: Print F1max threshold info
    print(f"[Debug] F1max on Messidor-2: {f1max_dr:.4f} at thr={thr_f1_dr:.6f}  "
          f"(prec={prec_f1_dr:.4f}, rec={rec_f1_dr:.4f})")
    
    # Use threshold selected on Messidor-2 validation (prevents test leakage)
    # All threshold adaptation methods use validation split, so tau_any is already set correctly
    tau_dr_actual = tau_any  # Use the same threshold selected for any-abnormal
    if use_val_split:
        print(f"📊 Using threshold selected on Messidor-2 validation for DR: {tau_dr_actual:.6f}")
    else:
        print(f"📊 Using RFMiD threshold for DR: {tau_dr_actual:.6f}")
    print(f"[Debug] % predicted positive @selected threshold: {(dr_scores >= tau_dr_actual).mean()*100:.2f}%")
    
    prec_dr, rec_dr, f1_dr, tp_dr, tn_dr, fp_dr, fn_dr = f1_at_threshold(y_dr, dr_scores, tau_dr_actual)
    
    sensitivity_dr = tp_dr / (tp_dr + fn_dr + 1e-8)
    specificity_dr = tn_dr / (tn_dr + fp_dr + 1e-8)
    balanced_accuracy_dr = 0.5 * (sensitivity_dr + specificity_dr)
    accuracy_dr = (tp_dr + tn_dr) / (tp_dr + tn_dr + fp_dr + fn_dr + 1e-8)
    print(f"DR - Sensitivity: {sensitivity_dr:.4f} | Specificity: {specificity_dr:.4f} | "
          f"Balanced Acc: {balanced_accuracy_dr:.4f} | Accuracy: {accuracy_dr:.4f}")

    # Referable DR (optional)
    if args.with_referable_dr:
        auc_rdr = safe_auc(y_dr_referr, dr_scores)
        prec_rdr, rec_rdr, f1_rdr, tp_rdr, tn_rdr, fp_rdr, fn_rdr = f1_at_threshold(y_dr_referr, dr_scores, tau_dr_actual)
        f1max_rdr, thr_f1_rdr, prec_f1_rdr, rec_f1_rdr = f1_max(y_dr_referr, dr_scores)
        
        sensitivity_rdr = tp_rdr / (tp_rdr + fn_rdr + 1e-8)
        specificity_rdr = tn_rdr / (tn_rdr + fp_rdr + 1e-8)
        balanced_accuracy_rdr = 0.5 * (sensitivity_rdr + specificity_rdr)
        accuracy_rdr = (tp_rdr + tn_rdr) / (tp_rdr + tn_rdr + fp_rdr + fn_rdr + 1e-8)
        print(f"Referable DR - Sensitivity: {sensitivity_rdr:.4f} | Specificity: {specificity_rdr:.4f} | "
              f"Balanced Acc: {balanced_accuracy_rdr:.4f} | Accuracy: {accuracy_rdr:.4f}")

    # Save metrics
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Any-abnormal
    with open(results_dir / "messidor2_any_abnormal_metrics.csv", "w") as f:
        f.write("Metric,Value\n")
        f.write(f"AUC (%),{auc_any*100:.4f}\n" if not np.isnan(auc_any) else f"AUC (%),nan\n")
        f.write(f"Threshold,{tau_any:.6f}\n")
        f.write(f"Precision@Thr,{prec_any*100:.4f}\n")
        f.write(f"Recall@Thr (%),{rec_any*100:.4f}\n")
        f.write(f"Sensitivity,{sensitivity_any:.6f}\n")
        f.write(f"Specificity,{specificity_any:.6f}\n")
        f.write(f"Balanced_Accuracy,{balanced_accuracy_any:.6f}\n")
        f.write(f"Accuracy,{accuracy_any:.6f}\n")
        f.write(f"TP,{int(tp_any)}\n")
        f.write(f"TN,{int(tn_any)}\n")
        f.write(f"FP,{int(fp_any)}\n")
        f.write(f"FN,{int(fn_any)}\n")
        f.write(f"F1max,{f1max_any:.6f}\n")
        f.write(f"F1max_Threshold,{thr_f1_any:.6f}\n")
        f.write(f"F1max_Precision,{prec_f1_any*100:.4f}\n")
        f.write(f"F1max_Recall (%),{rec_f1_any*100:.4f}\n")

    # DR-only results
    with open(results_dir / "messidor2_dr_results.csv", "w") as f:
        f.write("metric,value\n")
        f.write(f"auc(%),{auc_dr*100:.4f}\n" if not np.isnan(auc_dr) else f"auc(%),nan\n")
        f.write(f"threshold_rfmid_dr,{tau_dr:.6f}\n")
        f.write(f"threshold_used,{tau_dr_actual:.6f}\n")
        f.write(f"threshold_type,{threshold_method_used}\n")
        f.write(f"sensitivity,{sensitivity_dr:.6f}\n")
        f.write(f"specificity,{specificity_dr:.6f}\n")
        f.write(f"precision,{prec_dr:.6f}\n")
        f.write(f"recall,{rec_dr:.6f}\n")
        f.write(f"f1,{f1_dr:.6f}\n")
        f.write(f"accuracy,{accuracy_dr:.6f}\n")
        f.write(f"balanced_accuracy,{balanced_accuracy_dr:.6f}\n")
        f.write(f"tp,{int(tp_dr)}\n")
        f.write(f"tn,{int(tn_dr)}\n")
        f.write(f"fp,{int(fp_dr)}\n")
        f.write(f"fn,{int(fn_dr)}\n")
        f.write(f"f1max,{f1max_dr:.6f}\n")
        f.write(f"f1max_threshold,{thr_f1_dr:.6f}\n")
        f.write(f"f1max_precision,{prec_f1_dr:.6f}\n")
        f.write(f"f1max_recall,{rec_f1_dr:.6f}\n")

    # Referable DR (optional)
    if args.with_referable_dr:
        with open(results_dir / "messidor2_referable_dr_metrics.csv", "w") as f:
            f.write("Metric,Value\n")
            f.write(f"AUC (%),{auc_rdr*100:.4f}\n" if not np.isnan(auc_rdr) else f"AUC (%),nan\n")
            f.write(f"threshold_used,{tau_dr_actual:.6f}\n")
            f.write(f"threshold_type,{threshold_method_used}\n")
            f.write(f"threshold_rfmid_dr,{tau_dr:.6f}\n")  # Keep for reference
            f.write(f"Precision@Thr,{prec_rdr*100:.4f}\n")
            f.write(f"Recall@Thr (%),{rec_rdr*100:.4f}\n")
            f.write(f"Sensitivity,{sensitivity_rdr:.6f}\n")
            f.write(f"Specificity,{specificity_rdr:.6f}\n")
            f.write(f"Balanced_Accuracy,{balanced_accuracy_rdr:.6f}\n")
            f.write(f"Accuracy,{accuracy_rdr:.6f}\n")
            f.write(f"TP,{int(tp_rdr)}\n")
            f.write(f"TN,{int(tn_rdr)}\n")
            f.write(f"FP,{int(fp_rdr)}\n")
            f.write(f"FN,{int(fn_rdr)}\n")
            f.write(f"F1max,{f1max_rdr:.6f}\n")
            f.write(f"F1max_Threshold,{thr_f1_rdr:.6f}\n")
            f.write(f"F1max_Precision,{prec_f1_rdr*100:.4f}\n")
            f.write(f"F1max_Recall (%),{rec_f1_rdr*100:.4f}\n")

    # Save per-image outputs
    out_npz = results_dir / "messidor2_per_image_outputs.npz"
    y_pred_any_at_thr = (any_scores >= tau_any).astype(np.int8)
    np.savez(out_npz,
             ids=np.array(all_ids),
             y_true=y_any_abnormal.astype(np.int8),
             y_score=any_scores.astype(np.float32),
             y_pred_at_thr=y_pred_any_at_thr,
             thr_any=float(tau_any),
             dr_score=dr_scores.astype(np.float32),
             y_dr=y_dr.astype(np.int8),
             thr_dr=float(tau_dr_actual))  # Save actual threshold used, not original RFMiD threshold
    print(f"✅ Done. Results written under: {results_dir}")

if __name__ == "__main__":
    main()

