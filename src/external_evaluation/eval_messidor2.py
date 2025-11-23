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
    """Generic adapter for timm models with a standard multilabel head."""
    def __init__(self, timm_name: str, num_classes: int, drop_path: float = 0.2, head_dim: int = 256, fallback_names: list = None):
        super().__init__(timm_name, num_classes)
        self.timm_name = timm_name
        self.fallback_names = fallback_names or []
        self.drop_path = drop_path
        self.head_dim = head_dim

    def build(self) -> nn.Module:
        # Try primary name, then fallbacks if provided
        candidates = [self.timm_name] + self.fallback_names
        backbone = None
        used_name = None
        for name in candidates:
            try:
                backbone = timm.create_model(name, pretrained=True, num_classes=0, drop_path_rate=self.drop_path)
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
        multilabel_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_f, self.head_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.head_dim, self.num_classes),
            )
        
        class Wrapper(nn.Module):
            def __init__(self, backbone, multilabel_classifier):
                super().__init__()
                self.backbone = backbone
                self.classifier = multilabel_classifier  # For backward compatibility
                self.multilabel_classifier = multilabel_classifier
            def forward(self, x):
                z = self.backbone(x)
                multilabel_out = self.multilabel_classifier(z)
                # If a timm model returns tuple (rare in eval), take first
                if isinstance(multilabel_out, tuple):
                    multilabel_out = multilabel_out[0]
                return multilabel_out
        return Wrapper(backbone, multilabel_head)

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
            # weights force aux_logits=True at construction (matches train_cnn.py)
            model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
            # main head
            in_f = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_f, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, self.num_classes)
            )
            # aux head must also match num_classes (matches train_cnn.py structure)
            if model.AuxLogits is not None:
                aux_in = model.AuxLogits.fc.in_features
                model.AuxLogits.fc = nn.Linear(aux_in, self.num_classes)
        else:
            raise ValueError(f"Unknown CNN model: {self.model_name}")
        
        # Note: CNNs are returned directly (no wrapper) to match training script checkpoint structure
        # The training script saves models without wrappers, so we must match that structure
        return model

    def transforms(self, is_train: bool):
        """
        Use pretrained weights transforms to match train_cnn.py:
        - Uses ImageNet pretrained weights transforms (standard ImageNet preprocessing)
        - For training: adds RandomHorizontalFlip(0.5)
        - For eval: uses base transforms only
        """
        W = {
            "densenet121": DenseNet121_Weights.IMAGENET1K_V1,
            "resnet50": ResNet50_Weights.IMAGENET1K_V1,
            "efficientnet_b3": EfficientNet_B3_Weights.IMAGENET1K_V1,
            "inception_v3": Inception_V3_Weights.IMAGENET1K_V1,
        }[self.model_name]
        base = W.transforms(antialias=True)
        if is_train:
            # Light extra augmentation on top of the pretrained recipe
            return transforms.Compose([base, transforms.RandomHorizontalFlip(0.5)])
        return base

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
                
                # Use the same sophisticated prompts as in train_vlm.py for consistency
                class_prompts = {
                    "Disease_Risk": [
                        "a retinal fundus photograph with abnormal retinal findings including hemorrhages exudates drusen vessel changes or structural abnormalities",
                        "a retinal fundus photograph showing pathological changes such as retinal hemorrhages hard or soft exudates macular edema vessel tortuosity or optic disc abnormalities",
                        "a retinal fundus photograph with signs of retinal disease featuring microaneurysms cotton wool spots pigmentary changes chorioretinal atrophy or other pathological lesions"
                    ],
                    "DR": [
                        "a retinal fundus photograph with diabetic retinopathy featuring microaneurysms dot blot hemorrhages and hard exudates near the macula",
                        "a retinal fundus photograph with diabetic retinopathy showing scattered intraretinal hemorrhages cotton wool spots and possible neovascularization"
                    ],
                    "ARMD": [
                        "a retinal fundus photograph with age-related macular degeneration showing drusen and geographic atrophy",
                        "a retinal fundus photograph with age-related macular degeneration featuring large soft drusen and pigmentary changes"
                    ],
                    "MH": [
                        "a retinal fundus photograph with macular hole showing full-thickness foveal defect with operculum",
                        "a retinal fundus photograph with macular hole and surrounding cuff of subretinal fluid"
                    ],
                    "DN": [
                        "a retinal fundus photograph with diabetic maculopathy showing hard exudates and macular edema",
                        "a retinal fundus photograph with diabetic maculopathy featuring circinate rings of exudates and retinal thickening"
                    ],
                    "MYA": [
                        "a retinal fundus photograph with myopia showing myopic maculopathy and chorioretinal atrophy",
                        "a retinal fundus photograph with high myopia and posterior staphyloma"
                    ],
                    "BRVO": [
                        "a retinal fundus photograph with branch retinal vein occlusion showing flame-shaped hemorrhages and cotton wool spots",
                        "a retinal fundus photograph with branch retinal vein occlusion in the superior or inferior quadrant"
                    ],
                    "CRVO": [
                        "a retinal fundus photograph with central retinal vein occlusion showing extensive retinal hemorrhages and disc edema",
                        "a retinal fundus photograph with central retinal vein occlusion and macular edema"
                    ],
                    "RAO": [
                        "a retinal fundus photograph with retinal artery occlusion showing cherry red spot and retinal whitening",
                        "a retinal fundus photograph with central or branch retinal artery occlusion"
                    ],
                    "CSCR": [
                        "a retinal fundus photograph with central serous chorioretinopathy showing serous retinal detachment at the macula",
                        "a retinal fundus photograph with central serous chorioretinopathy and pigment epithelial detachment"
                    ],
                    "RPEC": [
                        "a retinal fundus photograph with retinal pigment epithelium changes showing mottling and hyper or hypopigmentation",
                        "a retinal fundus photograph with irregular retinal pigment epithelium at the macula and midperiphery"
                    ],
                    "MHL": [
                        "a retinal fundus photograph with lamellar macular hole showing partial thickness foveal defect with irregular foveal contour",
                        "a retinal fundus photograph with lamellar macular hole and epiretinal proliferation"
                    ],
                    "RP": [
                        "a retinal fundus photograph with retinitis pigmentosa showing bone spicule pigmentation attenuated vessels and waxy disc pallor",
                        "a retinal fundus photograph with peripheral bone spicule pigment and narrow arterioles typical of retinitis pigmentosa"
                    ],
                    "OTHER": [
                        "a retinal fundus photograph with an uncommon retinal pathology not listed among other classes",
                        "a retinal fundus photograph with miscellaneous retinal abnormality outside predefined categories"
                    ],
                }
                
                # Encode all class prompts and average per class (matches train_vlm.py)
                all_txt = []
                for cname in self.class_names:
                    variants = class_prompts.get(cname, [f"a retinal fundus photograph with {cname.lower()}"])
                    tokens = open_clip.tokenize(variants).to(device)
                    self.base_model.eval()
                    with torch.no_grad():
                        emb = self.base_model.encode_text(tokens)
                        emb = emb / emb.norm(dim=-1, keepdim=True)
                        emb = emb.mean(dim=0, keepdim=True)  # average over variants
                    all_txt.append(emb)
                
                txt = torch.cat(all_txt, dim=0)  # [num_classes, D]
                
                # Resize and copy to keep it as a registered buffer
                if self.cached_text_embeds.numel() == 0:
                    self.cached_text_embeds.resize_as_(txt.detach().cpu()).copy_(txt.detach().cpu())
                else:
                    self.cached_text_embeds.copy_(txt.detach().cpu())
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
    "coatnet0": lambda nc: TimmBackboneWithHead("coatnet_0_rw_224.sw_in1k", nc),  # Match train_hybrid.py
    "maxvit_tiny": lambda nc: TimmBackboneWithHead("maxvit_tiny_tf_224", nc),  # Match train_hybrid.py
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
    ap.add_argument("--thresholds", required=True, help="Path to RFMiD per-class thresholds .npy")
    ap.add_argument("--rfmid_train_csv", required=True, help="RFMiD training labels CSV (to recover label order)")
    ap.add_argument("--messidor_csv", required=True, help="Messidor-2 labels CSV")
    ap.add_argument("--images_dir", required=True, help="Messidor-2 images directory")
    ap.add_argument("--results_dir", required=True, help="Output directory for metrics")
    ap.add_argument("--eval_img_size", type=int, default=448, help="Input image size for evaluation (default: 448, larger than training)")
    ap.add_argument("--disable_clahe", action="store_true", help="Disable CLAHE preprocessing (use standard transforms)")
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
    
    # Load thresholds from RFMiD validation set
    if not Path(args.thresholds).exists():
        raise ValueError(f"Threshold file not found: {args.thresholds}")
        thresholds = np.load(args.thresholds)

    
    # Find DR index in label columns (always use DR head from multilabel classifier)
        dr_idx = _find_label_index(label_columns, DR_ALIASES)
        if dr_idx is None:
            raise ValueError(f"Could not find DR column in RFMiD labels. Checked aliases: {DR_ALIASES}")

    # Thresholds file handling: expect thresholds for all classes (including DR)
    expected_threshold_count = len(label_columns)
    if len(thresholds) != expected_threshold_count:
        raise ValueError(f"Threshold length mismatch: expected {expected_threshold_count} thresholds (for {len(label_columns)} label columns), "
                        f"but got {len(thresholds)} thresholds in file.")
    
    # Use DR threshold from RFMiD validation set (always use this fixed threshold)
    dr_idx_in_thresholds = dr_idx
    tau_dr = float(thresholds[dr_idx_in_thresholds])
    tau_any = tau_dr  # Always use RFMiD threshold for Messidor-2 evaluation
    print(f"✅ Using RFMiD validation threshold for DR: {tau_dr:.6f}")

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

    # Use all data for evaluation (no domain adaptation)
    mdf_test = mdf
    
    # Ground truths
    y_any_abnormal = (mdf_test["diagnosis"].astype(int) > 0).astype(np.int32).values
    y_dr = (mdf_test["diagnosis"].astype(int) > 0).astype(np.int32).values
    y_dr_referr = (mdf_test["diagnosis"].astype(int) >= 2).astype(np.int32).values if args.with_referable_dr else None

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    
    # Debug: Print checkpoint structure for CLIP models
    if "clip" in args.model_name.lower() or "biomedclip" in args.model_name.lower():
        print(f"  🔍 Checkpoint structure: top-level keys = {list(ckpt.keys())[:10]}")
        print(f"  🔍 State dict keys sample: {list(state.keys())[:10]}")
    
    # Build model via adapter - always use multilabel classifier (includes DR class)
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
            adapter = REGISTRY[model_name_lower](num_classes)
    else:
        raise ValueError(f"Unknown model: {args.model_name}. Available: {list(REGISTRY.keys())}")

    model = adapter.build().to(device)
    
    # Fix key mismatches (checkpoint may have different prefixes)
    model_keys = set(model.state_dict().keys())
    state_keys = set(state.keys())
    
    # Handle EfficientNetB3 classifier key mismatch FIRST (before other remapping)
    if model_name_lower == "efficientnet_b3":
        # Check if checkpoint has nested classifier keys (classifier.1.1.* instead of classifier.1.*)
        if any(k.startswith("classifier.1.1.") for k in state_keys):
            # Remap classifier.1.1.* -> classifier.1.* and classifier.1.4.* -> classifier.4.*
            # This handles checkpoints saved with nested Sequential structure
            new_state = {}
            remapped = False
            for k, v in state.items():
                if k.startswith("classifier.1.1."):
                    # Remap classifier.1.1.weight -> classifier.1.weight
                    new_key = k.replace("classifier.1.1.", "classifier.1.")
                    new_state[new_key] = v
                    remapped = True
                elif k.startswith("classifier.1.4."):
                    # Remap classifier.1.4.weight -> classifier.4.weight
                    new_key = k.replace("classifier.1.4.", "classifier.4.")
                    new_state[new_key] = v
                    remapped = True
                else:
                    new_state[k] = v
            if remapped:
                state = new_state
                state_keys = set(state.keys())  # Update state_keys after remapping
                print(f"  ✅ Remapped EfficientNetB3 classifier keys: classifier.1.1.* → classifier.1.*, classifier.1.4.* → classifier.4.*")
    
    # Handle key remapping for ViT/Hybrid models (checkpoint may use "classifier.*" but model expects "multilabel_classifier.*")
    if model_name_lower in ["swin_tiny", "vit_small", "deit_small", "crossvit_small", "coatnet0", "maxvit_tiny"]:
        # Remap "classifier.*" to "multilabel_classifier.*" if needed
        # Also add "classifier.*" keys for backward compatibility (model has both attributes pointing to same object)
        if any(k.startswith("classifier.") for k in state_keys) and not any(k.startswith("multilabel_classifier.") for k in state_keys):
            new_state = {}
            for k, v in state.items():
                if k.startswith("classifier."):
                    # Add both multilabel_classifier.* (primary) and classifier.* (backward compat)
                    new_state[k.replace("classifier.", "multilabel_classifier.")] = v
                    new_state[k] = v  # Keep original for backward compatibility
                else:
                    new_state[k] = v
            state = new_state
            state_keys = set(state.keys())  # Update state_keys after remapping
            print(f"  ✅ Remapped checkpoint keys: classifier.* → multilabel_classifier.* (and kept classifier.* for backward compatibility)")
    
    # If keys still don't match, try remapping common patterns
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
        
        # Handle CLIP cached_text_embeds buffer shape mismatch
        if "clip" in model_name_lower or "biomedclip" in model_name_lower:
            # Remove cached_text_embeds from state if shape doesn't match (will be recomputed)
            if "cached_text_embeds" in state:
                checkpoint_shape = state["cached_text_embeds"].shape
                model_shape = model.cached_text_embeds.shape
                if checkpoint_shape != model_shape:
                    print(f"  ℹ️  Removing cached_text_embeds from checkpoint (shape mismatch: {checkpoint_shape} vs {model_shape}), will be recomputed")
                    state = {k: v for k, v in state.items() if k != "cached_text_embeds"}
        
        # Debug: Check critical CLIP keys
        if "clip" in model_name_lower or "biomedclip" in model_name_lower:
            critical_keys = ["logit_scale", "class_bias", "base_model.visual"]
            has_logit_scale = any("logit_scale" in k for k in state_keys)
            has_class_bias = any("class_bias" in k for k in state_keys)
            has_base_visual = any("base_model.visual" in k for k in state_keys)
            print(f"  🔍 CLIP checkpoint check: logit_scale={has_logit_scale}, class_bias={has_class_bias}, base_model.visual={has_base_visual}")
            
            # Check if we need to remap keys (e.g., if checkpoint has different prefix)
            if not has_base_visual and not has_logit_scale:
                # Try to find keys with different prefixes
                visual_keys = [k for k in state_keys if "visual" in k.lower()]
                if visual_keys:
                    print(f"  ℹ️  Found visual keys with different prefix: {visual_keys[:3]}...")
                    # Try remapping common patterns
                    if any("base_model.visual" not in k and "visual" in k for k in state_keys):
                        # Check if we need to add "base_model." prefix
                        new_state = {}
                        for k, v in state.items():
                            if k.startswith("visual.") and not k.startswith("base_model."):
                                new_state["base_model." + k] = v
                            elif k.startswith("base_model.") or "logit_scale" in k or "class_bias" in k:
                                new_state[k] = v
                            else:
                                new_state[k] = v
                        state = new_state
                        state_keys = set(state.keys())
                        print(f"  ✅ Attempted key remapping for CLIP model")
        
        try:
            missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
            
            # Validate critical keys were loaded for CLIP
            if "clip" in model_name_lower or "biomedclip" in model_name_lower:
                # Check which keys from checkpoint actually matched and loaded
                model_state = model.state_dict()
                loaded_keys = set(model_state.keys())
                
                # Check if critical parameters exist in model
                logit_keys = [k for k in loaded_keys if "logit_scale" in k]
                bias_keys = [k for k in loaded_keys if "class_bias" in k]
                visual_keys = [k for k in loaded_keys if "base_model.visual" in k]
                
                # Check if they were actually loaded from checkpoint (not just initialized)
                logit_loaded = len(logit_keys) > 0 and any(k in state_keys for k in logit_keys)
                bias_loaded = len(bias_keys) > 0 and any(k in state_keys for k in bias_keys)
                visual_loaded = len(visual_keys) > 0 and any(k in state_keys for k in visual_keys[:5])  # Check first few
                
                if not (logit_loaded and bias_loaded and visual_loaded):
                    print(f"  ⚠️  CRITICAL WARNING: CLIP weights may not have loaded correctly!")
                    print(f"     Checkpoint has logit_scale: {logit_loaded}, class_bias: {bias_loaded}, base_model.visual: {visual_loaded}")
                    print(f"     Missing keys: {[k for k in missing_keys if 'logit_scale' in k or 'class_bias' in k or 'base_model.visual' in k][:5]}")
                    print(f"     This will cause poor performance. The model may be using untrained weights.")
                else:
                    print(f"  ✅ CLIP critical weights verified: logit_scale, class_bias, base_model.visual")
            
            if missing_keys:
                print(f"⚠️  Warning: Missing keys in {vlm_type} checkpoint: {missing_keys[:10]}..." if len(missing_keys) > 10 else f"⚠️  Warning: Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"⚠️  Warning: Unexpected keys in {vlm_type} checkpoint: {unexpected_keys[:10]}..." if len(unexpected_keys) > 10 else f"⚠️  Warning: Unexpected keys: {unexpected_keys}")
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

    # Use adapter's transforms for all models to match training pipeline
    # This ensures preprocessing is aligned with training code:
    # - CNNs: Pretrained weights transforms (matches train_cnn.py)
    # - ViT/Hybrid: timm transforms (matches train_vit.py / train_hybrid.py)
    # - CLIP/SigLIP: VLM-specific transforms (matches train_vlm.py)
    if model_name_lower in ["densenet121", "resnet50", "efficientnet_b3", "inception_v3"]:
        # CNN models: use pretrained weights transforms (matches train_cnn.py)
        tfm = adapter.transforms(is_train=False)
        print(f"✅ Using pretrained weights transforms for CNN model (matches train_cnn.py):")
        print(f"   ImageNet standard preprocessing from pretrained weights")
    else:
        # ViT, Hybrid, VLM models: use training-aligned transforms from adapter
        tfm = adapter.transforms(is_train=False)
        if "clip" in model_name_lower or "biomedclip" in model_name_lower:
            print(f"✅ Using CLIP transforms (matches train_vlm.py): CLIP normalization, size 224/336")
        elif "siglip" in model_name_lower:
            print(f"✅ Using SigLIP transforms (matches train_vlm.py): SigLIP normalization from timm config")
        else:
            # ViT or Hybrid models
            print(f"✅ Using timm transforms for {model_name_lower} (matches train_vit.py / train_hybrid.py)")
    
    # Create dataset
    ds_test = Messidor2Dataset(Path(args.images_dir), mdf_test, tfm)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=args.pin_memory)
    
    # Run inference
    use_amp = torch.cuda.is_available() or (torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False)
    dev_type = "cuda" if torch.cuda.is_available() else ("mps" if use_amp and hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    
    print(f"Running inference on {len(dl_test.dataset)} images ({len(dl_test)} batches)...")
    all_logits = []
    all_ids = []
    total_batches = len(dl_test)
    
    with torch.inference_mode(), torch.autocast(device_type=dev_type, enabled=(dev_type != "cpu")):
        for batch_idx, (xb, ids) in enumerate(dl_test, 1):
            xb = xb.to(device)
            outputs = model(xb)
            
            # Handle tuple output (take first element if tuple)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            if hasattr(logits, "logits"):
                logits = logits.logits
            elif isinstance(logits, (tuple, list)):
                logits = logits[0]
            
            all_logits.append(logits.cpu())
            all_ids.extend(ids)
            
            if batch_idx % 10 == 0 or batch_idx == total_batches:
                print(f"  Processed {batch_idx}/{total_batches} batches ({len(all_ids)} images)...", flush=True)
    
    all_logits = torch.cat(all_logits, dim=0)
    all_probs = torch.sigmoid(all_logits).numpy()
    
    print(f"✅ Inference complete: {all_probs.shape[0]} images, {all_probs.shape[1]} classes.")

    # Scores for endpoints (DR-only for Messidor-2)
    # Always use DR head from multilabel classifier with RFMiD threshold
    print(f"  Using DR head from multilabel classifier for Messidor-2 (DR-only dataset)")
    print(f"  Using RFMiD validation threshold: {tau_dr:.6f}")
    dr_scores = all_probs[:, dr_idx]
    any_scores = dr_scores  # DR-only, not max over all classes

    # Debug: Print threshold and score distribution
    print(f"\n[Debug] tau_dr (RFMiD): {tau_dr:.6f}")
    print(f"[Debug] dr_scores (test): min={dr_scores.min():.4f}  p25={np.percentile(dr_scores,25):.4f}  "
          f"median={np.median(dr_scores):.4f}  p75={np.percentile(dr_scores,75):.4f}  max={dr_scores.max():.4f}")
    print(f"[Debug] % predicted positive @tau_dr: {(dr_scores >= tau_dr).mean()*100:.2f}%")
    print(f"[Debug] Ground truth (test): {y_dr.sum()}/{len(y_dr)} positive ({y_dr.mean()*100:.2f}%)")

    # Always use RFMiD threshold (no threshold adaptation)
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
    
    # Always use RFMiD threshold (tau_dr)
    print(f"📊 Using RFMiD validation threshold for DR: {tau_dr:.6f}")
    print(f"[Debug] % predicted positive @RFMiD threshold: {(dr_scores >= tau_dr).mean()*100:.2f}%")
    
    prec_dr, rec_dr, f1_dr, tp_dr, tn_dr, fp_dr, fn_dr = f1_at_threshold(y_dr, dr_scores, tau_dr)
    
    sensitivity_dr = tp_dr / (tp_dr + fn_dr + 1e-8)
    specificity_dr = tn_dr / (tn_dr + fp_dr + 1e-8)
    balanced_accuracy_dr = 0.5 * (sensitivity_dr + specificity_dr)
    accuracy_dr = (tp_dr + tn_dr) / (tp_dr + tn_dr + fp_dr + fn_dr + 1e-8)
    print(f"DR - Sensitivity: {sensitivity_dr:.4f} | Specificity: {specificity_dr:.4f} | "
          f"Balanced Acc: {balanced_accuracy_dr:.4f} | Accuracy: {accuracy_dr:.4f}")

    # Calculate metrics at 80% specificity threshold
    thr_80spec_dr, spec_80spec_dr, sens_80spec_dr = pick_threshold_for_specificity(y_dr, dr_scores, target_spec=0.8)
    prec_80spec_dr, rec_80spec_dr, f1_80spec_dr, tp_80spec_dr, tn_80spec_dr, fp_80spec_dr, fn_80spec_dr = f1_at_threshold(y_dr, dr_scores, thr_80spec_dr)
    accuracy_80spec_dr = (tp_80spec_dr + tn_80spec_dr) / (tp_80spec_dr + tn_80spec_dr + fp_80spec_dr + fn_80spec_dr + 1e-8)
    balanced_accuracy_80spec_dr = 0.5 * (sens_80spec_dr + spec_80spec_dr)
    print(f"DR @80% Spec - Threshold: {thr_80spec_dr:.6f} | Sensitivity: {sens_80spec_dr:.4f} | Specificity: {spec_80spec_dr:.4f} | "
          f"F1: {f1_80spec_dr:.4f} | Accuracy: {accuracy_80spec_dr:.4f}")

    # Referable DR (optional)
    if args.with_referable_dr:
        auc_rdr = safe_auc(y_dr_referr, dr_scores)
        prec_rdr, rec_rdr, f1_rdr, tp_rdr, tn_rdr, fp_rdr, fn_rdr = f1_at_threshold(y_dr_referr, dr_scores, tau_dr)
        f1max_rdr, thr_f1_rdr, prec_f1_rdr, rec_f1_rdr = f1_max(y_dr_referr, dr_scores)
        
        sensitivity_rdr = tp_rdr / (tp_rdr + fn_rdr + 1e-8)
        specificity_rdr = tn_rdr / (tn_rdr + fp_rdr + 1e-8)
        balanced_accuracy_rdr = 0.5 * (sensitivity_rdr + specificity_rdr)
        accuracy_rdr = (tp_rdr + tn_rdr) / (tp_rdr + tn_rdr + fp_rdr + fn_rdr + 1e-8)
        print(f"Referable DR - Sensitivity: {sensitivity_rdr:.4f} | Specificity: {specificity_rdr:.4f} | "
              f"Balanced Acc: {balanced_accuracy_rdr:.4f} | Accuracy: {accuracy_rdr:.4f}")
        
        # Calculate Referable DR metrics at 80% specificity threshold
        thr_80spec_rdr, spec_80spec_rdr, sens_80spec_rdr = pick_threshold_for_specificity(y_dr_referr, dr_scores, target_spec=0.8)
        prec_80spec_rdr, rec_80spec_rdr, f1_80spec_rdr, tp_80spec_rdr, tn_80spec_rdr, fp_80spec_rdr, fn_80spec_rdr = f1_at_threshold(y_dr_referr, dr_scores, thr_80spec_rdr)
        accuracy_80spec_rdr = (tp_80spec_rdr + tn_80spec_rdr) / (tp_80spec_rdr + tn_80spec_rdr + fp_80spec_rdr + fn_80spec_rdr + 1e-8)
        balanced_accuracy_80spec_rdr = 0.5 * (sens_80spec_rdr + spec_80spec_rdr)
        print(f"Referable DR @80% Spec - Threshold: {thr_80spec_rdr:.6f} | Sensitivity: {sens_80spec_rdr:.4f} | Specificity: {spec_80spec_rdr:.4f} | "
              f"F1: {f1_80spec_rdr:.4f} | Accuracy: {accuracy_80spec_rdr:.4f}")

    # Save metrics
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # DR results (for Messidor-2, "any abnormal" = DR, so we only need DR results)
    with open(results_dir / "messidor2_dr_results.csv", "w") as f:
        f.write("metric,value\n")
        f.write(f"auc(%),{auc_dr*100:.4f}\n" if not np.isnan(auc_dr) else f"auc(%),nan\n")
        f.write(f"threshold_rfmid_dr,{tau_dr:.6f}\n")
        f.write(f"threshold_used,{tau_dr:.6f}\n")
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
        # Metrics at 80% specificity
        f.write(f"threshold_80spec,{thr_80spec_dr:.6f}\n")
        f.write(f"sensitivity_80spec,{sens_80spec_dr:.6f}\n")
        f.write(f"specificity_80spec,{spec_80spec_dr:.6f}\n")
        f.write(f"precision_80spec,{prec_80spec_dr:.6f}\n")
        f.write(f"recall_80spec,{rec_80spec_dr:.6f}\n")
        f.write(f"f1_80spec,{f1_80spec_dr:.6f}\n")
        f.write(f"accuracy_80spec,{accuracy_80spec_dr:.6f}\n")
        f.write(f"balanced_accuracy_80spec,{balanced_accuracy_80spec_dr:.6f}\n")
        f.write(f"tp_80spec,{int(tp_80spec_dr)}\n")
        f.write(f"tn_80spec,{int(tn_80spec_dr)}\n")
        f.write(f"fp_80spec,{int(fp_80spec_dr)}\n")
        f.write(f"fn_80spec,{int(fn_80spec_dr)}\n")

    # Referable DR (optional)
    if args.with_referable_dr:
        with open(results_dir / "messidor2_referable_dr_metrics.csv", "w") as f:
            f.write("Metric,Value\n")
            f.write(f"AUC (%),{auc_rdr*100:.4f}\n" if not np.isnan(auc_rdr) else f"AUC (%),nan\n")
            f.write(f"threshold_used,{tau_dr:.6f}\n")
            f.write(f"threshold_type,{threshold_method_used}\n")
            f.write(f"threshold_rfmid_dr,{tau_dr:.6f}\n")  # Keep for reference
            f.write(f"Precision@Thr,{prec_rdr*100:.4f}\n")
            f.write(f"Recall@Thr (%),{rec_rdr*100:.4f}\n")
            f.write(f"F1@Thr,{f1_rdr:.6f}\n")
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
            # Metrics at 80% specificity
            f.write(f"Threshold_80Spec,{thr_80spec_rdr:.6f}\n")
            f.write(f"Sensitivity_80Spec,{sens_80spec_rdr:.6f}\n")
            f.write(f"Specificity_80Spec,{spec_80spec_rdr:.6f}\n")
            f.write(f"Precision_80Spec,{prec_80spec_rdr*100:.4f}\n")
            f.write(f"Recall_80Spec (%),{rec_80spec_rdr*100:.4f}\n")
            f.write(f"F1_80Spec,{f1_80spec_rdr:.6f}\n")
            f.write(f"Accuracy_80Spec,{accuracy_80spec_rdr:.6f}\n")
            f.write(f"Balanced_Accuracy_80Spec,{balanced_accuracy_80spec_rdr:.6f}\n")
            f.write(f"TP_80Spec,{int(tp_80spec_rdr)}\n")
            f.write(f"TN_80Spec,{int(tn_80spec_rdr)}\n")
            f.write(f"FP_80Spec,{int(fp_80spec_rdr)}\n")
            f.write(f"FN_80Spec,{int(fn_80spec_rdr)}\n")

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
             thr_dr=float(tau_dr))  # RFMiD validation threshold
    print(f"✅ Done. Results written under: {results_dir}")

if __name__ == "__main__":
    main()

