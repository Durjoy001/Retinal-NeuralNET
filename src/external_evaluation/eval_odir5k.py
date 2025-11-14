# external_eval_odir_any.py
# Evaluate RFMiD-trained ViTs on ODIR-5K for Any-Abnormal only.
# - Prefers Disease_Risk head if available; otherwise supports pooling strategies.
# - Can apply RFMiD per-class threshold for Disease_Risk (from optimal_thresholds.npy)
#   or recalibrate a threshold on a small ODIR holdout (no weight updates).

import os, argparse, random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
from sklearn.model_selection import train_test_split, GroupShuffleSplit

# ----------------------
# Transforms + backbones (aligned with your training script)
# ----------------------
def vit_transforms(model_name: str, train: bool):
    model_map = {
        "swin_tiny": "swin_tiny_patch4_window7_224",
        "vit_small": "vit_small_patch16_224",
        "deit_small": "deit_small_patch16_224",
        "crossvit_small": "crossvit_15_240",
    }
    if model_name not in model_map:
        raise ValueError(f"Unknown ViT model name: {model_name}")
    m = timm.create_model(model_map[model_name], pretrained=True, num_classes=1)
    cfg = resolve_data_config({}, model=m)
    return create_transform(**cfg, is_training=train)

def build_model(model_name, num_classes, include_binary_head=False):
    """
    Build ViT model with optional binary head for Disease_Risk.
    If include_binary_head=True, returns model with both multilabel and binary heads.
    """
    model_name = model_name.lower()
    if model_name == "swin_tiny":
        backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0, drop_path_rate=0.2)
    elif model_name == "vit_small":
        backbone = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0, drop_path_rate=0.25)
    elif model_name == "deit_small":
        backbone = timm.create_model('deit_small_patch16_224', pretrained=True, num_classes=0, drop_path_rate=0.2)
    elif model_name == "crossvit_small":
        backbone = timm.create_model('crossvit_15_240', pretrained=True, num_classes=0, drop_path_rate=0.2)
    else:
        raise ValueError(f"Unknown ViT model name: {model_name}")
    in_f = backbone.num_features
    
    # Multi-label classifier head
    multilabel_classifier = nn.Sequential(
        nn.Dropout(0.6),
        nn.Linear(in_f, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )
    
    # Binary classifier head (for Disease_Risk - any abnormal vs normal)
    binary_classifier = None
    if include_binary_head:
        binary_classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(in_f, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1)  # Binary output
        )
    
    class ViTWrapper(nn.Module):
        def __init__(self, b, multilabel_head, binary_head=None):
            super().__init__()
            self.backbone = b
            self.classifier = multilabel_head  # For backward compatibility
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
    
    return ViTWrapper(backbone, multilabel_classifier, binary_classifier)

# ----------------------
# ODIR dataset (Any-Abnormal only)
# ----------------------
class ODIRAnyAbnDataset(Dataset):
    """
    One sample per eye. Label = 1 if any of {D,G,C,A,H,M,O} == 1, else 0.
    Expected columns:
      ID, Patient Age, Patient Sex, Left-Fundus, Right-Fundus,
      Left-Diagnostic Keywords, Right-Diagnostic Keywords, N,D,G,C,A,H,M,O
    """
    def __init__(self, df, img_root, transform):
        self.samples = []
        self.transform = transform
        self.img_root = Path(img_root)

        required = ["N","D","G","C","A","H","M","O","Left-Fundus","Right-Fundus","ID"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in ODIR file")

        for col in ["N","D","G","C","A","H","M","O"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        for _, r in df.iterrows():
            any_abn = int((r["D"]+r["G"]+r["C"]+r["A"]+r["H"]+r["M"]+r["O"]) > 0)

            for side in ["Left-Fundus","Right-Fundus"]:
                fname = str(r.get(side,"")).strip()
                if fname and fname != "-" and fname.lower() != "nan":
                    img_p = self.img_root / fname
                    # Check if file exists and has valid extension
                    if img_p.exists() and img_p.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".tif",".tiff"]:
                        self.samples.append({"img_path": img_p, "label": any_abn, "pid": r.get("ID", None)})

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            img = Image.open(s["img_path"]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, int(s["label"])
        except Exception as e:
            raise RuntimeError(f"Failed to load image {s['img_path']}: {e}")

# ----------------------
# RFMiD to ODIR label mapping
# ----------------------
def find_label_indices(rfmid_labels: List[str], target_names: List[str], odir_class: str) -> List[int]:
    """
    Find indices of RFMiD labels matching target names.
    First tries exact match, then fuzzy substring matching.
    Returns list of indices and reports which labels were found/not found.
    """
    found_indices = []
    found_names = []
    not_found = []
    
    # Build lowercase lookup for fuzzy matching
    label_lower_map = {name.lower(): (i, name) for i, name in enumerate(rfmid_labels)}
    
    for target in target_names:
        target_lower = target.lower()
        # Try exact match first
        if target in rfmid_labels:
            idx = rfmid_labels.index(target)
            found_indices.append(idx)
            found_names.append(target)
        # Try fuzzy substring match
        else:
            matched = False
            for label_lower, (idx, label_orig) in label_lower_map.items():
                if target_lower in label_lower or label_lower in target_lower:
                    if idx not in found_indices:  # Avoid duplicates
                        found_indices.append(idx)
                        found_names.append(label_orig)
                        matched = True
                        break
            if not matched:
                not_found.append(target)
    
    if not_found:
        print(f"    ⚠️ {odir_class}: Could not find RFMiD labels: {not_found}")
    if found_names:
        print(f"    ✅ {odir_class}: Found {len(found_names)} labels -> {found_names}")
    
    return found_indices

def build_odir_mapping(rfmid_labels: List[str]) -> Dict[str, List[int]]:
    """
    Return indices of RFMiD labels that correspond to each ODIR class:
    ODIR: D (DR), G (Glaucoma), C (Cataract), A (AMD),
          H (Hypertension), M (Myopia), O (Other)
    
    Uses robust matching: exact match first, then fuzzy substring fallback.
    Reports which labels were found/not found for transparency.
    """
    print(f"  Building RFMiD→ODIR mapping from {len(rfmid_labels)} RFMiD labels...")
    
    # Define target labels for each ODIR class (with fuzzy fallback candidates)
    mapping_spec = {
        "D": ["DR"],  # Diabetic Retinopathy
        "G": ["ODC"],  # Optic Disc Cupping (Glaucoma)
        "C": [],  # Cataract (intentionally empty - not in RFMiD)
        "A": ["ARMD", "RPEC"],  # AMD: Age-related Macular Degeneration, RPE Changes
        "H": ["AH", "TV", "ODE"],  # Hypertension: Arterial Hypertension, Tortuous Vessels, Optic Disc Edema
        "M": ["MYA", "TSLN", "ST"],  # Myopia: Myopia, Tessellated Fundus, Staphyloma
    }
    
    mapping = {}
    for odir_class, target_names in mapping_spec.items():
        if odir_class == "C":
            mapping["C"] = []  # Intentionally empty
            print(f"    ℹ️  C: Intentionally empty (no Cataract label in RFMiD)")
        else:
            indices = find_label_indices(rfmid_labels, target_names, odir_class)
            mapping[odir_class] = indices
    
    # O (Other): Curated set of non-overlapping "other" diseases
    # Only include frequent, distinct entities that don't overlap with D/G/A/H/M
    used = set(sum(mapping.values(), []))
    
    # Curated "Other" labels - only include actual disease entities
    # Exclude: ID, Disease_Risk, and labels already mapped to D/G/A/H/M
    other_candidates = [
        "BRVO", "CRVO",  # Retinal vein occlusions
        "ERM",  # Epiretinal membrane
        "MH", "MHL",  # Macular hole
        "CSR",  # Central serous retinopathy
        "LS", "MS",  # Laser scars, macular scars
        "DN",  # Drusen (if not already in AMD)
        "ODP",  # Optic disc pallor
        "AION",  # Anterior ischemic optic neuropathy
        "PT", "RT", "RS", "CRS",  # Pathological changes
        "EDN",  # Edema
        "RP",  # Retinal pigment changes
        "OTHER",  # Explicit "other" category
    ]
    
    # Find indices for curated "Other" labels
    other_indices = find_label_indices(rfmid_labels, other_candidates, "O")
    
    # If we found curated labels, use them; otherwise fall back to all remaining
    if other_indices:
        mapping["O"] = other_indices
        print(f"    ✅ O: {len(other_indices)} curated labels -> Other")
    else:
        # Fallback: use all remaining (but warn)
        others = [i for i, name in enumerate(rfmid_labels) 
                  if name not in ("ID", "Disease_Risk") and i not in used]
        mapping["O"] = others
        print(f"    ⚠️ O: No curated labels found, using {len(others)} remaining labels -> Other")
    
    return mapping

def noisyor_over_indices(probs, indices):
    """Noisy-OR pooling over specific indices"""
    # probs: [B, C], indices: list[int]
    if not indices: 
        return np.zeros((probs.shape[0],), dtype=np.float32)
    p = np.clip(probs[:, indices], 1e-6, 1 - 1e-6)
    return 1.0 - np.prod(1.0 - p, axis=1)

# ----------------------
# Temperature Scaling
# ----------------------
def fit_temperature_binary(logits_np, y_np):
    """
    Fit a single temperature parameter T for binary calibration.
    logits_np: shape [N]  (use the logit you will threshold on: drisk or odir_map pooled *logit*)
    y_np:      shape [N] in {0,1}
    Returns scalar temperature T>0
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    l = torch.tensor(logits_np, dtype=torch.float32, device=device).view(-1, 1)
    y = torch.tensor(y_np, dtype=torch.float32, device=device).view(-1, 1)

    T = nn.Parameter(torch.ones(1, device=device))
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.LBFGS([T], lr=0.5, max_iter=50, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = criterion(l / T.clamp_min(1e-3), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(T.detach().cpu().item())

# ----------------------
# Pooling strategies for Any-Abnormal when Disease_Risk is absent
# ----------------------
def pool_any(prob_matrix, mode="noisyor", T=1.0):
    """
    prob_matrix: numpy array [B, C] of per-class sigmoid probabilities.
    mode: "noisyor" | "max" | "logitsum"
    """
    p = np.clip(prob_matrix, 1e-6, 1-1e-6)
    if mode == "noisyor":
        return 1.0 - np.prod(1.0 - p, axis=1)
    elif mode == "max":
        return np.max(p, axis=1)
    elif mode == "logitsum":
        from scipy.special import logit, expit, logsumexp
        s = logit(p)
        return expit(logsumexp(s / float(T), axis=1))
    else:
        raise ValueError(f"Unknown pooling mode: {mode}")

# ----------------------
# Metrics
# ----------------------
def pick_thr_for_specificity(y_true, y_score, target_spec=0.8):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    spec = 1.0 - fpr
    idx = int(np.argmin(np.abs(spec - target_spec)))
    return float(thr[idx]), float(spec[idx]), float(tpr[idx])

def compute_f1max(y_true, y_score):
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    f1_use = f1[:-1]
    best = int(np.nanargmax(f1_use))
    return float(f1_use[best]), float(thr[best]), float(prec[best]), float(rec[best])

def patient_level_arrays(samples, y_eye, s_eye):
    """
    Build patient-level arrays from per-eye outputs using max-pooling of scores.
    samples: full_ds.samples subset (aligned with y_eye / s_eye order)
    y_eye:   per-eye binary labels
    s_eye:   per-eye scores
    Returns: y_patient (0/1), s_patient (float), n_patients
    """
    per_patient_scores = defaultdict(list)
    per_patient_labels = {}  # patient-level any-abnormal (same for both eyes in ODIR file)

    for i, s in enumerate(samples):
        pid = s["pid"]
        per_patient_scores[pid].append(float(s_eye[i]))
        # patient-level label = same for both eyes in your construction
        per_patient_labels[pid] = int(y_eye[i]) if pid not in per_patient_labels else per_patient_labels[pid] | int(y_eye[i])

    pids = sorted(per_patient_scores.keys())
    s_patient = np.array([max(per_patient_scores[pid]) for pid in pids], dtype=np.float32)
    y_patient = np.array([per_patient_labels[pid] for pid in pids], dtype=np.int32)
    return y_patient, s_patient, len(pids)

def eval_patient_level(samples, y_eye, s_eye, target_spec=0.80):
    yP, sP, nP = patient_level_arrays(samples, y_eye, s_eye)
    print(f"  Patient-level eval on {nP} patients")
    if len(np.unique(yP)) < 2:
        print("  ⚠️ Only one class at patient-level; ROC/Spec80 undefined.")
        return {"Patient_AUC": float("nan")}
    aucP = roc_auc_score(yP, sP)
    thrP, specP, sensP = pick_thr_for_specificity(yP, sP, target_spec)
    tn, fp, fn, tp = confusion_matrix(yP, (sP >= thrP).astype(int), labels=[0,1]).ravel()
    return {
        "Patient_AUC": float(aucP),
        "Patient_Spec80_Threshold": float(thrP),
        "Patient_Specificity@Spec80": float(specP),
        "Patient_Sensitivity@Spec80": float(sensP),
        "Patient_TP": int(tp), "Patient_TN": int(tn), "Patient_FP": int(fp), "Patient_FN": int(fn),
    }

# ----------------------
# Core evaluation
# ----------------------
def eval_model_on_odir(model_name_disp, model_name_key, ckpt_path, thresholds_path, rfmid_labels,
                       odir_df, img_dir, device, batch_size=16,
                       pooling="auto", logitsum_T=1.0, odir_final_pooling="noisyor",
                       use_rfmid_thresholds=False, drisk_idx=None,
                       recalibrate_frac=0.0, use_f1max_threshold=False, seed=42):

    print(f"  Loading ODIR-5K dataset...")
    tform = vit_transforms(model_name_key, train=False)
    full_ds = ODIRAnyAbnDataset(odir_df, img_dir, tform)
    print(f"  ✅ Loaded {len(full_ds)} ODIR-5K images")
    
    # Count positive/negative samples
    pos_count = sum(1 for s in full_ds.samples if s["label"] == 1)
    neg_count = len(full_ds) - pos_count
    print(f"  Label distribution: {pos_count} abnormal, {neg_count} normal ({pos_count/len(full_ds)*100:.1f}% abnormal)")

    # Optional recalibration split (patient-level if patient IDs available)
    if recalibrate_frac and recalibrate_frac > 0.0:
        print(f"  Splitting data: {recalibrate_frac*100:.0f}% for calibration, {(1-recalibrate_frac)*100:.0f}% for testing...")
        indices = np.arange(len(full_ds))
        patient_ids = [full_ds.samples[i]["pid"] for i in indices]
        if all(pid is not None for pid in patient_ids):
            print(f"  Using patient-level splitting (prevents data leakage)")
            gss = GroupShuffleSplit(n_splits=1, test_size=1-recalibrate_frac, random_state=seed)
            calib_idx, test_idx = next(gss.split(indices, groups=patient_ids))
            calib_patients = len(set(patient_ids[i] for i in calib_idx))
            test_patients = len(set(patient_ids[i] for i in test_idx))
            print(f"  Split: {len(calib_idx)} images ({calib_patients} patients) calibration, {len(test_idx)} images ({test_patients} patients) test")
        else:
            print(f"  Using regular split (patient IDs not available)")
            calib_idx, test_idx = train_test_split(
                indices, test_size=1-recalibrate_frac, random_state=seed, shuffle=True, stratify=None
            )
            print(f"  Split: {len(calib_idx)} images calibration, {len(test_idx)} images test")
    else:
        calib_idx, test_idx = None, np.arange(len(full_ds))
        print(f"  Using all {len(full_ds)} images for evaluation (no calibration split)")

    def subset_loader(ds, idxs):
        if idxs is None:
            return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        subset = torch.utils.data.Subset(ds, idxs)
        return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    calib_loader = subset_loader(full_ds, calib_idx) if calib_idx is not None else None
    test_loader  = subset_loader(full_ds, test_idx)

    # Check checkpoint first to see if it has binary head
    if not Path(ckpt_path).exists():
        print(f"  ⚠️ Missing checkpoint: {ckpt_path}")
        return None
    print(f"  Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    state_dict = state.get("model_state_dict", state)
    
    # Check if checkpoint has binary_classifier (new model with binary head)
    has_binary_head_in_ckpt = any(k.startswith("binary_classifier.") for k in state_dict.keys())
    
    # Build model using RFMiD label size to match head
    # Exclude Disease_Risk from num_classes if it exists (for new model architecture)
    if "Disease_Risk" in rfmid_labels:
        num_classes = len(rfmid_labels) - 1  # Exclude Disease_Risk from multilabel head
        print(f"  Building model: {model_name_key} ({num_classes} disease classes + binary head)...")
    else:
        num_classes = len(rfmid_labels)
        print(f"  Building model: {model_name_key} ({num_classes} classes)...")
    
    # Build model with binary head if checkpoint has it
    model = build_model(model_name_key, num_classes=num_classes, include_binary_head=has_binary_head_in_ckpt).to(device)
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    # Compatibility check: Verify classifier weights were loaded
    # For new models: multilabel_classifier.* and binary_classifier.*
    # For old models: classifier.*
    loaded_multilabel_classifier = any(k.startswith("multilabel_classifier.") for k in state_dict.keys())
    loaded_binary_classifier = any(k.startswith("binary_classifier.") for k in state_dict.keys())
    loaded_old_classifier = any(k.startswith("classifier.") for k in state_dict.keys())
    loaded_any_classifier = loaded_multilabel_classifier or loaded_old_classifier
    
    # Check for TIMM's built-in head (head.* or fc.*, but not our custom classifier.*)
    has_timm_head = any(k.startswith("head.") or k.startswith("fc.") for k in state_dict.keys())
    
    if missing_keys:
        # Filter out expected missing keys for new architecture
        if has_binary_head_in_ckpt:
            # New model: missing keys might be expected if we're loading old-style checkpoint
            missing_multilabel = [k for k in missing_keys if k.startswith("multilabel_classifier.")]
            missing_binary = [k for k in missing_keys if k.startswith("binary_classifier.")]
            if missing_multilabel or missing_binary:
                print(f"  ⚠️ Missing classifier keys (new architecture):")
                if missing_multilabel:
                    print(f"     Missing multilabel_classifier keys: {missing_multilabel[:3]}..." if len(missing_multilabel) > 3 else f"     Missing multilabel_classifier keys: {missing_multilabel}")
                if missing_binary:
                    print(f"     Missing binary_classifier keys: {missing_binary[:3]}..." if len(missing_binary) > 3 else f"     Missing binary_classifier keys: {missing_binary}")
            else:
                # Old-style classifier keys missing (expected for new architecture)
                missing_old_classifier = [k for k in missing_keys if k.startswith("classifier.")]
                if missing_old_classifier and not loaded_multilabel_classifier:
                    print(f"  ⚠️ Missing old classifier.* keys (expected for new architecture with multilabel_classifier.*)")
        else:
            # Old model: check for missing classifier.*
            missing_classifier = [k for k in missing_keys if k.startswith("classifier.")]
            if missing_classifier:
                print(f"  ❌ CRITICAL: Missing classifier weights in checkpoint!")
                print(f"     Missing classifier keys: {missing_classifier[:5]}..." if len(missing_classifier) > 5 else f"     Missing classifier keys: {missing_classifier}")
                print(f"     This means the classifier head will be RANDOMLY INITIALIZED, causing poor performance!")
                print(f"     Checkpoint likely used TIMM's built-in head, not the custom MLP head.")
            else:
                print(f"  ⚠️ Missing keys in checkpoint: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"  ⚠️ Missing keys: {missing_keys}")
    
    if unexpected_keys:
        unexpected_timm_head = [k for k in unexpected_keys if k.startswith("head.") or k.startswith("fc.")]
        if unexpected_timm_head:
            print(f"  ⚠️ WARNING: Checkpoint contains TIMM built-in head keys (head.* or fc.*)")
            print(f"     Unexpected TIMM head keys: {unexpected_timm_head[:5]}..." if len(unexpected_timm_head) > 5 else f"     Unexpected TIMM head keys: {unexpected_timm_head}")
            print(f"     This suggests the checkpoint was saved with TIMM's built-in head, not the custom MLP.")
        else:
            print(f"  ⚠️ Unexpected keys in checkpoint: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"  ⚠️ Unexpected keys: {unexpected_keys}")
    
    # Final compatibility assertion
    if not loaded_any_classifier:
        print(f"  ❌ CRITICAL: No classifier weights found in checkpoint!")
        print(f"     Checked for: multilabel_classifier.*, classifier.*")
        print(f"     This means the checkpoint was saved with a different architecture (likely TIMM's built-in head).")
        print(f"     The classifier will be randomly initialized, leading to poor performance.")
        print(f"     Solution: Either retrain with the custom MLP head, or rebuild the eval model to match training.")
    else:
        if has_binary_head_in_ckpt:
            print(f"  ✅ Multilabel classifier weights present: {loaded_multilabel_classifier}")
            print(f"  ✅ Binary classifier weights present: {loaded_binary_classifier}")
        else:
            print(f"  ✅ Classifier weights present in checkpoint: {loaded_any_classifier}")
        if has_timm_head:
            print(f"  ⚠️ WARNING: Checkpoint contains both custom classifier.* and TIMM head.*/fc.* keys.")
            print(f"     This is unusual. The custom classifier will be used, but verify checkpoint compatibility.")
    
    model.eval()
    print(f"  ✅ Model loaded and set to eval mode")
    
    if has_binary_head_in_ckpt:
        print(f"  ✅ Binary Disease_Risk head detected in checkpoint - will use directly (no pooling needed!)")

    # Disease_Risk setup
    has_drisk = ("Disease_Risk" in rfmid_labels)
    if has_drisk:
        drisk_idx = rfmid_labels.index("Disease_Risk")
        print(f"  ✅ Disease_Risk found at index {drisk_idx}")

    # Decide pooling behavior
    # PRIORITY: If model has binary head, use it directly (best option)
    if pooling == "auto":
        if has_binary_head_in_ckpt:
            pooling_mode = "binary_head"  # Best option: use trained binary head directly
            print(f"  Pooling mode: {pooling_mode} (auto-selected: using trained binary Disease_Risk head)")
        else:
            # Prefer odir_map for ODIR-5K (aligns RFMiD→ODIR semantic space)
            pooling_mode = "odir_map"
            print(f"  Pooling mode: {pooling_mode} (auto-selected for ODIR-5K: aligns RFMiD→ODIR semantic space)")
    else:
        pooling_mode = pooling
        print(f"  Pooling mode: {pooling_mode} (user-specified)")
        # Validate that binary head exists if user forces "binary_head" mode
        if pooling_mode == "binary_head" and not has_binary_head_in_ckpt:
            raise ValueError(f"Binary head not found in checkpoint, but pooling='binary_head' was requested. Use a checkpoint trained with train_vit_update.py")
        # Validate that Disease_Risk exists if user forces "drisk" mode
        if pooling_mode == "drisk" and not has_drisk:
            raise ValueError(f"Disease_Risk not found in RFMiD labels, but pooling='drisk' was requested. Available labels: {rfmid_labels[:5]}...")
    
    # Build ODIR mapping if using odir_map pooling
    odir_mapping = None
    odir_active_classes = None  # Classes to use in Any-Abn pool (excludes empty ones)
    if pooling_mode == "odir_map":
        odir_mapping = build_odir_mapping(rfmid_labels)
        # Identify empty classes (except "C" which is intentionally empty)
        empty_classes = [k for k, v in odir_mapping.items() if len(v) == 0 and k != "C"]
        if empty_classes:
            print(f"  ⚠️ Warning: Empty ODIR classes found: {empty_classes}. These will be dropped from Any-Abn pool.")
            # Drop empty classes from the Any-Abn computation
            odir_active_classes = [k for k in ["D", "G", "C", "A", "H", "M", "O"] 
                                   if k not in empty_classes or k == "C"]
            print(f"  ✅ Using {len(odir_active_classes)} ODIR classes for Any-Abn: {odir_active_classes}")
        else:
            odir_active_classes = ["D", "G", "C", "A", "H", "M", "O"]  # All classes
        total_mapped = sum(len(v) for v in odir_mapping.values())
        print(f"  ✅ Built RFMiD→ODIR mapping ({total_mapped} RFMiD classes mapped)")
        print(f"  Final Any-Abn pooling: {odir_final_pooling}" + (f" (T={logitsum_T})" if odir_final_pooling == "logitsum" else ""))
        
        # Warn if using odir_map without recalibration (RFMiD thresholds don't apply)
        if recalibrate_frac == 0.0:
            print(f"  ⚠️ Warning: Using odir_map without recalibration (--recalibrate_frac=0).")
            print(f"     RFMiD thresholds don't apply to odir_map pooling. External ODIR is uncalibrated.")
            print(f"     Recommendation: Use --recalibrate_frac 0.2 for temperature scaling + threshold calibration.")

    # Optional RFMiD threshold (Disease_Risk only)
    rfmid_thr = None
    if use_rfmid_thresholds and thresholds_path is not None and Path(thresholds_path).exists():
        thr_array = np.load(thresholds_path)
        if has_drisk and drisk_idx is not None and drisk_idx < len(thr_array):
            rfmid_thr = float(thr_array[drisk_idx])
            print(f"  ✅ RFMiD Disease_Risk threshold: {rfmid_thr:.6f}")
        else:
            print("  ⚠️ Disease_Risk not available in thresholds; skipping fixed RFMiD threshold.")

    def collect_scores(loader, dataset_name="data", temperature=None, return_logits=False):
        """
        Collect scores (and optionally logits) from model inference.
        If temperature is provided, scales logits before computing probabilities.
        """
        y_true, y_score, all_logits = [], [], []
        total_batches = len(loader)
        print(f"  Running inference on {len(loader.dataset)} {dataset_name} images ({total_batches} batches)...")
        with torch.inference_mode():
            for batch_idx, (x, y) in enumerate(loader, 1):
                x = x.to(device)
                outputs = model(x)
                
                # Handle both single output (old model) and tuple output (new model with binary head)
                if isinstance(outputs, tuple):
                    multilabel_logits, binary_logits = outputs
                else:
                    multilabel_logits = outputs
                    binary_logits = None
                
                # Apply temperature scaling if provided
                if temperature is not None and temperature != 1.0:
                    if binary_logits is not None:
                        binary_logits = binary_logits / max(float(temperature), 1e-3)
                    multilabel_logits = multilabel_logits / max(float(temperature), 1e-3)
                
                # Store logits if requested
                if return_logits:
                    if binary_logits is not None:
                        all_logits.append(binary_logits.cpu().numpy())
                    else:
                        all_logits.append(multilabel_logits.cpu().numpy())
                
                # Use binary head directly if available (best option)
                if pooling_mode == "binary_head":
                    if binary_logits is None:
                        raise ValueError("binary_head pooling mode requested but model doesn't have binary head")
                    pooled = torch.sigmoid(binary_logits.squeeze()).cpu().numpy()  # [B]
                else:
                    probs = torch.sigmoid(multilabel_logits).cpu().numpy()  # [B, C]
                    if pooling_mode == "drisk":
                        pooled = probs[:, drisk_idx]
                    elif pooling_mode == "odir_map":
                        # Map RFMiD -> ODIR classes, then Any-Abn across active classes only
                        odir_probs = {}
                        for cls in ["D", "G", "C", "A", "H", "M", "O"]:
                            odir_probs[cls] = noisyor_over_indices(probs, odir_mapping[cls])
                        # Stack only active classes (drop empty ones except C)
                        odir_vec_list = [odir_probs[cls] for cls in odir_active_classes]
                        if odir_vec_list:
                            odir_vec = np.stack(odir_vec_list, axis=1)  # [B, num_active]
                            # Use specified final pooling method (noisyor or logitsum)
                            if odir_final_pooling == "logitsum":
                                pooled = pool_any(odir_vec, mode="logitsum", T=logitsum_T)
                            else:  # default: noisyor
                                pooled = 1.0 - np.prod(1.0 - np.clip(odir_vec, 1e-6, 1-1e-6), axis=1)
                        else:
                            # Fallback: if all classes are empty, use noisyor over all RFMiD classes
                            print("  ⚠️ All ODIR classes empty, falling back to noisyor over all RFMiD classes")
                            pooled = pool_any(probs, mode="noisyor")
                    elif pooling_mode == "noisyor":
                        pooled = pool_any(probs, mode="noisyor")
                    elif pooling_mode == "max":
                        pooled = pool_any(probs, mode="max")
                    elif pooling_mode == "logitsum":
                        pooled = pool_any(probs, mode="logitsum", T=logitsum_T)
                    else:
                        raise ValueError("Unknown pooling mode")
                y_score.append(pooled)
                y_true.append(y.numpy())
                
                # Progress indicator
                if batch_idx % 10 == 0 or batch_idx == total_batches:
                    processed = len(y_true) * batch_size
                    actual_processed = min(processed, len(loader.dataset))
                    print(f"    Processed {batch_idx}/{total_batches} batches (~{actual_processed} images)...", flush=True)
        
        y_true_concat = np.concatenate(y_true)
        y_score_concat = np.concatenate(y_score)
        print(f"  ✅ Inference complete: {len(y_true_concat)} images")
        
        if return_logits:
            logits_concat = np.vstack(all_logits)
            return y_true_concat, y_score_concat, logits_concat
        return y_true_concat, y_score_concat

    # Calibrate temperature and threshold on ODIR (optional)
    calib_thr = None
    temperature = None
    if calib_loader is not None:
        # Step 1: Fit temperature scaling on calibration logits
        print(f"\n  🌡️  Fitting temperature scaling on calibration set...")
        y_c_temp, _, logits_c = collect_scores(calib_loader, "calibration", return_logits=True)
        
        # Extract the pooled logit that will be thresholded
        if pooling_mode == "binary_head":
            # Binary head logits are already 1D [N]
            l_c_pooled = logits_c.squeeze() if logits_c.ndim > 1 else logits_c
        elif pooling_mode == "drisk":
            l_c_pooled = logits_c[:, drisk_idx]
        elif pooling_mode == "odir_map":
            # Compute pooled prob from unscaled logits, then convert to logit
            probs_c = 1.0 / (1.0 + np.exp(-logits_c))  # sigmoid
            odir_probs = {}
            for cls in ["D", "G", "C", "A", "H", "M", "O"]:
                odir_probs[cls] = noisyor_over_indices(probs_c, odir_mapping[cls])
            # Stack only active classes (drop empty ones except C)
            odir_vec_list = [odir_probs[cls] for cls in odir_active_classes]
            if odir_vec_list:
                odir_vec = np.stack(odir_vec_list, axis=1)
                # Use specified final pooling method (noisyor or logitsum)
                if odir_final_pooling == "logitsum":
                    pooled_prob = pool_any(odir_vec, mode="logitsum", T=logitsum_T)
                else:  # default: noisyor
                    pooled_prob = 1.0 - np.prod(1.0 - np.clip(odir_vec, 1e-6, 1-1e-6), axis=1)
            else:
                # Fallback: if all classes are empty, use noisyor over all RFMiD classes
                pooled_prob = pool_any(probs_c, mode="noisyor")
            # Convert pooled prob back to logit
            l_c_pooled = np.log(np.clip(pooled_prob, 1e-6, 1-1e-6) / np.clip(1 - pooled_prob, 1e-6, 1-1e-6))
        elif pooling_mode == "noisyor":
            probs_c = 1.0 / (1.0 + np.exp(-logits_c))
            pooled_prob = pool_any(probs_c, mode="noisyor")
            l_c_pooled = np.log(np.clip(pooled_prob, 1e-6, 1-1e-6) / np.clip(1 - pooled_prob, 1e-6, 1-1e-6))
        elif pooling_mode == "max":
            probs_c = 1.0 / (1.0 + np.exp(-logits_c))
            pooled_prob = pool_any(probs_c, mode="max")
            l_c_pooled = np.log(np.clip(pooled_prob, 1e-6, 1-1e-6) / np.clip(1 - pooled_prob, 1e-6, 1-1e-6))
        elif pooling_mode == "logitsum":
            probs_c = 1.0 / (1.0 + np.exp(-logits_c))
            pooled_prob = pool_any(probs_c, mode="logitsum", T=logitsum_T)
            l_c_pooled = np.log(np.clip(pooled_prob, 1e-6, 1-1e-6) / np.clip(1 - pooled_prob, 1e-6, 1-1e-6))
        else:
            raise ValueError("Unknown pooling mode for temperature scaling")
        
        # Fit temperature
        unique_classes_calib = np.unique(y_c_temp)
        if len(unique_classes_calib) >= 2:
            temperature = fit_temperature_binary(l_c_pooled, y_c_temp)
            print(f"  ✅ Learned temperature T={temperature:.3f}")
        else:
            print(f"  ⚠️ Cannot fit temperature: only one class in calibration set")
        
        # Step 2: Calibrate threshold on temperature-scaled calibration scores
        y_c, s_c = collect_scores(calib_loader, "calibration", temperature=temperature)
        # Guard calibration threshold computation
        if len(unique_classes_calib) < 2:
            print(f"  ⚠️ Warning: Only one class in calibration set, cannot compute threshold")
            calib_thr = None
        else:
            if use_f1max_threshold:
                print(f"\n  📊 Calibrating threshold on temperature-scaled calibration set (using F1max)...")
                f1max_c, thr_f1_c, prec_f1_c, rec_f1_c = compute_f1max(y_c, s_c)
                calib_thr = thr_f1_c
                sens_at_thr = rec_f1_c
                # Calculate specificity at F1max threshold
                fpr, tpr, thresholds = roc_curve(y_c, s_c)
                idx = np.argmin(np.abs(thresholds - thr_f1_c))
                spec_at_thr = 1.0 - fpr[idx] if idx < len(fpr) else 0.0
                print(f"  ✅ Calibrated threshold (F1max): {calib_thr:.6f} (gives F1={f1max_c:.3f}, prec={prec_f1_c:.3f}, rec={sens_at_thr:.3f}, spec={spec_at_thr:.3f} on calibration set)")
            else:
                print(f"\n  📊 Calibrating threshold on temperature-scaled calibration set (target: 80% specificity)...")
                calib_thr, spec_at_thr, sens_at_thr = pick_thr_for_specificity(y_c, s_c, target_spec=0.80)
                print(f"  ✅ Calibrated threshold: {calib_thr:.6f} (gives spec={spec_at_thr:.3f}, sens={sens_at_thr:.3f} on calibration set)")

    # Evaluate on test split (with temperature scaling if available)
    print(f"\n  📊 Evaluating on test set...")
    y_t, s_t = collect_scores(test_loader, "test", temperature=temperature)
    
    # Map the sampled indices back to sample metadata for patient-level evaluation
    if test_idx is not None:
        test_samples = [full_ds.samples[i] for i in test_idx]
    else:
        test_samples = full_ds.samples
    
    # Patient-level evaluation (max pooling of L/R scores per patient)
    print(f"\n  📊 Patient-level evaluation...")
    patient_metrics = eval_patient_level(test_samples, y_t, s_t, target_spec=0.80)
    print(f"  Patient-level AUC={patient_metrics['Patient_AUC']:.4f} | "
          f"Sens@Spec≈0.80={patient_metrics.get('Patient_Sensitivity@Spec80', float('nan')):.4f}")
    
    # Print score distribution
    print(f"  Score distribution: min={s_t.min():.4f}, p25={np.percentile(s_t, 25):.4f}, "
          f"median={np.median(s_t):.4f}, p75={np.percentile(s_t, 75):.4f}, max={s_t.max():.4f}")
    print(f"  Ground truth: {y_t.sum()}/{len(y_t)} abnormal ({y_t.mean()*100:.1f}%)")
    
    # Check for single-class edge case (e.g., patient-level split yields all abnormal)
    unique_classes = np.unique(y_t)
    single_class = (len(unique_classes) < 2)
    
    if single_class:
        print(f"  ⚠️ Only one class present in test split; ROC/Spec80 and F1max are undefined.")
        auc = float("nan")
        # Build metrics dict early with NaNs for thresholded views
        results = {
            "AUC": auc,
            "F1max": float("nan"),
            "F1max_Threshold": float("nan"),
            "F1max_Precision": float("nan"),
            "F1max_Recall": float("nan"),
        }
    else:
        print(f"  Computing metrics...")
        auc = roc_auc_score(y_t, s_t)
        # Compute F1max once (same for all thresholds)
        f1max, thr_f1, prec_f1, rec_f1 = compute_f1max(y_t, s_t)
        results = {
            "AUC": auc,
            "F1max": float(f1max),
            "F1max_Threshold": float(thr_f1),
            "F1max_Precision": float(prec_f1),
            "F1max_Recall": float(rec_f1),
        }
        print(f"  ✅ AUC: {auc:.4f}, F1max: {f1max:.4f} at threshold {thr_f1:.6f}")

    def eval_at_thr(name, thr):
        y_pred = (s_t >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_t, y_pred, labels=[0,1]).ravel()
        spec = tn / (tn + fp + 1e-8)
        sens = tp / (tp + fn + 1e-8)
        prec = tp / (tp + fp + 1e-8)
        rec  = sens
        acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        balanced_acc = 0.5 * (sens + spec)
        return {
            f"{name}_Threshold": float(thr),
            f"{name}_Specificity": float(spec),
            f"{name}_Sensitivity": float(sens),
            f"{name}_Precision": float(prec),
            f"{name}_Recall": float(rec),
            f"{name}_Accuracy": float(acc),
            f"{name}_Balanced_Accuracy": float(balanced_acc),
            f"{name}_TP": int(tp),
            f"{name}_TN": int(tn),
            f"{name}_FP": int(fp),
            f"{name}_FN": int(fn),
        }

    # Threshold-based metrics only computed if not single class
    if not single_class:
        print(f"\n  Computing metrics at different thresholds...")
        
        # Always compute F1max threshold metrics
        f1max_thr_metrics = eval_at_thr("F1max", thr_f1)
        results.update(f1max_thr_metrics)
        print(f"    F1max: threshold={thr_f1:.6f}, sens={f1max_thr_metrics['F1max_Sensitivity']:.3f}, spec={f1max_thr_metrics['F1max_Specificity']:.3f}, F1={f1max:.3f}")
        
        thr80, spec80, sens80 = pick_thr_for_specificity(y_t, s_t, target_spec=0.80)
        results.update(eval_at_thr("Spec80_onTest", thr80))
        print(f"    Spec80_onTest: threshold={thr80:.6f}, sens={sens80:.3f}, spec={spec80:.3f}")

        if rfmid_thr is not None and pooling_mode == "drisk":
            rfmid_metrics = eval_at_thr("RFMiD_thr", rfmid_thr)
            results.update(rfmid_metrics)
            print(f"    RFMiD_thr: threshold={rfmid_thr:.6f}, sens={rfmid_metrics['RFMiD_thr_Sensitivity']:.3f}, "
                  f"spec={rfmid_metrics['RFMiD_thr_Specificity']:.3f}, acc={rfmid_metrics['RFMiD_thr_Accuracy']:.3f}")
        elif rfmid_thr is not None:
            print(f"    ⚠️ Skipping RFMiD_thr metrics because pooling_mode != 'drisk' (threshold not comparable).")

        if calib_thr is not None:
            calib_metrics = eval_at_thr("ODIR_calib_thr", calib_thr)
            results.update(calib_metrics)
            print(f"    ODIR_calib_thr: threshold={calib_thr:.6f}, sens={calib_metrics['ODIR_calib_thr_Sensitivity']:.3f}, "
                  f"spec={calib_metrics['ODIR_calib_thr_Specificity']:.3f}, acc={calib_metrics['ODIR_calib_thr_Accuracy']:.3f}")

    # Add patient-level metrics to results
    results.update(patient_metrics)
    
    # Add temperature scaling info if used
    if temperature is not None:
        results["Temperature"] = float(temperature)

    return y_t, s_t, results, (calib_idx, test_idx)

# ----------------------
# Main
# ----------------------
def main(args):
    # Seed everything for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_root = Path("results") / "External" / "ODIR-5K"
    results_root.mkdir(parents=True, exist_ok=True)

    # RFMiD label schema from your training CSV
    rfmid_train = pd.read_csv(args.rfmid_train_csv)
    rfmid_labels = [c for c in rfmid_train.columns if c != "ID"]

    # Load ODIR annotations
    odir_df = pd.read_excel(args.odir_xlsx)

    # Models + paths (adjust if your tree differs)
    # All models now use the updated checkpoints with binary head (trained with train_vit_update.py)
    # Using binary Disease_Risk head checkpoint (best for any-abnormal detection)
    model_cfgs = {
        "SwinTiny":      {"name": "swin_tiny",      "ckpt": "results/ViT/SwinTiny/Updated/swin_tiny_rfmid_best_any_abnormal.pth", "thr": "results/ViT/SwinTiny/Updated/optimal_thresholds.npy"},
        "ViTSmall":      {"name": "vit_small",      "ckpt": "results/ViT/ViTSmall/Updated/vit_small_rfmid_best_any_abnormal.pth", "thr": "results/ViT/ViTSmall/Updated/optimal_thresholds.npy"},
        "DeiTSmall":     {"name": "deit_small",     "ckpt": "results/ViT/DeiTSmall/Updated/deit_small_rfmid_best_any_abnormal.pth", "thr": "results/ViT/DeiTSmall/Updated/optimal_thresholds.npy"},
        "CrossViTSmall": {"name": "crossvit_small", "ckpt": "results/ViT/CrossViTSmall/Updated/crossvit_small_rfmid_best_any_abnormal.pth", "thr": "results/ViT/CrossViTSmall/Updated/optimal_thresholds.npy"},
    }

    # Filter models if --model argument is provided
    if args.model:
        available_models = list(model_cfgs.keys())
        if args.model not in model_cfgs:
            print(f"❌ Model '{args.model}' not found. Available models: {available_models}")
            return
        model_cfgs = {args.model: model_cfgs[args.model]}
    
    summary = []
    for disp, cfg in model_cfgs.items():
        print(f"\n========== {disp} (Any-Abnormal on ODIR-5K) ==========")
        outdir = results_root / disp
        outdir.mkdir(parents=True, exist_ok=True)

        res = eval_model_on_odir(
            model_name_disp=disp,
            model_name_key=cfg["name"],
            ckpt_path=cfg["ckpt"],
            thresholds_path=(cfg["thr"] if args.use_rfmid_thresholds else None),
            rfmid_labels=rfmid_labels,
            odir_df=odir_df,
            img_dir=args.odir_img_dir,
            device=device,
            batch_size=args.batch_size,
            pooling=("auto" if args.pooling == "auto" else args.pooling),
            logitsum_T=args.logitsum_T,
            odir_final_pooling=args.odir_final_pooling,
            use_rfmid_thresholds=args.use_rfmid_thresholds,
            recalibrate_frac=args.recalibrate_frac,
            use_f1max_threshold=args.use_f1max_threshold,
            seed=args.seed
        )

        if res is None: 
            continue

        y_true, y_score, metrics, split_idxs = res

        # Save per-image outputs
        np.savez(outdir / "odir_anyabnormal_outputs.npz",
                 y_true=y_true.astype(np.int8),
                 y_score=y_score.astype(np.float32))

        # Write metrics CSV
        with open(outdir / "metrics_any_abnormal.csv", "w") as f:
            f.write("metric,value\n")
            for k,v in metrics.items():
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    f.write(f"{k},nan\n")
                elif isinstance(v, float):
                    f.write(f"{k},{v:.6f}\n")
                else:
                    f.write(f"{k},{v}\n")

        print(f"  AUC={metrics['AUC']:.4f}")
        if 'Patient_AUC' in metrics:
            print(f"  Patient-level AUC={metrics['Patient_AUC']:.4f}")
            if not np.isnan(metrics.get('Patient_Sensitivity@Spec80', np.nan)):
                print(f"  Patient-level @Spec≈0.80: sens={metrics['Patient_Sensitivity@Spec80']:.4f} spec={metrics['Patient_Specificity@Spec80']:.4f}")
        if 'RFMiD_thr_Sensitivity' in metrics:
            print(f"  @RFMiD thr: sens={metrics['RFMiD_thr_Sensitivity']:.4f} spec={metrics['RFMiD_thr_Specificity']:.4f}")
        if 'ODIR_calib_thr_Sensitivity' in metrics:
            print(f"  @ODIR-calib thr: sens={metrics['ODIR_calib_thr_Sensitivity']:.4f} spec={metrics['ODIR_calib_thr_Specificity']:.4f}")
        if 'Spec80_onTest_Sensitivity' in metrics:
            print(f"  @Spec≈0.80 on test: sens={metrics['Spec80_onTest_Sensitivity']:.4f} thr={metrics['Spec80_onTest_Threshold']:.4f}")
        else:
            print("  @Spec≈0.80 on test: n/a (single-class test split)")

        # Add to summary
        row = {"Model": disp}
        row.update({k: v for k,v in metrics.items() if isinstance(v, (int,float))})
        summary.append(row)

    if summary:
        pd.DataFrame(summary).to_csv(results_root / "summary_any_abnormal.csv", index=False)
        print(f"\n✅ Wrote summary: {results_root/'summary_any_abnormal.csv'}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--odir_xlsx", required=True, help="Path to ODIR-5K annotations Excel (V2).")
    p.add_argument("--odir_img_dir", required=True, help="Folder with ODIR training images (contains the Left/Right filenames).")
    p.add_argument("--rfmid_train_csv", required=True, help="Path to RFMiD training labels CSV to get label schema.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--pooling", choices=["auto","noisyor","max","logitsum","drisk","odir_map","binary_head"], default="auto",
                   help="auto: binary_head if available, else odir_map; 'binary_head': use trained binary Disease_Risk head directly (best for new models); 'odir_map': maps RFMiD classes into ODIR classes before Any-Abn.")
    p.add_argument("--logitsum_T", type=float, default=1.0, help="Temperature for logitsum pooling.")
    p.add_argument("--odir_final_pooling", choices=["noisyor", "logitsum"], default="noisyor",
                   help="Final pooling method for odir_map mode: 'noisyor' (default) or 'logitsum' (less sensitive to long tails, use with --logitsum_T > 1).")
    p.add_argument("--use_rfmid_thresholds", action="store_true",
                   help="If set, apply RFMiD per-class threshold for Disease_Risk from optimal_thresholds.npy.")
    p.add_argument("--recalibrate_frac", type=float, default=0.2,
                   help="Fraction of ODIR used ONLY to calibrate temperature scaling + threshold (no training). "
                        "Default 0.2. Set to 0.0 to disable. RECOMMENDED for odir_map pooling to lift sensitivity.")
    p.add_argument("--use_f1max_threshold", action="store_true",
                   help="Use F1max threshold instead of 80%% specificity for calibration. Better for screening (higher sensitivity).")
    p.add_argument("--model", type=str, default=None,
                   help="Evaluate only this model (e.g., 'SwinTiny', 'ViTSmall'). If not specified, evaluates all models.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
