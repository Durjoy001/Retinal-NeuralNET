# -*- coding: utf-8 -*-
"""
External evaluation on ODIR-5K for models trained on RFMiD.

Aligned with eval_messidor2.py pipeline:
- Uses training-aligned preprocessing via adapter pattern
- Uses Disease_Risk head from multilabel classifier (required)
- Uses fixed RFMiD validation threshold (standard approach)

Usage:
    python -m src.external_evaluation.eval_odir5k \
        --model_name swin_tiny \
        --checkpoint results/ViT/SwinTiny/swin_tiny_rfmid_best.pth \
        --thresholds results/ViT/SwinTiny/optimal_thresholds.npy \
        --rfmid_train_csv data/RFMiD_Challenge_Dataset/2. Groundtruths/a. RFMiD_Training_Labels.csv \
        --odir_xlsx data/External_Dataset/ODIR-5k/ODIR-5K_Training_Annotations(Updated)_V2.xlsx \
        --odir_img_dir data/External_Dataset/ODIR-5k/ODIR-5K_Training_Dataset \
        --results_dir results/External/ODIR-5K/SwinTiny
"""

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

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix

# Import adapter classes from eval_messidor2.py
from .eval_messidor2 import (
    ModelAdapter, TimmBackboneWithHead, CNNAdapter, CLIPAdapter, SigLIPAdapter
)

# Registry: map model_name -> adapter factory (same as eval_messidor2.py)
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
def eval_model_on_odir(model_name_key, ckpt_path, thresholds_path, rfmid_labels,
                       odir_df, img_dir, device, batch_size=16, seed=42):

    # Load RFMiD thresholds
    if not Path(thresholds_path).exists():
        raise ValueError(f"Threshold file not found: {thresholds_path}")
    thresholds = np.load(thresholds_path)
    
    # Find Disease_Risk index (required for "any abnormal" detection)
    # Disease_Risk = any abnormal vs normal (binary classification)
    if "Disease_Risk" not in rfmid_labels:
        raise ValueError(f"Disease_Risk column not found in RFMiD labels. Available columns: {rfmid_labels[:10]}...")
    
    disease_risk_idx = rfmid_labels.index("Disease_Risk")
    print(f"✅ Found Disease_Risk column at index {disease_risk_idx} (required for any abnormal detection)")
    
    # Thresholds file handling: expect thresholds for all classes (including Disease_Risk)
    expected_threshold_count = len(rfmid_labels)
    if len(thresholds) != expected_threshold_count:
        raise ValueError(f"Threshold length mismatch: expected {expected_threshold_count} thresholds (for {len(rfmid_labels)} label columns), "
                        f"but got {len(thresholds)} thresholds in file.")
    
    # Use Disease_Risk threshold from RFMiD validation set (always use this fixed threshold)
    tau_dr = float(thresholds[disease_risk_idx])
    print(f"✅ Using RFMiD validation threshold for Disease_Risk: {tau_dr:.6f}")
    
    # Load checkpoint
    if not Path(ckpt_path).exists():
        print(f"  ⚠️ Missing checkpoint: {ckpt_path}")
        return None
    print(f"  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    
    # Build model via adapter - always use multilabel classifier (includes DR class)
    num_classes = len(rfmid_labels)
    model_name_lower = model_name_key.lower()
    
    # Special handling for CLIP and SigLIP (need class_names)
    if "clip" in model_name_lower or "biomedclip" in model_name_lower:
        try:
            from .eval_messidor2 import HAS_OPENCLIP
            if not HAS_OPENCLIP:
                raise ImportError("open_clip required for CLIP models")
        except ImportError:
            raise ImportError("open_clip required for CLIP models")
        adapter = CLIPAdapter(model_name_lower, num_classes, rfmid_labels)
    elif "siglip" in model_name_lower:
        adapter = SigLIPAdapter(model_name_lower, num_classes, rfmid_labels)
    elif model_name_lower in REGISTRY:
        adapter = REGISTRY[model_name_lower](num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name_key}. Available: {list(REGISTRY.keys())}")
    
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
    
    missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
    if missing_keys:
        print(f"⚠️  Warning: Missing keys: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"⚠️  Warning: Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"⚠️  Warning: Unexpected keys: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"⚠️  Warning: Unexpected keys: {unexpected_keys}")
    
    model.eval()
    print(f"  ✅ Model loaded and set to eval mode")
    
    # Use adapter's transforms for all models to match training pipeline
    tfm = adapter.transforms(is_train=False)
    if model_name_lower in ["densenet121", "resnet50", "efficientnet_b3", "inception_v3"]:
        print(f"✅ Using pretrained weights transforms for CNN model (matches train_cnn.py):")
        print(f"   ImageNet standard preprocessing from pretrained weights")
    elif model_name_lower in ["swin_tiny", "vit_small", "deit_small", "crossvit_small", "coatnet0", "maxvit_tiny"]:
        print(f"✅ Using timm transforms for {model_name_lower} (matches train_vit.py / train_hybrid.py)")
    elif "clip" in model_name_lower or "biomedclip" in model_name_lower:
        print(f"✅ Using CLIP transforms (matches train_vlm.py): CLIP normalization, size 224/336")
    elif "siglip" in model_name_lower:
        print(f"✅ Using SigLIP transforms (matches train_vlm.py): SigLIP normalization from timm config")
    
    # Load ODIR-5K dataset
    print(f"  Loading ODIR-5K dataset...")
    full_ds = ODIRAnyAbnDataset(odir_df, img_dir, tfm)
    print(f"  ✅ Loaded {len(full_ds)} ODIR-5K images")
    
    # Count positive/negative samples
    pos_count = sum(1 for s in full_ds.samples if s["label"] == 1)
    neg_count = len(full_ds) - pos_count
    print(f"  Label distribution: {pos_count} abnormal, {neg_count} normal ({pos_count/len(full_ds)*100:.1f}% abnormal)")
    
    # Use all data for evaluation (no calibration split - standard approach)
    test_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # Always use Disease_Risk head from multilabel classifier with RFMiD threshold
    print(f"  Using Disease_Risk head from multilabel classifier for ODIR-5K (any abnormal detection)")
    print(f"  Using RFMiD validation threshold: {tau_dr:.6f}")

    # Run inference on test set
    print(f"\n  📊 Running inference on test set...")
    all_logits = []
    all_ids = []
    y_true_list = []
    total_batches = len(test_loader)
    
    use_amp = torch.cuda.is_available() or (torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False)
    dev_type = "cuda" if torch.cuda.is_available() else ("mps" if use_amp and hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    
    with torch.inference_mode(), torch.autocast(device_type=dev_type, enabled=(dev_type != "cpu")):
        for batch_idx, (xb, yb) in enumerate(test_loader, 1):
            xb = xb.to(device)
            outputs = model(xb)
            
            # Handle tuple output (take first element if tuple)
            if isinstance(outputs, tuple):
                multilabel_logits = outputs[0]
            else:
                multilabel_logits = outputs
            if hasattr(multilabel_logits, "logits"):
                multilabel_logits = multilabel_logits.logits
            elif isinstance(multilabel_logits, (tuple, list)):
                multilabel_logits = multilabel_logits[0]
            
            all_logits.append(multilabel_logits.cpu())
            y_true_list.append(yb.numpy())
            
            # Progress indicator
            if batch_idx % 10 == 0 or batch_idx == total_batches:
                processed = batch_idx * batch_size
                actual_processed = min(processed, len(test_loader.dataset))
                print(f"    Processed {batch_idx}/{total_batches} batches (~{actual_processed} images)...", flush=True)
    
    all_logits = torch.cat(all_logits, dim=0)
    all_probs = torch.sigmoid(all_logits).numpy()
    y_t = np.concatenate(y_true_list)
    
    # Use Disease_Risk head from multilabel classifier
    disease_risk_scores = all_probs[:, disease_risk_idx]
    any_scores = disease_risk_scores  # For ODIR-5K, "any abnormal" = Disease_Risk head
    
    print(f"✅ Inference complete: {all_probs.shape[0]} images, {all_probs.shape[1]} classes.")
    print(f"  Using Disease_Risk head from multilabel classifier for any abnormal detection")
    
    # Debug: Print threshold and score distribution
    print(f"\n[Debug] tau_disease_risk (RFMiD): {tau_dr:.6f}")
    print(f"[Debug] disease_risk_scores (test): min={disease_risk_scores.min():.4f}  p25={np.percentile(disease_risk_scores,25):.4f}  "
          f"median={np.median(disease_risk_scores):.4f}  p75={np.percentile(disease_risk_scores,75):.4f}  max={disease_risk_scores.max():.4f}")
    print(f"[Debug] % predicted positive @tau_disease_risk: {(disease_risk_scores >= tau_dr).mean()*100:.2f}%")
    print(f"[Debug] Ground truth (test): {y_t.sum()}/{len(y_t)} positive ({y_t.mean()*100:.2f}%)")
    
    s_t = any_scores
    
    # Patient-level evaluation (max pooling of L/R scores per patient)
    print(f"\n  📊 Patient-level evaluation...")
    test_samples = full_ds.samples
    patient_metrics = eval_patient_level(test_samples, y_t, s_t, target_spec=0.80)
    print(f"  Patient-level AUC={patient_metrics['Patient_AUC']:.4f} | "
          f"Sens@Spec≈0.80={patient_metrics.get('Patient_Sensitivity@Spec80', float('nan')):.4f}")
    
    # Check for single-class edge case
    unique_classes = np.unique(y_t)
    single_class = (len(unique_classes) < 2)
    
    if single_class:
        print(f"  ⚠️ Only one class present in test set; ROC/Spec80 and F1max are undefined.")
        auc = float("nan")
        f1max = float("nan")
        thr_f1 = float("nan")
        prec_f1 = float("nan")
        rec_f1 = float("nan")
    else:
        print(f"  Computing metrics...")
        auc = roc_auc_score(y_t, s_t)
        f1max, thr_f1, prec_f1, rec_f1 = compute_f1max(y_t, s_t)
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

    # Build results dictionary
    results = {
        "AUC": auc if not single_class else float("nan"),
        "F1max": float(f1max) if not single_class else float("nan"),
        "F1max_Threshold": float(thr_f1) if not single_class else float("nan"),
        "F1max_Precision": float(prec_f1) if not single_class else float("nan"),
        "F1max_Recall": float(rec_f1) if not single_class else float("nan"),
    }
    
    # Threshold-based metrics only computed if not single class
    if not single_class:
        print(f"\n  Computing metrics at different thresholds...")
        
        # Always compute F1max threshold metrics
        f1max_thr_metrics = eval_at_thr("F1max", thr_f1)
        results.update(f1max_thr_metrics)
        print(f"    F1max: threshold={thr_f1:.6f}, sens={f1max_thr_metrics['F1max_Sensitivity']:.3f}, spec={f1max_thr_metrics['F1max_Specificity']:.3f}, F1={f1max:.3f}")
        
        # Compute Spec80 threshold metrics
        thr80, spec80, sens80 = pick_thr_for_specificity(y_t, s_t, target_spec=0.80)
        results.update(eval_at_thr("Spec80_onTest", thr80))
        print(f"    Spec80_onTest: threshold={thr80:.6f}, sens={sens80:.3f}, spec={spec80:.3f}")

        # Always use RFMiD threshold (standard approach)
        print(f"    RFMiD_thr: threshold={tau_dr:.6f} (fixed from RFMiD validation)")
        rfmid_metrics = eval_at_thr("RFMiD_thr", tau_dr)
        results.update(rfmid_metrics)
        print(f"    RFMiD_thr: sens={rfmid_metrics['RFMiD_thr_Sensitivity']:.3f}, "
              f"spec={rfmid_metrics['RFMiD_thr_Specificity']:.3f}, acc={rfmid_metrics['RFMiD_thr_Accuracy']:.3f}")

    # Add patient-level metrics to results
    results.update(patient_metrics)
    results["threshold_rfmid_disease_risk"] = float(tau_dr)
    results["threshold_used"] = float(tau_dr)
    results["threshold_type"] = "rfmid"
    results["head_used"] = "Disease_Risk"

    return y_t, s_t, results

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

    # Models + paths (all 12 models: CNN, ViT, Hybrid, VLM)
    model_cfgs = {
        # CNNs
        "Densenet121":   {"name": "densenet121",    "ckpt": "results/CNN/Densenet121/densenet121_rfmid_best.pth", "thr": "results/CNN/Densenet121/optimal_thresholds.npy"},
        "ResNet50":      {"name": "resnet50",       "ckpt": "results/CNN/ResNet50/resnet50_rfmid_best.pth", "thr": "results/CNN/ResNet50/optimal_thresholds.npy"},
        "EfficientNetB3": {"name": "efficientnet_b3", "ckpt": "results/CNN/EfficientNetB3/efficientnet_b3_rfmid_best.pth", "thr": "results/CNN/EfficientNetB3/optimal_thresholds.npy"},
        "InceptionV3":   {"name": "inception_v3",    "ckpt": "results/CNN/InceptionV3/inception_v3_rfmid_best.pth", "thr": "results/CNN/InceptionV3/optimal_thresholds.npy"},
        
        # ViTs
        "SwinTiny":      {"name": "swin_tiny",      "ckpt": "results/ViT/SwinTiny/swin_tiny_rfmid_best.pth", "thr": "results/ViT/SwinTiny/optimal_thresholds.npy"},
        "ViTSmall":      {"name": "vit_small",      "ckpt": "results/ViT/ViTSmall/vit_small_rfmid_best.pth", "thr": "results/ViT/ViTSmall/optimal_thresholds.npy"},
        "DeiTSmall":     {"name": "deit_small",     "ckpt": "results/ViT/DeiTSmall/deit_small_rfmid_best.pth", "thr": "results/ViT/DeiTSmall/optimal_thresholds.npy"},
        "CrossViTSmall": {"name": "crossvit_small", "ckpt": "results/ViT/CrossViTSmall/crossvit_small_rfmid_best.pth", "thr": "results/ViT/CrossViTSmall/optimal_thresholds.npy"},
        
        # Hybrids
        "CoAtNet0":      {"name": "coatnet0",       "ckpt": "results/Hybrid/CoAtNet0/coatnet0_rfmid_best.pth", "thr": "results/Hybrid/CoAtNet0/optimal_thresholds.npy"},
        "MaxViTTiny":    {"name": "maxvit_tiny",    "ckpt": "results/Hybrid/MaxViTTiny/maxvit_tiny_rfmid_best.pth", "thr": "results/Hybrid/MaxViTTiny/optimal_thresholds.npy"},
        
        # VLMs
        "CLIPViTB16":    {"name": "clip_vit_b16",   "ckpt": "results/VLM/CLIPViTB16/clip_vit_b16_rfmid_best.pth", "thr": "results/VLM/CLIPViTB16/optimal_thresholds.npy"},
        "SigLIPBase384": {"name": "siglip_base_384", "ckpt": "results/VLM/SigLIPBase384/siglip_base_384_rfmid_best.pth", "thr": "results/VLM/SigLIPBase384/optimal_thresholds.npy"},
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
            model_name_key=cfg["name"],
            ckpt_path=cfg["ckpt"],
            thresholds_path=cfg["thr"],
            rfmid_labels=rfmid_labels,
            odir_df=odir_df,
            img_dir=args.odir_img_dir,
            device=device,
            batch_size=args.batch_size,
            seed=args.seed
        )

        if res is None: 
            continue

        y_true, y_score, metrics = res

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
    p = argparse.ArgumentParser(description="Evaluate RFMiD-trained models on ODIR-5K (aligned with eval_messidor2.py)")
    p.add_argument("--odir_xlsx", required=True, help="Path to ODIR-5K annotations Excel (V2).")
    p.add_argument("--odir_img_dir", required=True, help="Folder with ODIR training images (contains the Left/Right filenames).")
    p.add_argument("--rfmid_train_csv", required=True, help="Path to RFMiD training labels CSV to get label schema.")
    p.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    p.add_argument("--model", type=str, default=None,
                   help="Evaluate only this model. Available: Densenet121, ResNet50, EfficientNetB3, InceptionV3, "
                        "SwinTiny, ViTSmall, DeiTSmall, CrossViTSmall, CoAtNet0, MaxViTTiny, CLIPViTB16, SigLIPBase384. "
                        "If not specified, evaluates all 12 models.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = p.parse_args()
    main(args)
