# external_eval_odir_any.py
# Evaluate RFMiD-trained ViTs on ODIR-5K for Any-Abnormal only.
# - Prefers Disease_Risk head if available; otherwise supports pooling strategies.
# - Can apply RFMiD per-class threshold for Disease_Risk (from optimal_thresholds.npy)
#   or recalibrate a threshold on a small ODIR holdout (no weight updates).

import os, argparse
from pathlib import Path
from collections import defaultdict
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

def build_model(model_name, num_classes):
    model_name = model_name.lower()
    if model_name == "swin_tiny":
        backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0, drop_path_rate=0.2)
    elif model_name == "vit_small":
        backbone = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0, drop_path_rate=0.2)
    elif model_name == "deit_small":
        backbone = timm.create_model('deit_small_patch16_224', pretrained=True, num_classes=0, drop_path_rate=0.2)
    elif model_name == "crossvit_small":
        backbone = timm.create_model('crossvit_15_240', pretrained=True, num_classes=0, drop_path_rate=0.2)
    else:
        raise ValueError(f"Unknown ViT model name: {model_name}")
    in_f = backbone.num_features
    classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_f, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    class ViTWrapper(nn.Module):
        def __init__(self, b, head): super().__init__(); self.backbone=b; self.classifier=head
        def forward(self,x): return self.classifier(self.backbone(x))
    return ViTWrapper(backbone, classifier)

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
            if self.transform: img = self.transform(img)
            return img, int(s["label"])
        except Exception as e:
            raise RuntimeError(f"Failed to load image {s['img_path']}: {e}")

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
                       pooling="auto", logitsum_T=1.0,
                       use_rfmid_thresholds=False, drisk_idx=None,
                       recalibrate_frac=0.0, seed=42):

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
        # Try patient-level splitting to prevent data leakage
        patient_ids = [full_ds.samples[i]["pid"] for i in indices]
        if all(pid is not None for pid in patient_ids):
            # Use patient-level grouping
            print(f"  Using patient-level splitting (prevents data leakage)")
            gss = GroupShuffleSplit(n_splits=1, test_size=1-recalibrate_frac, random_state=seed)
            calib_idx, test_idx = next(gss.split(indices, groups=patient_ids))
            calib_idx = indices[calib_idx]
            test_idx = indices[test_idx]
            calib_patients = len(set(patient_ids[i] for i in calib_idx))
            test_patients = len(set(patient_ids[i] for i in test_idx))
            print(f"  Split: {len(calib_idx)} images ({calib_patients} patients) calibration, {len(test_idx)} images ({test_patients} patients) test")
        else:
            # Fallback to regular split if patient IDs not available
            print(f"  Using regular split (patient IDs not available)")
            calib_idx, test_idx = train_test_split(indices, test_size=1-recalibrate_frac, random_state=seed, shuffle=True, stratify=None)
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

    # Build model using RFMiD label size to match head
    num_classes = len(rfmid_labels)
    print(f"  Building model: {model_name_key} ({num_classes} classes)...")
    model = build_model(model_name_key, num_classes=num_classes).to(device)
    if not Path(ckpt_path).exists():
        print(f"  ⚠️ Missing checkpoint: {ckpt_path}")
        return None
    print(f"  Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    # Handle both "model_state_dict" and raw state dict formats
    state_dict = state.get("model_state_dict", state)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"  ⚠️ Missing keys in checkpoint: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"  ⚠️ Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"  ⚠️ Unexpected keys in checkpoint: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"  ⚠️ Unexpected keys: {unexpected_keys}")
    model.eval()
    print(f"  ✅ Model loaded and set to eval mode")

    # Disease_Risk setup
    has_drisk = ("Disease_Risk" in rfmid_labels)
    if has_drisk:
        drisk_idx = rfmid_labels.index("Disease_Risk")
        print(f"  ✅ Disease_Risk found at index {drisk_idx} (will use for any-abnormal pooling)")

    # Decide pooling behavior
    if pooling == "auto":
        pooling_mode = "drisk" if has_drisk else "noisyor"
        print(f"  Pooling mode: {pooling_mode} (auto-selected)")
    else:
        pooling_mode = pooling
        print(f"  Pooling mode: {pooling_mode} (user-specified)")
        # Validate that Disease_Risk exists if user forces "drisk" mode
        if pooling_mode == "drisk" and not has_drisk:
            raise ValueError(f"Disease_Risk not found in RFMiD labels, but pooling='drisk' was requested. Available labels: {rfmid_labels[:5]}...")

    # Optional RFMiD threshold (Disease_Risk only)
    rfmid_thr = None
    if use_rfmid_thresholds and thresholds_path is not None and Path(thresholds_path).exists():
        thr_array = np.load(thresholds_path)
        if has_drisk and drisk_idx is not None and drisk_idx < len(thr_array):
            rfmid_thr = float(thr_array[drisk_idx])
            print(f"  ✅ RFMiD Disease_Risk threshold: {rfmid_thr:.6f}")
        else:
            print("  ⚠️ Disease_Risk not available in thresholds; skipping fixed RFMiD threshold.")

    def collect_scores(loader, dataset_name="data"):
        y_true, y_score = [], []
        total_batches = len(loader)
        print(f"  Running inference on {len(loader.dataset)} {dataset_name} images ({total_batches} batches)...")
        with torch.inference_mode():
            for batch_idx, (x, y) in enumerate(loader, 1):
                x = x.to(device)
                logits = model(x)
                probs = torch.sigmoid(logits).cpu().numpy()  # [B, C]
                if pooling_mode == "drisk":
                    pooled = probs[:, drisk_idx]
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
        return y_true_concat, y_score_concat

    # Calibrate threshold on ODIR (optional)
    calib_thr = None
    if calib_loader is not None:
        print(f"\n  📊 Calibrating threshold on ODIR calibration set (target: 80% specificity)...")
        y_c, s_c = collect_scores(calib_loader, "calibration")
        # Guard calibration threshold computation
        unique_classes_calib = np.unique(y_c)
        if len(unique_classes_calib) < 2:
            print(f"  ⚠️ Warning: Only one class in calibration set, cannot compute threshold")
        else:
            calib_thr, spec_at_thr, sens_at_thr = pick_thr_for_specificity(y_c, s_c, target_spec=0.80)
            print(f"  ✅ Calibrated threshold: {calib_thr:.6f} (gives spec={spec_at_thr:.3f}, sens={sens_at_thr:.3f} on calibration set)")

    # Evaluate on test split
    print(f"\n  📊 Evaluating on test set...")
    y_t, s_t = collect_scores(test_loader, "test")
    
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
        # 1) Spec-80 threshold computed on the SAME ODIR test (for a pure AUC/curve summary)
        thr80, spec80, sens80 = pick_thr_for_specificity(y_t, s_t, target_spec=0.80)
        results.update(eval_at_thr("Spec80_onTest", thr80))
        print(f"    Spec80_onTest: threshold={thr80:.6f}, sens={sens80:.3f}, spec={spec80:.3f}")

        # 2) RFMiD Disease_Risk threshold (if provided and using Disease_Risk pooling)
        if rfmid_thr is not None and pooling_mode == "drisk":
            rfmid_metrics = eval_at_thr("RFMiD_thr", rfmid_thr)
            results.update(rfmid_metrics)
            print(f"    RFMiD_thr: threshold={rfmid_thr:.6f}, sens={rfmid_metrics['RFMiD_thr_Sensitivity']:.3f}, "
                  f"spec={rfmid_metrics['RFMiD_thr_Specificity']:.3f}, acc={rfmid_metrics['RFMiD_thr_Accuracy']:.3f}")
        elif rfmid_thr is not None:
            print(f"    ⚠️ Skipping RFMiD_thr metrics because pooling_mode != 'drisk' (threshold not comparable).")

        # 3) ODIR recalibrated threshold (if used)
        if calib_thr is not None:
            calib_metrics = eval_at_thr("ODIR_calib_thr", calib_thr)
            results.update(calib_metrics)
            print(f"    ODIR_calib_thr: threshold={calib_thr:.6f}, sens={calib_metrics['ODIR_calib_thr_Sensitivity']:.3f}, "
                  f"spec={calib_metrics['ODIR_calib_thr_Specificity']:.3f}, acc={calib_metrics['ODIR_calib_thr_Accuracy']:.3f}")

    # Add patient-level metrics to results
    results.update(patient_metrics)

    return y_t, s_t, results, (calib_idx, test_idx)

# ----------------------
# Main
# ----------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_root = Path("results") / "External" / "ODIR-5K"
    results_root.mkdir(parents=True, exist_ok=True)

    # RFMiD label schema from your training CSV
    rfmid_train = pd.read_csv(args.rfmid_train_csv)
    rfmid_labels = [c for c in rfmid_train.columns if c != "ID"]

    # Load ODIR annotations
    odir_df = pd.read_excel(args.odir_xlsx)

    # Models + paths (adjust if your tree differs)
    model_cfgs = {
        "SwinTiny":      {"name": "swin_tiny",      "ckpt": "results/ViT/SwinTiny/swin_tiny_rfmid_best.pth",      "thr": "results/ViT/SwinTiny/optimal_thresholds.npy"},
        "ViTSmall":      {"name": "vit_small",      "ckpt": "results/ViT/ViTSmall/vit_small_rfmid_best.pth",      "thr": "results/ViT/ViTSmall/optimal_thresholds.npy"},
        "DeiTSmall":     {"name": "deit_small",     "ckpt": "results/ViT/DeiTSmall/deit_small_rfmid_best.pth",    "thr": "results/ViT/DeiTSmall/optimal_thresholds.npy"},
        "CrossViTSmall": {"name": "crossvit_small", "ckpt": "results/ViT/CrossViTSmall/crossvit_small_rfmid_best.pth","thr":"results/ViT/CrossViTSmall/optimal_thresholds.npy"},
    }

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
            use_rfmid_thresholds=args.use_rfmid_thresholds,
            recalibrate_frac=args.recalibrate_frac,
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
    p.add_argument("--pooling", choices=["auto","noisyor","max","logitsum","drisk"], default="auto",
                   help="auto: Disease_Risk if present else Noisy-OR; or force a pooling mode.")
    p.add_argument("--logitsum_T", type=float, default=1.0, help="Temperature for logitsum pooling.")
    p.add_argument("--use_rfmid_thresholds", action="store_true",
                   help="If set, apply RFMiD per-class threshold for Disease_Risk from optimal_thresholds.npy.")
    p.add_argument("--recalibrate_frac", type=float, default=0.0,
                   help="Fraction of ODIR used ONLY to calibrate a spec≈0.80 threshold (no training). 0 disables.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
