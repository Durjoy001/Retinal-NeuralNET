#!/usr/bin/env python3
"""
Comprehensive statistical tests (DeLong and McNemar) for all 12 models.
Performs pairwise comparisons between all models.
"""

import sys
import os
import math
from pathlib import Path
import numpy as np
import pandas as pd

# =========================
#  Statistical test functions (embedded from statistical_tests.py)
# =========================

def _norm_sf(z):
    """Survival function 1 - CDF for standard normal, no scipy needed."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _compute_theta(pos_scores, neg_scores):
    """Pairwise comparison matrix theta_ij."""
    pos = pos_scores[:, None]
    neg = neg_scores[None, :]
    greater = (pos > neg).astype(float)
    equal = (pos == neg).astype(float)
    theta = greater + 0.5 * equal
    return theta


def delong_two_sample_auc(y_true, scores1, scores2):
    """DeLong test for two correlated ROC AUCs."""
    y_true = np.asarray(y_true)
    scores1 = np.asarray(scores1)
    scores2 = np.asarray(scores2)

    if y_true.ndim != 1:
        raise ValueError("y_true must be 1D")
    if scores1.shape != y_true.shape or scores2.shape != y_true.shape:
        raise ValueError("scores and y_true must have the same shape")

    pos_idx = y_true == 1
    neg_idx = y_true == 0

    n_pos = np.sum(pos_idx)
    n_neg = np.sum(neg_idx)

    if n_pos == 0 or n_neg == 0:
        raise ValueError("DeLong requires at least one positive and one negative sample.")

    pos1, neg1 = scores1[pos_idx], scores1[neg_idx]
    pos2, neg2 = scores2[pos_idx], scores2[neg_idx]

    theta1 = _compute_theta(pos1, neg1)
    theta2 = _compute_theta(pos2, neg2)

    auc1 = np.mean(theta1)
    auc2 = np.mean(theta2)

    V10_1 = np.mean(theta1, axis=1)
    V10_2 = np.mean(theta2, axis=1)
    V01_1 = np.mean(theta1, axis=0)
    V01_2 = np.mean(theta2, axis=0)

    def _sample_var(x):
        if len(x) <= 1:
            return 0.0
        return np.var(x, ddof=1)

    def _sample_cov(x, y):
        if len(x) <= 1:
            return 0.0
        return np.cov(x, y, ddof=1)[0, 1]

    s10_1 = _sample_var(V10_1)
    s10_2 = _sample_var(V10_2)
    s10_12 = _sample_cov(V10_1, V10_2)
    s01_1 = _sample_var(V01_1)
    s01_2 = _sample_var(V01_2)
    s01_12 = _sample_cov(V01_1, V01_2)

    var1 = s10_1 / n_pos + s01_1 / n_neg
    var2 = s10_2 / n_pos + s01_2 / n_neg
    cov12 = s10_12 / n_pos + s01_12 / n_neg

    return auc1, auc2, var1, var2, cov12


def delong_roc_test(y_true, scores1, scores2):
    """Perform DeLong test for the null hypothesis: AUC1 == AUC2."""
    auc1, auc2, var1, var2, cov12 = delong_two_sample_auc(y_true, scores1, scores2)
    diff = auc1 - auc2
    var_diff = var1 + var2 - 2.0 * cov12

    if var_diff <= 0:
        se_diff = 0.0
        z = 0.0
        p_value = 1.0
    else:
        se_diff = math.sqrt(var_diff)
        z = diff / se_diff
        p_value = 2.0 * _norm_sf(abs(z))

    return {
        "auc1": auc1,
        "auc2": auc2,
        "diff": diff,
        "se_diff": se_diff,
        "z": z,
        "p_value": p_value,
    }


def mcnemar_test(y_true, pred1, pred2, exact=False):
    """McNemar test for paired binary predictions of two models."""
    y_true = np.asarray(y_true)
    p1 = np.asarray(pred1)
    p2 = np.asarray(pred2)

    if p1.shape != p2.shape:
        raise ValueError("pred1 and pred2 must have the same shape")

    b = int(np.sum((p1 == 1) & (p2 == 0)))
    c = int(np.sum((p1 == 0) & (p2 == 1)))
    n = b + c

    if n == 0:
        return {
            "b": b,
            "c": c,
            "statistic": 0.0,
            "p_value": 1.0,
            "method": "no discordant pairs (identical predictions)",
        }

    if exact and n <= 25:
        k = min(b, c)
        from math import comb
        p_le_k = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
        p_ge_n_minus_k = sum(comb(n, i) for i in range(n - k, n + 1)) * (0.5 ** n)
        p_value = 2.0 * min(p_le_k, p_ge_n_minus_k)
        p_value = min(p_value, 1.0)
        return {
            "b": b,
            "c": c,
            "statistic": None,
            "p_value": p_value,
            "method": "exact binomial McNemar",
        }

    diff = abs(b - c)
    statistic = (diff - 1.0) ** 2 / float(n)
    z = math.sqrt(statistic)
    p_value = 2.0 * _norm_sf(z)

    return {
        "b": b,
        "c": c,
        "statistic": statistic,
        "p_value": p_value,
        "method": "chi-square McNemar with continuity correction",
    }


def load_any_abnormal_npz(path):
    """Helper to load test outputs from an .npz file."""
    data = np.load(path)
    y_true = data["y_true"]
    y_score = data["y_score"]
    return y_true, y_score

# Define all 12 models with their file paths and prefixes
MODELS = {
    # CNN models
    "Densenet121": {
        "path": "results/CNN/Densenet121",
        "file": "cnn_anyabnormal_test_outputs.npz"
    },
    "EfficientNetB3": {
        "path": "results/CNN/EfficientNetB3",
        "file": "cnn_anyabnormal_test_outputs.npz"
    },
    "InceptionV3": {
        "path": "results/CNN/InceptionV3",
        "file": "cnn_anyabnormal_test_outputs.npz"
    },
    "ResNet50": {
        "path": "results/CNN/ResNet50",
        "file": "cnn_anyabnormal_test_outputs.npz"
    },
    # ViT models
    "CrossViTSmall": {
        "path": "results/ViT/CrossViTSmall",
        "file": "vit_anyabnormal_test_outputs.npz"
    },
    "DeiTSmall": {
        "path": "results/ViT/DeiTSmall",
        "file": "vit_anyabnormal_test_outputs.npz"
    },
    "SwinTiny": {
        "path": "results/ViT/SwinTiny",
        "file": "vit_anyabnormal_test_outputs.npz"
    },
    "ViTSmall": {
        "path": "results/ViT/ViTSmall",
        "file": "vit_anyabnormal_test_outputs.npz"
    },
    # Hybrid models
    "CoAtNet0": {
        "path": "results/Hybrid/CoAtNet0",
        "file": "hybrid_anyabnormal_test_outputs.npz"
    },
    "MaxViTTiny": {
        "path": "results/Hybrid/MaxViTTiny",
        "file": "hybrid_anyabnormal_test_outputs.npz"
    },
    # VLM models
    "CLIPViTB16": {
        "path": "results/VLM/CLIPViTB16",
        "file": "vlm_anyabnormal_test_outputs.npz"
    },
    "SigLIPBase384": {
        "path": "results/VLM/SigLIPBase384",
        "file": "vlm_anyabnormal_test_outputs.npz"
    },
}


def load_f1max_threshold(model_path):
    """Load F1max threshold from overall_any_abnormal_metrics.csv"""
    metrics_file = Path(model_path) / "overall_any_abnormal_metrics.csv"
    if not metrics_file.exists():
        return None
    
    try:
        df = pd.read_csv(metrics_file)
        # Convert to dictionary
        metrics_dict = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
        threshold = float(metrics_dict.get("F1max_Threshold", 0))
        return threshold
    except Exception as e:
        print(f"Warning: Could not load threshold for {model_path}: {e}")
        return None


def load_all_models():
    """Load test outputs and thresholds for all models."""
    base_dir = Path(__file__).parent.parent
    y_true_shared = None
    scores = {}
    thresholds = {}
    
    print("Loading all model outputs...")
    for model_name, config in MODELS.items():
        model_path = base_dir / config["path"]
        npz_file = model_path / config["file"]
        
        if not npz_file.exists():
            print(f"Warning: {npz_file} not found, skipping {model_name}")
            continue
        
        try:
            y_true, y_score = load_any_abnormal_npz(str(npz_file))
            scores[model_name] = y_score
            
            # Verify y_true consistency
            if y_true_shared is None:
                y_true_shared = y_true
            else:
                if not np.array_equal(y_true_shared, y_true):
                    print(f"Warning: y_true mismatch for {model_name}")
            
            # Load F1max threshold
            threshold = load_f1max_threshold(model_path)
            if threshold is not None:
                thresholds[model_name] = threshold
            else:
                print(f"Warning: Could not load threshold for {model_name}, using default 0.5")
                thresholds[model_name] = 0.5
            
            print(f"  ✓ {model_name}: {len(y_score)} samples, threshold={thresholds[model_name]:.4f}")
            
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            continue
    
    print(f"\nLoaded {len(scores)} models successfully.\n")
    return y_true_shared, scores, thresholds


def perform_delong_tests(y_true, scores):
    """Perform pairwise DeLong tests for all model pairs."""
    print("=" * 100)
    print("DELONG TEST RESULTS (Pairwise AUC Comparisons)")
    print("=" * 100)
    
    model_names = sorted(scores.keys())
    results = []
    
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i >= j:  # Only do upper triangle (avoid duplicates and self-comparisons)
                continue
            
            try:
                res = delong_roc_test(y_true, scores[model1], scores[model2])
                
                results.append({
                    "Model1": model1,
                    "Model2": model2,
                    "AUC1": res["auc1"],
                    "AUC2": res["auc2"],
                    "AUC_Diff": res["diff"],
                    "SE_Diff": res["se_diff"],
                    "Z_Score": res["z"],
                    "P_Value": res["p_value"],
                    "Significant": "Yes" if res["p_value"] < 0.05 else "No"
                })
                
                # Print summary
                sig_marker = "***" if res["p_value"] < 0.001 else "**" if res["p_value"] < 0.01 else "*" if res["p_value"] < 0.05 else ""
                print(f"{model1:20s} vs {model2:20s}: "
                      f"AUC1={res['auc1']:.4f}, AUC2={res['auc2']:.4f}, "
                      f"diff={res['diff']:+.4f}, z={res['z']:+.3f}, "
                      f"p={res['p_value']:.4e} {sig_marker}")
                
            except Exception as e:
                print(f"Error comparing {model1} vs {model2}: {e}")
                continue
    
    # Create DataFrame and save
    df = pd.DataFrame(results)
    
    # Save to CSV
    output_file = Path(__file__).parent.parent / "statistical_tests" / "delong_test_results.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\n✅ DeLong test results saved to: {output_file}")
    
    return df


def perform_mcnemar_tests(y_true, scores, thresholds):
    """Perform pairwise McNemar tests for all model pairs using F1max thresholds."""
    print("\n" + "=" * 100)
    print("MCNEMAR TEST RESULTS (Pairwise Binary Prediction Comparisons at F1max Threshold)")
    print("=" * 100)
    
    model_names = sorted(scores.keys())
    results = []
    
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i >= j:  # Only do upper triangle
                continue
            
            try:
                # Get binary predictions at F1max threshold
                pred1 = (scores[model1] >= thresholds[model1]).astype(int)
                pred2 = (scores[model2] >= thresholds[model2]).astype(int)
                
                res = mcnemar_test(y_true, pred1, pred2, exact=False)
                
                results.append({
                    "Model1": model1,
                    "Model2": model2,
                    "Threshold1": thresholds[model1],
                    "Threshold2": thresholds[model2],
                    "b": res["b"],  # Model1=1, Model2=0
                    "c": res["c"],  # Model1=0, Model2=1
                    "Statistic": res["statistic"] if res["statistic"] is not None else np.nan,
                    "P_Value": res["p_value"],
                    "Method": res["method"],
                    "Significant": "Yes" if res["p_value"] < 0.05 else "No"
                })
                
                # Print summary
                sig_marker = "***" if res["p_value"] < 0.001 else "**" if res["p_value"] < 0.01 else "*" if res["p_value"] < 0.05 else ""
                print(f"{model1:20s} vs {model2:20s}: "
                      f"b={res['b']:4d}, c={res['c']:4d}, "
                      f"stat={res['statistic']:.3f if res['statistic'] is not None else 'N/A':>8s}, "
                      f"p={res['p_value']:.4e} {sig_marker}")
                
            except Exception as e:
                print(f"Error comparing {model1} vs {model2}: {e}")
                continue
    
    # Create DataFrame and save
    df = pd.DataFrame(results)
    
    # Save to CSV
    output_file = Path(__file__).parent.parent / "statistical_tests" / "mcnemar_test_results.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\n✅ McNemar test results saved to: {output_file}")
    
    return df


def create_summary_tables(delong_df, mcnemar_df):
    """Create summary tables showing significant differences."""
    print("\n" + "=" * 100)
    print("SUMMARY: SIGNIFICANT DIFFERENCES (p < 0.05)")
    print("=" * 100)
    
    # DeLong summary
    print("\n📊 DeLong Test - Significant AUC Differences:")
    print("-" * 100)
    sig_delong = delong_df[delong_df["P_Value"] < 0.05].copy()
    if len(sig_delong) > 0:
        sig_delong = sig_delong.sort_values("P_Value")
        print(sig_delong[["Model1", "Model2", "AUC1", "AUC2", "AUC_Diff", "P_Value"]].to_string(index=False))
        print(f"\nTotal significant pairs: {len(sig_delong)} / {len(delong_df)}")
    else:
        print("No significant differences found.")
    
    # McNemar summary
    print("\n📊 McNemar Test - Significant Prediction Differences:")
    print("-" * 100)
    sig_mcnemar = mcnemar_df[mcnemar_df["P_Value"] < 0.05].copy()
    if len(sig_mcnemar) > 0:
        sig_mcnemar = sig_mcnemar.sort_values("P_Value")
        print(sig_mcnemar[["Model1", "Model2", "b", "c", "P_Value"]].to_string(index=False))
        print(f"\nTotal significant pairs: {len(sig_mcnemar)} / {len(mcnemar_df)}")
    else:
        print("No significant differences found.")
    
    # Create pivot tables for easier visualization
    print("\n" + "=" * 100)
    print("PIVOT TABLES (P-Values)")
    print("=" * 100)
    
    # DeLong pivot
    print("\n📊 DeLong Test P-Values Matrix:")
    print("-" * 100)
    delong_pivot = delong_df.pivot(index="Model1", columns="Model2", values="P_Value")
    print(delong_pivot.to_string())
    
    # McNemar pivot
    print("\n📊 McNemar Test P-Values Matrix:")
    print("-" * 100)
    mcnemar_pivot = mcnemar_df.pivot(index="Model1", columns="Model2", values="P_Value")
    print(mcnemar_pivot.to_string())


def main():
    """Main function to run all statistical tests."""
    print("=" * 100)
    print("COMPREHENSIVE STATISTICAL TESTS FOR ALL 12 MODELS")
    print("=" * 100)
    print("\nModels to compare:")
    for i, model_name in enumerate(sorted(MODELS.keys()), 1):
        print(f"  {i:2d}. {model_name}")
    print(f"\nTotal comparisons: {len(MODELS) * (len(MODELS) - 1) // 2} pairs\n")
    
    # Load all models
    y_true, scores, thresholds = load_all_models()
    
    if len(scores) < 2:
        print("Error: Need at least 2 models to perform comparisons.")
        return
    
    # Perform DeLong tests
    delong_df = perform_delong_tests(y_true, scores)
    
    # Perform McNemar tests
    mcnemar_df = perform_mcnemar_tests(y_true, scores, thresholds)
    
    # Create summary tables
    create_summary_tables(delong_df, mcnemar_df)
    
    print("\n" + "=" * 100)
    print("✅ ALL STATISTICAL TESTS COMPLETED")
    print("=" * 100)
    print(f"\nResults saved to:")
    print(f"  - statistical_tests/delong_test_results.csv")
    print(f"  - statistical_tests/mcnemar_test_results.csv")


if __name__ == "__main__":
    main()

