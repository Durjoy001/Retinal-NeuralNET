import math
import numpy as np

# =========================
#  Helpers: normal CDF/SF
# =========================

def _norm_sf(z):
    """Survival function 1 - CDF for standard normal, no scipy needed."""
    # sf(z) = 0.5 * erfc(z / sqrt(2))
    return 0.5 * math.erfc(z / math.sqrt(2.0))


# =========================
#  DeLong AUC comparison
# =========================

def _compute_theta(pos_scores, neg_scores):
    """
    Pairwise comparison matrix theta_ij:
      = 1 if pos_i > neg_j
      = 0.5 if pos_i == neg_j
      = 0 otherwise
    Returns:
      theta: shape (n_pos, n_neg)
    """
    pos = pos_scores[:, None]      # (n_pos, 1)
    neg = neg_scores[None, :]      # (1, n_neg)
    greater = (pos > neg).astype(float)
    equal = (pos == neg).astype(float)
    theta = greater + 0.5 * equal
    return theta


def delong_two_sample_auc(y_true, scores1, scores2):
    """
    DeLong test for two correlated ROC AUCs (same y_true, two models).

    Args:
        y_true  : array-like, shape (N,), 0/1 labels
        scores1 : array-like, shape (N,), continuous scores for model 1
        scores2 : array-like, shape (N,), continuous scores for model 2

    Returns:
        auc1, auc2, var1, var2, cov12
        (AUCs and variance/covariance estimates)
    """
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

    # Theta matrices
    theta1 = _compute_theta(pos1, neg1)  # (n_pos, n_neg)
    theta2 = _compute_theta(pos2, neg2)

    # AUC estimates
    auc1 = np.mean(theta1)
    auc2 = np.mean(theta2)

    # V10: average over negatives, per positive
    V10_1 = np.mean(theta1, axis=1)
    V10_2 = np.mean(theta2, axis=1)

    # V01: average over positives, per negative
    V01_1 = np.mean(theta1, axis=0)
    V01_2 = np.mean(theta2, axis=0)

    # Sample variances / covariances
    def _sample_var(x):
        # ddof=1 for sample variance; guard for n=1
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
    """
    Perform DeLong test for the null hypothesis: AUC1 == AUC2.

    Returns:
        dict with:
            auc1, auc2, diff, se_diff, z, p_value
    """
    auc1, auc2, var1, var2, cov12 = delong_two_sample_auc(y_true, scores1, scores2)
    diff = auc1 - auc2
    var_diff = var1 + var2 - 2.0 * cov12

    if var_diff <= 0:
        # numeric edge case; treat as no difference
        se_diff = 0.0
        z = 0.0
        p_value = 1.0
    else:
        se_diff = math.sqrt(var_diff)
        z = diff / se_diff
        # two-sided p-value
        p_value = 2.0 * _norm_sf(abs(z))

    return {
        "auc1": auc1,
        "auc2": auc2,
        "diff": diff,
        "se_diff": se_diff,
        "z": z,
        "p_value": p_value,
    }


# =========================
#  McNemar test
# =========================

def mcnemar_test(y_true, pred1, pred2, exact=False):
    """
    McNemar test for paired binary predictions of two models.

    Args:
        y_true : array-like, shape (N,), not strictly needed for McNemar
                (it uses disagreement table between pred1 and pred2),
                but can be useful for constructing confusion logic.
        pred1  : array-like, shape (N,), binary predictions (0/1) from model 1
        pred2  : array-like, shape (N,), binary predictions (0/1) from model 2
        exact  : if True, use exact binomial test for b+c <= 25,
                 otherwise use chi-square with continuity correction.

    Returns:
        dict with:
            b, c, statistic, p_value, method
        where:
            b = # cases pred1=1, pred2=0
            c = # cases pred1=0, pred2=1
    """
    y_true = np.asarray(y_true)
    p1 = np.asarray(pred1)
    p2 = np.asarray(pred2)

    if p1.shape != p2.shape:
        raise ValueError("pred1 and pred2 must have the same shape")

    # We care about discordant pairs (b, c)
    b = int(np.sum((p1 == 1) & (p2 == 0)))
    c = int(np.sum((p1 == 0) & (p2 == 1)))
    n = b + c

    if n == 0:
        # models are identical
        return {
            "b": b,
            "c": c,
            "statistic": 0.0,
            "p_value": 1.0,
            "method": "no discordant pairs (identical predictions)",
        }

    # Exact binomial test for small samples
    if exact and n <= 25:
        # two-sided exact p-value under Binomial(n, 0.5)
        # P = 2 * min(P(X <= min(b,c)), P(X >= max(b,c)))
        k = min(b, c)
        from math import comb

        # Compute tail probabilities
        # P(X <= k)
        p_le_k = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
        # P(X >= n-k)
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

    # Chi-square with continuity correction
    # statistic = (|b - c| - 1)^2 / (b + c)
    diff = abs(b - c)
    statistic = (diff - 1.0) ** 2 / float(n)

    # Chi-square(1) p-value using normal approximation of sqrt(chi2)
    # or we approximate using normal: Z^2 ~ chi2_1, so P(chi2 >= stat) = 2*sf(sqrt(stat))
    # For 1 d.f., chi-square and squared normal are equivalent.
    z = math.sqrt(statistic)
    p_value = 2.0 * _norm_sf(z)

    return {
        "b": b,
        "c": c,
        "statistic": statistic,
        "p_value": p_value,
        "method": "chi-square McNemar with continuity correction",
    }


# =========================
#  Example usage
# =========================

def load_any_abnormal_npz(path):
    """
    Helper to load test outputs from an .npz file of one model.

    Expected keys:
        y_true : (N,) 0/1 any-abnormal labels
        y_score: (N,) continuous scores (probabilities) for any-abnormal

    You can adapt this if your npz has different names.
    """
    data = np.load(path)
    y_true = data["y_true"]
    y_score = data["y_score"]
    return y_true, y_score


def example_comparisons():
    """
    Example: Compare SwinTiny vs ResNet50 and SwinTiny vs DenseNet121
    using DeLong (AUC) and McNemar (fixed threshold).
    Adjust file paths and thresholds to your own outputs.
    """
    # ---- 1. Define file paths for your saved outputs ----
    # You must adapt these to your actual results structure.
    files = {
        "ResNet50":      "results/CNN/ResNet50/anyabnormal_test_outputs.npz",
        "DenseNet121":   "results/CNN/DenseNet121/anyabnormal_test_outputs.npz",
        "SwinTiny":      "results/ViT/SwinTiny/anyabnormal_test_outputs.npz",
    }

    # Load all models
    y_true_shared = None
    scores = {}
    for name, path in files.items():
        y_true, y_score = load_any_abnormal_npz(path)
        scores[name] = y_score
        if y_true_shared is None:
            y_true_shared = y_true
        else:
            # Sanity check: ensure labels identical for all models
            if not np.array_equal(y_true_shared, y_true):
                raise ValueError(f"y_true mismatch for model {name}")

    # ---- 2. DeLong test: AUC comparison ----
    ref_model = "SwinTiny"
    print("=== DeLong AUC comparison vs SwinTiny ===")
    for model in files.keys():
        if model == ref_model:
            continue
        res = delong_roc_test(y_true_shared, scores[ref_model], scores[model])
        print(f"{ref_model} vs {model}:")
        print(f"  AUC_{ref_model}: {res['auc1']:.4f}")
        print(f"  AUC_{model}:   {res['auc2']:.4f}")
        print(f"  diff: {res['diff']:.4f}, z={res['z']:.3f}, p={res['p_value']:.4g}\n")

    # ---- 3. McNemar test: sensitivity at fixed specificity ----
    # Here we need binary predictions at a chosen threshold.
    # Example: use your F1max thresholds from the table (for demo).
    thresholds = {
        "ResNet50":    0.5758,   # your original F1max threshold (or spec80)
        "DenseNet121": 0.6775,
        "SwinTiny":    0.8958,
    }

    print("=== McNemar test at model-specific thresholds (e.g., F1max) ===")
    ref_model = "SwinTiny"
    y_true = y_true_shared

    pred_ref = (scores[ref_model] >= thresholds[ref_model]).astype(int)

    for model in files.keys():
        if model == ref_model:
            continue
        pred_other = (scores[model] >= thresholds[model]).astype(int)
        mc = mcnemar_test(y_true, pred_ref, pred_other, exact=False)
        print(f"{ref_model} vs {model}:")
        print(f"  b (ref=1, other=0): {mc['b']}, c (ref=0, other=1): {mc['c']}")
        print(f"  statistic: {mc['statistic']}, p={mc['p_value']:.4g}")
        print(f"  method: {mc['method']}\n")


if __name__ == "__main__":
    # Remove or comment this out when you integrate into your codebase.
    # This is just an example driver.
    example_comparisons()
