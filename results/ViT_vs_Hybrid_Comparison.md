# ViT vs Hybrid Models Comparison Results

## 1. Overall Test Results Comparison (Per-Class Metrics)

| Model | Architecture | Test AUC | Test Micro AUC | Test Loss | Balanced Acc | Sensitivity | Specificity | Macro F1 | Micro F1 | Precision | Recall | F1max | Best Val AUC |
|-------|--------------|----------|----------------|-----------|--------------|-------------|-------------|----------|----------|-----------|--------|-------|--------------|
| **SwinTiny** | ViT | **0.9109** | 0.9702 | 1.593 | **0.8909** | **0.8902** | 0.8915 | 0.305 | **0.5094** | 0.3568 | **0.8902** | **0.7429** | **0.9214** |
| **CrossViTSmall** | ViT | **0.9106** | 0.9631 | 1.355 | 0.8677 | 0.8655 | 0.8699 | 0.276 | 0.4568 | 0.3103 | 0.8655 | 0.6760 | 0.8988 |
| **DeiTSmall** | ViT | 0.9056 | 0.9636 | 1.499 | 0.8713 | 0.8638 | **0.8789** | 0.302 | 0.4725 | 0.3252 | 0.8638 | 0.6932 | 0.8805 |
| **ViTSmall** | ViT | 0.8944 | 0.9504 | **1.214** | 0.8602 | 0.8409 | 0.8795 | 0.294 | 0.4641 | 0.3205 | 0.8409 | 0.6366 | 0.8894 |
| **MaxViTTiny** | Hybrid | 0.8997 | 0.9505 | **0.9868** | 0.8744 | 0.8698 | 0.8790 | 0.303 | 0.4753 | 0.3270 | 0.8698 | 0.6337 | 0.8978 |
| **CoAtNet0** | Hybrid | 0.8974 | 0.9505 | 1.034 | 0.8755 | 0.8570 | **0.8940** | **0.3395** | 0.5004 | **0.3533** | 0.8570 | 0.6349 | 0.8993 |

## 2. Any-Abnormal Detection Results Comparison

| Model | Architecture | AUC (%) | Precision@0.80spec | Recall@0.80spec | F1max | TP | TN | FP | FN |
|-------|--------------|---------|-------------------|----------------|-------|----|----|----|----|
| **SwinTiny** | ViT | **97.76** | 97.55 | **94.47** | **0.9642** | 478 | 122 | 12 | 28 |
| **MaxViTTiny** | Hybrid | 96.63 | **97.31** | 92.89 | 0.9531 | 470 | 121 | 13 | 36 |
| **CoAtNet0** | Hybrid | 96.60 | **98.27** | 89.72 | 0.9471 | 454 | 126 | 8 | 52 |
| **DeiTSmall** | ViT | 96.59 | **98.28** | 90.12 | 0.9465 | 456 | 126 | 8 | 50 |
| **CrossViTSmall** | ViT | 95.22 | 97.16 | 87.94 | 0.9369 | 445 | 121 | 13 | 61 |
| **ViTSmall** | ViT | 95.17 | 97.55 | 86.56 | 0.9374 | 438 | 123 | 11 | 68 |

## 3. Key Findings

### 🏆 Top Performing Models

#### Per-Class Task:
- **SwinTiny (ViT)**: Best AUC (0.9109), Best Balanced Accuracy (0.8909), Best Sensitivity (0.8902), Best Micro F1 (0.5094), Best F1max (0.7429)
- **CoAtNet0 (Hybrid)**: Best Specificity (0.8940), Best Macro F1 (0.3395), Best Precision (0.3533), Lowest Loss (among Hybrid models)

#### Any-Abnormal Task:
- **SwinTiny (ViT)**: Best AUC (97.76%), Best Recall (94.47%), Best F1max (0.9642)
- **CoAtNet0 (Hybrid)**: Best Precision (98.27%), Fewest False Positives (FP=8)

### 📊 Architecture Type Comparison

#### ViT Models:
- ✅ **Strengths**: SwinTiny performs best across multiple metrics, especially for any-abnormal detection
- ✅ **Characteristics**: Better AUC and balanced accuracy
- ⚠️ **Weaknesses**: Generally higher training loss (1.2-1.6)

#### Hybrid Models:
- ✅ **Strengths**: Lower training loss (0.99-1.03), CoAtNet0 has better specificity
- ✅ **Characteristics**: Higher precision in any-abnormal detection task
- ⚠️ **Weaknesses**: AUC slightly lower than best ViT models

### 💡 Key Metrics Analysis

1. **Test AUC Ranking**: SwinTiny (0.9109) > CrossViTSmall (0.9106) > DeiTSmall (0.9056) > MaxViTTiny (0.8997) > CoAtNet0 (0.8974) > ViTSmall (0.8944)

2. **Any-Abnormal AUC Ranking**: SwinTiny (97.76%) > MaxViTTiny (96.63%) > CoAtNet0 (96.60%) > DeiTSmall (96.59%) > CrossViTSmall (95.22%) > ViTSmall (95.17%)

3. **Training Efficiency** (lower loss is better): MaxViTTiny (loss=0.9868) > ViTSmall (loss=1.214) > CoAtNet0 (loss=1.034) > CrossViTSmall (loss=1.355) > DeiTSmall (loss=1.499) > SwinTiny (loss=1.593)

## 4. Summary

### ViT Models:
- **SwinTiny** is the top-performing model, excelling in both overall classification and any-abnormal detection tasks
- ViT architectures generally have stronger AUC and sensitivity

### Hybrid Models:
- **MaxViTTiny** and **CoAtNet0** provide lower training loss and good precision
- Hybrid architectures have advantages in specificity and precision
- For any-abnormal detection, MaxViTTiny performs close to the best ViT model

### Recommendations:
- For **best overall performance**: Choose **SwinTiny (ViT)**
- For **balanced performance and efficiency**: Choose **MaxViTTiny (Hybrid)** or **CoAtNet0 (Hybrid)**
- For **higher precision** (fewer false positives): Choose **CoAtNet0 (Hybrid)**
