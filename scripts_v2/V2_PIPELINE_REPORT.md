# Comprehensive Model Sweep & Feature Engineering Report

## Executive Summary

We ran **21 model configurations** across **2 feature sets** to find the best combination of performance and generalization. The core finding: **there is a fundamental ceiling around 0.69 OOF Macro F1** with the current bag-level aggregation approach, and all high-OOF models suffer from significant overfitting.

---

## Complete Model Sweep Results (47 v2 Features)

Models marked with `<<<` have a train-val gap below 0.15 (good generalization).

| Rank | Model | Train F1 | OOF F1 | Gap | Tier |
|:---:|:---|:---:|:---:|:---:|:---|
| 1 | LGBM (leaves=31, depth=6) | 0.9953 | **0.6914** | 0.304 | Tree-based |
| 2 | LGBM (leaves=20, depth=5) | 0.9852 | **0.6856** | 0.300 | Tree-based |
| 3 | RF (500, unlimited depth) | 0.9258 | **0.6665** | 0.259 | Tree-based |
| 4 | XGB (depth=6, lr=0.05) | 0.9510 | 0.6633 | 0.288 | Tree-based |
| 5 | LGBM (leaves=15, depth=4) | 0.9242 | 0.6595 | 0.265 | Tree-based |
| 6 | XGB (depth=4, lr=0.03) | 0.8575 | 0.6428 | 0.215 | Tree-based |
| 7 | GradientBoosting (depth=4) | 0.8627 | 0.6427 | 0.220 | Tree-based |
| 8 | ExtraTrees (300, unlimited) | 0.7716 | 0.6406 | 0.131 <<< | Ensemble |
| 9 | LGBM (leaves=10, depth=3) | 0.7817 | 0.6380 | 0.144 <<< | Tree-based |
| 10 | RF (500, depth=12) | 0.8038 | 0.6370 | 0.167 | Tree-based |
| 11 | ExtraTrees (300, depth=12) | 0.7544 | 0.6363 | 0.118 <<< | Ensemble |
| 12 | RF (300, depth=10) | 0.7433 | 0.6276 | 0.116 <<< | Tree-based |
| 13 | ExtraTrees (300, depth=8) | 0.6978 | 0.6239 | 0.074 <<< | Ensemble |
| 14 | RF (200, depth=8) | 0.7054 | 0.6221 | 0.083 <<< | Tree-based |
| 15 | RF (100, depth=6) | 0.6623 | 0.6074 | 0.055 <<< | Tree-based |
| 16 | **Logistic Regression** | 0.6179 | **0.6027** | **0.015** <<< | **Classical** |
| 17 | Decision Tree (depth=5) | 0.6262 | 0.5739 | 0.052 <<< | Classical |
| 18 | Decision Tree (depth=7) | 0.6683 | 0.5674 | 0.101 <<< | Classical |
| 19 | Decision Tree (unlimited) | 0.6997 | 0.5560 | 0.144 <<< | Classical |
| 20 | Decision Tree (depth=3) | 0.5665 | 0.5545 | 0.012 <<< | Classical |
| 21 | Decision Tree (depth=10) | 0.6952 | 0.5543 | 0.141 <<< | Classical |

---

## V3 Feature Set (60 Features with Interactions) vs V2 (47 Features)

| Model | V2 OOF | V3 OOF | Improvement |
|:---|:---:|:---:|:---:|
| RF (500, unlimited) | 0.6665 | 0.6670 | +0.0005 |
| ExtraTrees (300, unlimited) | 0.6406 | 0.6363 | -0.0043 |
| XGB (depth=6) | 0.6359 | 0.6361 | +0.0002 |
| LGBM (leaves=31, depth=6) | 0.6914 | 0.6928 | +0.0014 |

**Conclusion**: V3 interaction features provide **negligible improvement** (<0.002). The v2 47-feature set is sufficient.

---

## Key Findings

### 1. The Overfitting Wall
Every model that reaches OOF > 0.65 has a train-val gap > 0.15. This is a fundamental property of the dataset:
- Only **3,360 bags** with **3-7 members each**
- Gradient boosting models (LGBM, XGB) always memorize at 0.99+ training F1
- Even Random Forest reaches 0.92 training F1 at its best OOF config

### 2. Best Models by Category

| Category | Best Model | OOF F1 | Gap | Submission |
|:---|:---|:---:|:---:|:---|
| **Classical** | Logistic Regression | 0.6027 | 0.015 | `v2_baseline_lr/` |
| **Decision Tree** | DT (depth=5) | 0.5739 | 0.052 | `v2_decision_tree/` |
| **Random Forest** | RF (500, unlimited) | 0.6665 | 0.259 | `v2_random_forest/` |
| **RF Conservative** | RF (200, depth=8) | 0.6221 | 0.083 | `v2_rf_conservative/` |
| **LightGBM** | LGBM (31, depth=6) | 0.6914 | 0.304 | `v2_improved_lgbm/` |
| **LGBM + v3 Features** | LGBM v3 (60 feats) | 0.6928 | 0.303 | `v2_lgbm_v3feats/` |

### 3. Achieving 0.70+ OOF

Our LGBM already hits 0.69 OOF. To push past 0.70:
- Adding interaction features (v3) gave only +0.001
- Higher model complexity always increases the gap
- The ceiling appears to be a **data limitation**, not a modeling limitation
- With 3,360 bags of 3-7 people, the statistical signal in bag-level aggregation is inherently noisy

### 4. Recommended Submissions (Priority Order)

1. **`v2_lgbm_v3feats/`** — Highest OOF (0.69), 5-fold averaged, v3 features
2. **`v2_improved_lgbm/`** — Same model on v2 features (simpler, comparable performance)
3. **`v2_rf_conservative/`** — Low gap (0.08), should generalize well to Kaggle
4. **`v2_baseline_lr/`** — If it scores within 0.02 of others on Kaggle, use this (Simplicity Rule wins)

---

## All Generated Submissions

| Folder | Model | OOF F1 | Gap |
|:---|:---|:---:|:---:|
| `v2_baseline_lr/` | Logistic Regression | 0.6022 | 0.016 |
| `v2_decision_tree/` | Decision Tree (depth=5) | 0.5739 | 0.052 |
| `v2_random_forest/` | RF (500, unlimited) | 0.6665 | 0.259 |
| `v2_rf_conservative/` | RF (200, depth=8) | 0.6221 | 0.083 |
| `v2_improved_lgbm/` | LGBM (v2, 47 feats) | 0.6931 | 0.303 |
| `v2_lgbm_v3feats/` | LGBM (v3, 60 feats) | 0.6928 | 0.303 |
| `v2_best/` | Ultra-Reg LGBM | 0.6587 | 0.266 |
