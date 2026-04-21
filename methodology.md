# GUARDIAN ML — ML Methodology

**Version:** 1.0.0  
**Standard:** Springer/Elsevier Technical Report Format

---

## Abstract

GUARDIAN ML implements a multi-model supervised classification and unsupervised anomaly detection framework for geospatial risk assessment. The system ingests structured datasets with optional geographic coordinates, applies a reproducible preprocessing pipeline, trains an ensemble of classifiers optimized for imbalanced risk data, and outputs a composite risk score incorporating spatial and temporal context. All hyperparameters, random seeds, and evaluation protocols are fully configurable and reproducible.

---

## 1. Problem Formulation

Given a dataset **D** = {(**x**ᵢ, yᵢ)} where:
- **x**ᵢ ∈ ℝⁿ is a feature vector (tabular, optionally geospatial)
- yᵢ ∈ {0, 1} is a binary risk label (0 = nominal, 1 = at-risk)

The system learns a risk scoring function **f**: ℝⁿ → [0,1] that assigns a composite risk score to each observation. For unlabeled data, the system falls back to unsupervised anomaly scoring.

---

## 2. Preprocessing Pipeline

### 2.1 Input Validation
- Duplicate row removal
- Constant-column (zero-variance) detection and removal
- Type coercion: numeric-parseable object columns converted to float64
- Missing value flagging: columns with >50% missingness trigger warnings

### 2.2 Geospatial Feature Engineering
If latitude (φ) and longitude (λ) columns are detected:

```
φ_rad = radians(φ)
λ_rad = radians(λ)
d_eq  = √(φ² + λ²)          # Euclidean distance from equator/meridian
Q     = 2·[φ≥0] + [λ≥0]     # Quadrant encoding ∈ {0,1,2,3}
```

### 2.3 Temporal Feature Extraction
For detected datetime columns `t`:
```
t_year, t_month, t_dayofweek, t_hour, t_quarter
```

### 2.4 Categorical Encoding
Ordinal label encoding: categories sorted lexicographically and mapped to {0, 1, …, K-1}. Unknown categories at inference time mapped to -1.

### 2.5 Outlier Handling
**IQR Method (default):**
```
lower = Q1 - 1.5 × IQR
upper = Q3 + 1.5 × IQR
x_clipped = clip(x, lower, upper)
```

**Z-score method (optional):**
```
z = |x - μ| / σ
x[z > threshold] = NaN  → imputed in next step
```

### 2.6 Imputation
`SimpleImputer` with configurable strategy: `median` (default), `mean`, `most_frequent`.

### 2.7 Scaling
| Strategy  | Formula                            | Use Case                       |
|-----------|------------------------------------|--------------------------------|
| Standard  | z = (x - μ) / σ                   | General purpose (default)      |
| MinMax    | z = (x - min) / (max - min)       | Bounded inputs                 |
| Robust    | z = (x - Q2) / (Q3 - Q1)         | Heavy outliers present         |

### 2.8 Train / Validation / Test Split
Stratified split preserving class proportions:
- Train:      70% (default)
- Validation: 10%
- Test:       20%

---

## 3. Model Training

### 3.1 Random Forest
**Algorithm:** Breiman (2001) bagging ensemble of decision trees.

```
n_estimators = 200
max_depth    = 15
min_samples_split = 5
class_weight = "balanced"    # handles class imbalance
n_jobs       = -1            # parallelism
random_state = 42
```

**Feature importance:** Gini impurity decrease averaged over all trees.

### 3.2 XGBoost
**Algorithm:** Chen & Guestrin (2016) gradient boosting with regularization.

```
n_estimators     = 300
max_depth        = 6
learning_rate    = 0.05   (shrinkage)
subsample        = 0.8    (row sampling)
colsample_bytree = 0.8    (column sampling)
eval_metric      = "logloss"
```

### 3.3 Logistic Regression
**Algorithm:** L2-regularized logistic regression.

```
C            = 1.0          (inverse regularization strength)
max_iter     = 1000
class_weight = "balanced"
solver       = "lbfgs"
```

Used as a calibrated probabilistic baseline.

### 3.4 Isolation Forest (Anomaly Detection)
**Algorithm:** Liu et al. (2008) — anomalous points are isolated in fewer splits.

```
n_estimators  = 200
contamination = 0.1    (expected fraction of anomalies)
random_state  = 42
```

Output: `score_samples(X)` → raw anomaly score (lower = more anomalous).

---

## 4. Evaluation Protocol

### 4.1 Metrics
For each supervised model on the validation set:

| Metric        | Formula                                       | Notes                          |
|---------------|-----------------------------------------------|--------------------------------|
| Accuracy      | (TP + TN) / N                                 | Overall correctness            |
| Precision     | TP / (TP + FP)                                | Positive predictive value      |
| Recall        | TP / (TP + FN)                                | Sensitivity / detection rate   |
| F1 (weighted) | 2 × P × R / (P + R)                          | Primary selection metric       |
| ROC-AUC       | ∫ TPR d(FPR)                                 | Discrimination ability         |
| Avg Precision | ∫ P d(R)                                     | Area under PR curve            |

### 4.2 Cross-Validation
Stratified K-Fold (K=5) cross-validation on the training set:

```
for each fold k ∈ {1…5}:
    train on D \ Dₖ
    evaluate on Dₖ
CV_score = mean(fold_scores) ± std(fold_scores)
```

### 4.3 Model Selection
Best model selected by **weighted F1** on the held-out validation set (not the CV set, to prevent optimistic bias).

---

## 5. Composite Risk Scoring

The composite risk score integrates multiple evidence signals:

```
R(x) = w₁·P(y=1|x) + w₂·A(x) + w₃·D(x) + w₄·T(x)
```

Where:

| Component       | Symbol | Default Weight | Description                              |
|-----------------|--------|---------------|------------------------------------------|
| Model prob      | P      | 0.50          | Classifier P(y=1|x)                      |
| Anomaly score   | A      | 0.25          | Normalized IsoForest score               |
| Spatial density | D      | 0.15          | KDE local density index                  |
| Temporal weight | T      | 0.10          | Recency exponential weight               |

Weights are normalized to sum to 1.0. All components mapped to [0, 1].

**Spatial Density (Kernel Density Estimation):**
```
D(xᵢ) = normalize(log KDE(xᵢ))
KDE: Gaussian kernel, bandwidth=0.5° (geographic)
```

**Temporal Recency:**
```
T(xᵢ) = (tᵢ - t_min) / (t_max - t_min)
```

**Risk Classification:**
```
LOW    : R(x) ∈ [0.00, 0.33)
MEDIUM : R(x) ∈ [0.33, 0.66)
HIGH   : R(x) ∈ [0.66, 1.00]
```

---

## 6. Reproducibility

All stochastic operations use a single configurable seed (`random_seed = 42` by default) passed to:
- `train_test_split` (sklearn)
- `StratifiedKFold` (sklearn)
- `RandomForestClassifier` (sklearn)
- `XGBClassifier` (xgboost)
- `IsolationForest` (sklearn)

Model artifacts serialized via `joblib` for byte-exact reproducibility across Python versions.

---

## 7. References

1. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
2. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*.
3. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation Forest. *ICDM 2008*.
4. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
5. Silverman, B. W. (1986). *Density Estimation for Statistics and Data Analysis*. Chapman & Hall.
