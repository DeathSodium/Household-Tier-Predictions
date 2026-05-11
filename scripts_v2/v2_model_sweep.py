"""
V2 Model Sweep: Decision Tree, Random Forest, ExtraTrees, XGBoost, LightGBM
All on the same clean 47-feature set for fair comparison.
Also includes a v3 feature set with interaction features.
"""
import pandas as pd
import numpy as np
import os, warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')

def oof_evaluate(X, y, model_fn, model_name, n_splits=5):
    """Run 5-fold OOF and return scores."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    tr_scores, va_scores = [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = model_fn()
        model.fit(X_tr, y_tr)

        tr_pred = model.predict(X_tr)
        va_pred = model.predict(X_va)
        oof[va_idx] = va_pred

        tr_scores.append(f1_score(y_tr, tr_pred, average='macro'))
        va_scores.append(f1_score(y_va, va_pred, average='macro'))

    avg_tr = np.mean(tr_scores)
    avg_va = np.mean(va_scores)
    gap = avg_tr - avg_va
    return avg_tr, avg_va, gap, oof

def oof_evaluate_lgbm(X, y, params, model_name, n_splits=5):
    """Special handler for LightGBM with early stopping."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    tr_scores, va_scores = [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])

        tr_pred = model.predict(X_tr)
        va_pred = model.predict(X_va)
        oof[va_idx] = va_pred

        tr_scores.append(f1_score(y_tr, tr_pred, average='macro'))
        va_scores.append(f1_score(y_va, va_pred, average='macro'))

    avg_tr = np.mean(tr_scores)
    avg_va = np.mean(va_scores)
    gap = avg_tr - avg_va
    return avg_tr, avg_va, gap, oof


def run_sweep():
    print("=" * 70)
    print("COMPREHENSIVE MODEL SWEEP")
    print("=" * 70)

    df = pd.read_csv('data/processed/train_v2.csv')
    X = df.drop(['bag_id', 'label'], axis=1)
    y = df['label'].astype(int)

    results = []

    # ===== 1. Decision Tree (various depths) =====
    for depth in [3, 5, 7, 10, None]:
        name = f"DecisionTree(depth={depth})"
        tr, va, gap, _ = oof_evaluate(X, y,
            lambda d=depth: DecisionTreeClassifier(max_depth=d, min_samples_leaf=20, random_state=42),
            name)
        results.append((name, tr, va, gap))
        print(f"  {name:45s} | Train: {tr:.4f} | OOF: {va:.4f} | Gap: {gap:.4f}")

    # ===== 2. Random Forest (various configs) =====
    rf_configs = [
        ("RF(100, depth=6, leaf=30)", dict(n_estimators=100, max_depth=6, min_samples_leaf=30)),
        ("RF(200, depth=8, leaf=20)", dict(n_estimators=200, max_depth=8, min_samples_leaf=20)),
        ("RF(300, depth=10, leaf=15)", dict(n_estimators=300, max_depth=10, min_samples_leaf=15)),
        ("RF(500, depth=12, leaf=10)", dict(n_estimators=500, max_depth=12, min_samples_leaf=10)),
        ("RF(500, depth=None, leaf=5)", dict(n_estimators=500, max_depth=None, min_samples_leaf=5)),
    ]
    for name, cfg in rf_configs:
        tr, va, gap, _ = oof_evaluate(X, y,
            lambda c=cfg: RandomForestClassifier(**c, random_state=42),
            name)
        results.append((name, tr, va, gap))
        print(f"  {name:45s} | Train: {tr:.4f} | OOF: {va:.4f} | Gap: {gap:.4f}")

    # ===== 3. ExtraTrees =====
    for depth in [8, 12, None]:
        name = f"ExtraTrees(300, depth={depth})"
        tr, va, gap, _ = oof_evaluate(X, y,
            lambda d=depth: ExtraTreesClassifier(n_estimators=300, max_depth=d, min_samples_leaf=10, random_state=42),
            name)
        results.append((name, tr, va, gap))
        print(f"  {name:45s} | Train: {tr:.4f} | OOF: {va:.4f} | Gap: {gap:.4f}")

    # ===== 4. Gradient Boosting (sklearn) =====
    name = "GradientBoosting(200, depth=4, lr=0.05)"
    tr, va, gap, _ = oof_evaluate(X, y,
        lambda: GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                           min_samples_leaf=30, random_state=42),
        name)
    results.append((name, tr, va, gap))
    print(f"  {name:45s} | Train: {tr:.4f} | OOF: {va:.4f} | Gap: {gap:.4f}")

    # ===== 5. XGBoost =====
    xgb_configs = [
        ("XGB(depth=4, lr=0.03, leaf=50)", dict(n_estimators=1000, max_depth=4, learning_rate=0.03,
            min_child_weight=50, subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=1.0,
            random_state=42, verbosity=0, eval_metric='mlogloss')),
        ("XGB(depth=6, lr=0.05, leaf=30)", dict(n_estimators=500, max_depth=6, learning_rate=0.05,
            min_child_weight=30, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=0.5,
            random_state=42, verbosity=0, eval_metric='mlogloss')),
    ]
    for name, cfg in xgb_configs:
        tr, va, gap, _ = oof_evaluate(X, y,
            lambda c=cfg: xgb.XGBClassifier(**c),
            name)
        results.append((name, tr, va, gap))
        print(f"  {name:45s} | Train: {tr:.4f} | OOF: {va:.4f} | Gap: {gap:.4f}")

    # ===== 6. LightGBM (various regularization levels) =====
    lgbm_configs = [
        ("LGBM(leaves=10, depth=3, leaf=100, L1=3)", dict(
            n_estimators=2000, learning_rate=0.03, num_leaves=10, max_depth=3,
            min_data_in_leaf=100, feature_fraction=0.6, bagging_fraction=0.7, bagging_freq=5,
            lambda_l1=3.0, lambda_l2=3.0, random_state=42, verbose=-1)),
        ("LGBM(leaves=15, depth=4, leaf=80, L1=2)", dict(
            n_estimators=2000, learning_rate=0.03, num_leaves=15, max_depth=4,
            min_data_in_leaf=80, feature_fraction=0.6, bagging_fraction=0.7, bagging_freq=5,
            lambda_l1=2.0, lambda_l2=2.0, random_state=42, verbose=-1)),
        ("LGBM(leaves=20, depth=5, leaf=60, L1=1)", dict(
            n_estimators=2000, learning_rate=0.03, num_leaves=20, max_depth=5,
            min_data_in_leaf=60, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5,
            lambda_l1=1.0, lambda_l2=1.0, random_state=42, verbose=-1)),
        ("LGBM(leaves=31, depth=6, leaf=50, L1=0.5)", dict(
            n_estimators=2000, learning_rate=0.03, num_leaves=31, max_depth=6,
            min_data_in_leaf=50, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5,
            lambda_l1=0.5, lambda_l2=0.5, random_state=42, verbose=-1)),
    ]
    for name, cfg in lgbm_configs:
        tr, va, gap, _ = oof_evaluate_lgbm(X, y, cfg, name)
        results.append((name, tr, va, gap))
        print(f"  {name:45s} | Train: {tr:.4f} | OOF: {va:.4f} | Gap: {gap:.4f}")

    # ===== 7. Logistic Regression (reference) =====
    name = "LogisticRegression(C=1.0)"
    # Need to scale for LR
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    tr, va, gap, _ = oof_evaluate(X_scaled, y,
        lambda: LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=42),
        name)
    results.append((name, tr, va, gap))
    print(f"  {name:45s} | Train: {tr:.4f} | OOF: {va:.4f} | Gap: {gap:.4f}")

    # ===== SUMMARY =====
    print("\n" + "=" * 70)
    print("RESULTS SORTED BY OOF F1")
    print("=" * 70)
    results.sort(key=lambda x: -x[2])  # Sort by OOF descending
    print(f"{'Model':45s} | {'Train':>6s} | {'OOF':>6s} | {'Gap':>6s}")
    print("-" * 70)
    for name, tr, va, gap in results:
        flag = " <<<" if gap < 0.15 else ""
        print(f"{name:45s} | {tr:.4f} | {va:.4f} | {gap:.4f}{flag}")

    # Save results
    df_res = pd.DataFrame(results, columns=['Model', 'Train_F1', 'OOF_F1', 'Gap'])
    df_res.to_csv('scripts_v2/model_sweep_results.csv', index=False)
    print("\nResults saved to scripts_v2/model_sweep_results.csv")

if __name__ == '__main__':
    run_sweep()
