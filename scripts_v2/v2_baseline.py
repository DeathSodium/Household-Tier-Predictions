"""
V2 Phase 1: Baseline Model — Logistic Regression (Classical)
Simplicity Tier: HIGHEST
"""
import pandas as pd
import numpy as np
import os, joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def run_baseline(train_path, test_path, sub_dir, model_dir):
    print("=" * 60)
    print("PHASE 1: BASELINE — Logistic Regression")
    print("=" * 60)

    df = pd.read_csv(train_path)
    X = df.drop(['bag_id', 'label'], axis=1)
    y = df['label'].astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(df))
    train_scores, val_scores = [], []

    scaler = StandardScaler()

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        # Scale
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=42)
        model.fit(X_tr_s, y_tr)

        tr_pred = model.predict(X_tr_s)
        va_pred = model.predict(X_va_s)
        oof_preds[va_idx] = va_pred

        tr_f1 = f1_score(y_tr, tr_pred, average='macro')
        va_f1 = f1_score(y_va, va_pred, average='macro')
        train_scores.append(tr_f1)
        val_scores.append(va_f1)
        print(f"  Fold {fold+1}: Train={tr_f1:.4f}  Val={va_f1:.4f}  Gap={tr_f1-va_f1:.4f}")

    avg_tr = np.mean(train_scores)
    avg_va = np.mean(val_scores)
    print(f"\n  AVG Train: {avg_tr:.4f}  AVG OOF Val: {avg_va:.4f}  AVG Gap: {avg_tr-avg_va:.4f}")
    print(f"\n{classification_report(y, oof_preds, target_names=['lower','middle','upper'])}")

    # Train final model on all data for submission
    X_all_s = scaler.fit_transform(X)
    final_model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=42)
    final_model.fit(X_all_s, y)

    # Predict test
    df_test = pd.read_csv(test_path)
    bag_ids = df_test['bag_id']
    X_test = df_test.drop(['bag_id'], axis=1)
    # Align columns
    for col in X.columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[X.columns]
    X_test_s = scaler.transform(X_test)

    preds = final_model.predict(X_test_s)

    sub = pd.DataFrame({'bag_id': bag_ids, 'label': preds})
    os.makedirs(sub_dir, exist_ok=True)
    sub.to_csv(os.path.join(sub_dir, 'submission.csv'), index=False)

    # Save model info
    info = f"""# V2 Baseline: Logistic Regression

## Simplicity Tier: Classical (HIGHEST)

## Performance
* **Avg Train Macro F1**: {avg_tr:.4f}
* **Avg OOF Val Macro F1**: {avg_va:.4f}
* **Train-Val Gap**: {avg_tr-avg_va:.4f}

## Per-Fold Results
| Fold | Train F1 | Val F1 | Gap |
|:---|:---:|:---:|:---:|
{chr(10).join(f'| Fold {i+1} | {train_scores[i]:.4f} | {val_scores[i]:.4f} | {train_scores[i]-val_scores[i]:.4f} |' for i in range(5))}

## Parameters
* C=1.0, solver=lbfgs, max_iter=1000
* StandardScaler applied

## Features
* **Count**: {X.shape[1]}
* **Source**: v2_feature_engineering.py (clean 47-feature set)
"""
    with open(os.path.join(sub_dir, 'MODEL_INFO.md'), 'w') as f:
        f.write(info)
    print(f"\nSubmission saved to {sub_dir}")

if __name__ == '__main__':
    run_baseline(
        'data/processed/train_v2.csv',
        'data/processed/test_v2.csv',
        'submissions/v2_baseline_lr',
        'models/v2'
    )
