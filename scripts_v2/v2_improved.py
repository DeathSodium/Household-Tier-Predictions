"""
V2 Phase 2: Improved Model — Single Regularized LightGBM
Simplicity Tier: Tree-based (MEDIUM)
Uses 5-fold averaging at inference for stability.
"""
import pandas as pd
import numpy as np
import os, joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import lightgbm as lgb

def run_improved(train_path, test_path, sub_dir, model_dir):
    print("=" * 60)
    print("PHASE 2: IMPROVED — Single Regularized LightGBM")
    print("=" * 60)

    df = pd.read_csv(train_path)
    X = df.drop(['bag_id', 'label'], axis=1)
    y = df['label'].astype(int)
    features = X.columns.tolist()

    df_test = pd.read_csv(test_path)
    bag_ids = df_test['bag_id']
    X_test = df_test.drop(['bag_id'], axis=1)
    for col in features:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[features]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(df))
    test_probs = np.zeros((len(df_test), 3))
    train_scores, val_scores = [], []
    models = []

    params = {
        'n_estimators': 2000,
        'learning_rate': 0.03,
        'num_leaves': 20,
        'max_depth': 6,
        'min_data_in_leaf': 50,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.5,
        'lambda_l2': 0.5,
        'random_state': 42,
        'verbose': -1,
    }

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(stopping_rounds=100)])

        tr_pred = model.predict(X_tr)
        va_pred = model.predict(X_va)
        oof_preds[va_idx] = va_pred

        # Accumulate test probabilities for averaging
        test_probs += model.predict_proba(X_test) / 5

        tr_f1 = f1_score(y_tr, tr_pred, average='macro')
        va_f1 = f1_score(y_va, va_pred, average='macro')
        train_scores.append(tr_f1)
        val_scores.append(va_f1)
        models.append(model)
        print(f"  Fold {fold+1}: Train={tr_f1:.4f}  Val={va_f1:.4f}  Gap={tr_f1-va_f1:.4f}")

    avg_tr = np.mean(train_scores)
    avg_va = np.mean(val_scores)
    print(f"\n  AVG Train: {avg_tr:.4f}  AVG OOF Val: {avg_va:.4f}  AVG Gap: {avg_tr-avg_va:.4f}")
    print(f"\n{classification_report(y, oof_preds, target_names=['lower','middle','upper'])}")

    # Generate submission from averaged probabilities
    final_preds = np.argmax(test_probs, axis=1)

    sub = pd.DataFrame({'bag_id': bag_ids, 'label': final_preds})
    os.makedirs(sub_dir, exist_ok=True)
    sub.to_csv(os.path.join(sub_dir, 'submission.csv'), index=False)

    # Save models
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump({'models': models, 'features': features, 'params': params},
                os.path.join(model_dir, 'v2_improved_lgbm.joblib'))

    # Save model info
    info = f"""# V2 Improved: Single Regularized LightGBM (5-Fold Averaged)

## Simplicity Tier: Tree-based (MEDIUM)

## Performance
* **Avg Train Macro F1**: {avg_tr:.4f}
* **Avg OOF Val Macro F1**: {avg_va:.4f}
* **Train-Val Gap**: {avg_tr-avg_va:.4f}

## Per-Fold Results
| Fold | Train F1 | Val F1 | Gap |
|:---|:---:|:---:|:---:|
{chr(10).join(f'| Fold {i+1} | {train_scores[i]:.4f} | {val_scores[i]:.4f} | {train_scores[i]-val_scores[i]:.4f} |' for i in range(5))}

## Parameters
```python
{params}
```

## Inference Strategy
* Average predicted probabilities from 5 fold models, then argmax
* This reduces variance and improves generalization

## Features
* **Count**: {len(features)}
* **Source**: v2_feature_engineering.py (clean 47-feature set)
"""
    with open(os.path.join(sub_dir, 'MODEL_INFO.md'), 'w', encoding='utf-8') as f:
        f.write(info)
    print(f"\nSubmission saved to {sub_dir}")

if __name__ == '__main__':
    run_improved(
        'data/processed/train_v2.csv',
        'data/processed/test_v2.csv',
        'submissions/v2_improved_lgbm',
        'models/v2'
    )
