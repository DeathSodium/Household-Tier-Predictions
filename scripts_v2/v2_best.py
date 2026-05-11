"""
V2 Phase 3: Best Model
Option A: Heavily regularized LightGBM (5-fold avg)
Option B: Soft-voting LightGBM + RandomForest
Picks whichever has higher OOF F1.
"""
import pandas as pd
import numpy as np
import os, joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

def run_best(train_path, test_path, sub_dir, model_dir):
    print("=" * 60)
    print("PHASE 3: BEST MODEL — Comparing Options")
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

    # =================== Option A: Ultra-regularized LightGBM ===================
    print("\n--- Option A: Ultra-Regularized LightGBM ---")
    oof_a = np.zeros(len(df))
    test_probs_a = np.zeros((len(df_test), 3))
    tr_a, va_a = [], []

    params_a = {
        'n_estimators': 2000,
        'learning_rate': 0.03,
        'num_leaves': 15,
        'max_depth': 4,
        'min_data_in_leaf': 100,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'lambda_l1': 2.0,
        'lambda_l2': 2.0,
        'random_state': 42,
        'verbose': -1,
    }

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        m = lgb.LGBMClassifier(**params_a)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(stopping_rounds=100)])
        oof_a[va_idx] = m.predict(X_va)
        test_probs_a += m.predict_proba(X_test) / 5
        tr_f1 = f1_score(y_tr, m.predict(X_tr), average='macro')
        va_f1 = f1_score(y_va, oof_a[va_idx], average='macro')
        tr_a.append(tr_f1); va_a.append(va_f1)
        print(f"  Fold {fold+1}: Train={tr_f1:.4f}  Val={va_f1:.4f}  Gap={tr_f1-va_f1:.4f}")
    avg_a = np.mean(va_a)
    gap_a = np.mean(tr_a) - avg_a
    print(f"  Option A OOF: {avg_a:.4f}  Gap: {gap_a:.4f}")

    # =================== Option B: LightGBM + RandomForest Soft Vote ===================
    print("\n--- Option B: LightGBM + RandomForest Soft Vote ---")
    oof_b = np.zeros(len(df))
    test_probs_b = np.zeros((len(df_test), 3))
    tr_b, va_b = [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        m_lgb = lgb.LGBMClassifier(**params_a)
        m_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(stopping_rounds=100)])

        m_rf = RandomForestClassifier(n_estimators=300, max_depth=8,
                                      min_samples_leaf=30, random_state=42)
        m_rf.fit(X_tr, y_tr)

        # Average probabilities
        va_prob = (m_lgb.predict_proba(X_va) + m_rf.predict_proba(X_va)) / 2
        va_pred = np.argmax(va_prob, axis=1)
        oof_b[va_idx] = va_pred

        test_probs_b += (m_lgb.predict_proba(X_test) + m_rf.predict_proba(X_test)) / 2 / 5

        tr_prob = (m_lgb.predict_proba(X_tr) + m_rf.predict_proba(X_tr)) / 2
        tr_pred = np.argmax(tr_prob, axis=1)
        tr_f1 = f1_score(y_tr, tr_pred, average='macro')
        va_f1 = f1_score(y_va, va_pred, average='macro')
        tr_b.append(tr_f1); va_b.append(va_f1)
        print(f"  Fold {fold+1}: Train={tr_f1:.4f}  Val={va_f1:.4f}  Gap={tr_f1-va_f1:.4f}")
    avg_b = np.mean(va_b)
    gap_b = np.mean(tr_b) - avg_b
    print(f"  Option B OOF: {avg_b:.4f}  Gap: {gap_b:.4f}")

    # =================== Pick Winner ===================
    print("\n" + "=" * 60)
    if avg_a >= avg_b:
        winner = "A (Ultra-Reg LightGBM)"
        final_preds = np.argmax(test_probs_a, axis=1)
        best_oof = avg_a
        best_gap = gap_a
        best_params = params_a
        oof_final = oof_a
    else:
        winner = "B (LightGBM + RandomForest)"
        final_preds = np.argmax(test_probs_b, axis=1)
        best_oof = avg_b
        best_gap = gap_b
        best_params = params_a  # lgbm params
        oof_final = oof_b
    print(f"WINNER: Option {winner} | OOF F1: {best_oof:.4f} | Gap: {best_gap:.4f}")
    print(f"\n{classification_report(y, oof_final, target_names=['lower','middle','upper'])}")

    # Save submission
    sub = pd.DataFrame({'bag_id': bag_ids, 'label': final_preds})
    os.makedirs(sub_dir, exist_ok=True)
    sub.to_csv(os.path.join(sub_dir, 'submission.csv'), index=False)

    info = f"""# V2 Best Model: {winner}

## Simplicity Tier: Ensemble/Tree-based

## Performance
* **Avg OOF Val Macro F1**: {best_oof:.4f}
* **Train-Val Gap**: {best_gap:.4f}

## Winning Option
* **{winner}**
* Option A (Ultra-Reg LGBM) OOF: {avg_a:.4f}, Gap: {gap_a:.4f}
* Option B (LGBM+RF Vote) OOF: {avg_b:.4f}, Gap: {gap_b:.4f}

## Features
* **Count**: {len(features)}
* **Source**: v2_feature_engineering.py
"""
    with open(os.path.join(sub_dir, 'MODEL_INFO.md'), 'w', encoding='utf-8') as f:
        f.write(info)
    print(f"\nSubmission saved to {sub_dir}")

if __name__ == '__main__':
    run_best(
        'data/processed/train_v2.csv',
        'data/processed/test_v2.csv',
        'submissions/v2_best',
        'models/v2'
    )
