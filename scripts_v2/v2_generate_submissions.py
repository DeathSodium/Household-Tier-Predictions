"""
Generate submissions for Decision Tree and Random Forest models.
Also generate a 5-fold averaged RF submission (best generalizer with small gap).
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

def generate_submission(X_train, y_train, X_test, bag_ids, model_fn, sub_dir, model_name, is_lgbm=False):
    print(f"\n--- {model_name} ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y_train))
    tr_scores, va_scores = [], []

    if is_lgbm:
        test_probs = np.zeros((len(X_test), 3))
    else:
        test_probs = np.zeros((len(X_test), 3))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

        m = model_fn()
        if is_lgbm:
            m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        else:
            m.fit(X_tr, y_tr)

        oof[va_idx] = m.predict(X_va)
        test_probs += m.predict_proba(X_test) / 5
        tr_scores.append(f1_score(y_tr, m.predict(X_tr), average='macro'))
        va_scores.append(f1_score(y_va, oof[va_idx], average='macro'))

    avg_tr = np.mean(tr_scores)
    avg_va = np.mean(va_scores)
    gap = avg_tr - avg_va
    print(f"  Train: {avg_tr:.4f} | OOF: {avg_va:.4f} | Gap: {gap:.4f}")
    print(classification_report(y_train, oof, target_names=['lower','middle','upper']))

    preds = np.argmax(test_probs, axis=1)
    sub = pd.DataFrame({'bag_id': bag_ids, 'label': preds})
    os.makedirs(sub_dir, exist_ok=True)
    sub.to_csv(os.path.join(sub_dir, 'submission.csv'), index=False)

    info = f"""# {model_name}

## Performance
* **Avg Train Macro F1**: {avg_tr:.4f}
* **Avg OOF Val Macro F1**: {avg_va:.4f}
* **Train-Val Gap**: {gap:.4f}

## Per-Fold Results
| Fold | Train F1 | Val F1 | Gap |
|:---|:---:|:---:|:---:|
{chr(10).join(f'| Fold {i+1} | {tr_scores[i]:.4f} | {va_scores[i]:.4f} | {tr_scores[i]-va_scores[i]:.4f} |' for i in range(5))}

## Features: {X_train.shape[1]} (v2 clean set)
"""
    with open(os.path.join(sub_dir, 'MODEL_INFO.md'), 'w', encoding='utf-8') as f:
        f.write(info)
    print(f"  Saved to {sub_dir}")

df = pd.read_csv('data/processed/train_v2.csv')
X = df.drop(['bag_id', 'label'], axis=1)
y = df['label'].astype(int)
features = X.columns.tolist()

df_test = pd.read_csv('data/processed/test_v2.csv')
bag_ids = df_test['bag_id']
X_test = df_test.drop(['bag_id'], axis=1)
for col in features:
    if col not in X_test.columns:
        X_test[col] = 0
X_test = X_test[features]

# 1. Decision Tree (depth=5) — simplest possible
generate_submission(X, y, X_test, bag_ids,
    lambda: DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=42),
    'submissions/v2_decision_tree', 'Decision Tree (depth=5, leaf=20)')

# 2. Random Forest (best balanced config)
generate_submission(X, y, X_test, bag_ids,
    lambda: RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_leaf=5, random_state=42),
    'submissions/v2_random_forest', 'Random Forest (500 trees, unlimited depth)')

# 3. Random Forest (conservative, smallest gap)
generate_submission(X, y, X_test, bag_ids,
    lambda: RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=20, random_state=42),
    'submissions/v2_rf_conservative', 'Random Forest Conservative (200, depth=8, leaf=20)')

# 4. LGBM best config (5-fold averaged) — v3 features
df3 = pd.read_csv('data/processed/train_v3.csv')
X3 = df3.drop(['bag_id', 'label'], axis=1)
y3 = df3['label'].astype(int)
feats3 = X3.columns.tolist()

df_t3 = pd.read_csv('data/processed/test_v3.csv')
bag_ids3 = df_t3['bag_id']
X_t3 = df_t3.drop(['bag_id'], axis=1)
for col in feats3:
    if col not in X_t3.columns:
        X_t3[col] = 0
X_t3 = X_t3[feats3]

generate_submission(X3, y3, X_t3, bag_ids3,
    lambda: lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=31, max_depth=6,
        min_data_in_leaf=50, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5,
        lambda_l1=0.5, lambda_l2=0.5, random_state=42, verbose=-1),
    'submissions/v2_lgbm_v3feats', 'LGBM v3 Features (60 feats, 5-fold avg)', is_lgbm=True)
