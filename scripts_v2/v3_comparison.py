"""
V3 Feature Test: Run top models on v3 (60 features with interactions)
vs v2 (47 features) to see if interactions help.
"""
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')

def oof_eval(X, y, model_fn, name):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    tr_s, va_s = [], []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        m = model_fn()
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        if isinstance(m, lgb.LGBMClassifier):
            m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        else:
            m.fit(X_tr, y_tr)
        oof[va_idx] = m.predict(X_va)
        tr_s.append(f1_score(y_tr, m.predict(X_tr), average='macro'))
        va_s.append(f1_score(y_va, oof[va_idx], average='macro'))
    return np.mean(tr_s), np.mean(va_s), np.mean(tr_s)-np.mean(va_s)

print("=" * 70)
print("V3 vs V2 FEATURE COMPARISON (Top Models)")
print("=" * 70)

for feat_name, path in [("V2 (47 feats)", "data/processed/train_v2.csv"),
                         ("V3 (60 feats)", "data/processed/train_v3.csv")]:
    df = pd.read_csv(path)
    X = df.drop(['bag_id', 'label'], axis=1)
    y = df['label'].astype(int)
    print(f"\n--- {feat_name} ---")

    models = [
        ("RF(500, None, leaf=5)", lambda: RandomForestClassifier(500, max_depth=None, min_samples_leaf=5, random_state=42)),
        ("ExtraTrees(300, None)", lambda: ExtraTreesClassifier(300, max_depth=None, min_samples_leaf=10, random_state=42)),
        ("XGB(depth=6, lr=0.05)", lambda: xgb.XGBClassifier(500, max_depth=6, learning_rate=0.05,
            min_child_weight=30, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=0.5,
            random_state=42, verbosity=0, eval_metric='mlogloss')),
        ("LGBM(leaves=31, depth=6)", lambda: lgb.LGBMClassifier(
            n_estimators=2000, learning_rate=0.03, num_leaves=31, max_depth=6,
            min_data_in_leaf=50, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5,
            lambda_l1=0.5, lambda_l2=0.5, random_state=42, verbose=-1)),
    ]
    for name, fn in models:
        tr, va, gap = oof_eval(X, y, fn, name)
        print(f"  {name:35s} | Train: {tr:.4f} | OOF: {va:.4f} | Gap: {gap:.4f}")
