import pandas as pd
import numpy as np
import os
import warnings
from scipy.stats import entropy
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import lightgbm as lgb

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
import os

KAGGLE_INPUT = '/kaggle/input/competitions/code-rush-26-ml-module'
LOCAL_INPUT  = os.path.join('data', 'raw')

def find_file(filename):
    """Try Kaggle competition path first, then local fallback."""
    kaggle_path = os.path.join(KAGGLE_INPUT, filename)
    if os.path.exists(kaggle_path):
        return kaggle_path
    local_path = os.path.join(LOCAL_INPUT, filename)
    if os.path.exists(local_path):
        return local_path
    raise FileNotFoundError(
        f"Could not find '{filename}'.\n"
        f"  Tried Kaggle : {kaggle_path}\n"
        f"  Tried local  : {local_path}"
    )

TRAIN_PATH   = find_file('Coderush-26-ML-Train.csv')
TEST_PATH    = find_file('Coderush-26-ML-test.csv')
OUT_DIR      = '/kaggle/working' if os.path.exists('/kaggle/working') else 'submissions/v2_lgbm_v3feats_notebook'
N_FOLDS      = 5
RANDOM_STATE = 42

os.makedirs(OUT_DIR, exist_ok=True)
print(f"Train : {TRAIN_PATH}")
print(f"Test  : {TEST_PATH}")
print(f"Output: {OUT_DIR}")




# =============================================================================
# STEP 1: FEATURE ENGINEERING  (V3 — bag-level aggregation + interactions)
# =============================================================================
def build_features(input_df, has_label=True):
    """
    Transforms raw instance-level rows into bag-level aggregated features.
    Returns a bag-level DataFrame.
    """
    df = input_df.copy()

    # --- Individual-level derivations ---
    df['age']           = df['survey_year'] - df['year_of_birth']
    df['is_married']    = (df['marital_status'] == 'Married-civ-spouse').astype(int)
    df['is_high_edu']   = (df['education_num'] >= 13).astype(int)   # Bachelors+
    df['is_professional'] = df['occupation'].isin(
        ['Prof-specialty', 'Exec-managerial']).astype(int)
    df['has_capital']   = (df['capital_gain'] > 0).astype(int)
    df['works_long']    = (df['hours_per_week'] > 40).astype(int)
    df['is_private']    = (df['workclass'] == 'Private').astype(int)

    # --- Aggregation dictionary ---
    agg = {}

    # Core numerics
    for col in ['age', 'education_num', 'hours_per_week', 'capital_gain', 'capital_loss']:
        agg[col] = ['mean', 'std', 'median']

    # Binary indicator proportions
    for col in ['is_married', 'is_high_edu', 'is_professional',
                'has_capital', 'works_long', 'is_private']:
        agg[col] = ['mean']

    # Bag size
    agg['bag_id'] = ['count']

    # Label (train only)
    if has_label:
        label_map = {'lower': 0, 'middle': 1, 'upper': 2}
        df['label_enc'] = df['label'].map(label_map)
        agg['label_enc'] = ['first']

    # Run aggregation
    df_agg = df.groupby('bag_id').agg(agg)
    df_agg.columns = ['_'.join(c).strip() for c in df_agg.columns]

    renames = {'bag_id_count': 'bag_size'}
    if has_label:
        renames['label_enc_first'] = 'label'
    df_agg.rename(columns=renames, inplace=True)

    # --- Categorical proportions ---
    # Education tier
    for tier in ['Primary', 'Secondary', 'Higher']:
        df_agg[f'edu_tier_{tier}_pct'] = df.groupby('bag_id')['education_tier'].apply(
            lambda x, t=tier: (x == t).mean())

    # Marital status
    for val in ['Married-civ-spouse', 'Never-married', 'Divorced', 'Widowed']:
        safe = val.replace('-', '_').replace(' ', '_')
        df_agg[f'marital_{safe}_pct'] = df.groupby('bag_id')['marital_status'].apply(
            lambda x, v=val: (x == v).mean())

    # Occupation
    for val in ['Prof-specialty', 'Exec-managerial', 'Craft-repair', 'Sales', 'Adm-clerical']:
        safe = val.replace('-', '_').replace(' ', '_')
        df_agg[f'occ_{safe}_pct'] = df.groupby('bag_id')['occupation'].apply(
            lambda x, v=val: (x == v).mean())

    # Relationship
    for val in ['Husband', 'Not-in-family', 'Own-child', 'Wife']:
        safe = val.replace('-', '_').replace(' ', '_')
        df_agg[f'rel_{safe}_pct'] = df.groupby('bag_id')['relationship'].apply(
            lambda x, v=val: (x == v).mean())

    # Diversity entropy
    for col in ['education', 'occupation', 'marital_status', 'workclass']:
        df_agg[f'{col}_entropy'] = df.groupby('bag_id')[col].apply(
            lambda x: entropy(x.value_counts(normalize=True)))

    # Demographics
    df_agg['sex_Male_pct']  = df.groupby('bag_id')['sex'].apply(lambda x: (x == 'Male').mean())
    df_agg['race_White_pct']= df.groupby('bag_id')['race'].apply(lambda x: (x == 'White').mean())
    df_agg['race_Black_pct']= df.groupby('bag_id')['race'].apply(lambda x: (x == 'Black').mean())

    # IQR for core numerics
    for col in ['age', 'education_num', 'hours_per_week']:
        q25 = df.groupby('bag_id')[col].quantile(0.25)
        q75 = df.groupby('bag_id')[col].quantile(0.75)
        df_agg[f'{col}_iqr'] = q75 - q25

    # --- Interaction features (V3 additions) ---
    df_agg['edu_x_hours']               = df_agg['education_num_mean'] * df_agg['hours_per_week_mean']
    df_agg['edu_x_age']                 = df_agg['education_num_mean'] / (df_agg['age_mean'] + 1)
    df_agg['married_x_highedu']         = df_agg['is_married_mean'] * df_agg['is_high_edu_mean']
    df_agg['professional_x_highedu']    = df_agg['is_professional_mean'] * df_agg['is_high_edu_mean']
    df_agg['capital_x_hours']           = df_agg['capital_gain_mean'] * df_agg['hours_per_week_mean']
    df_agg['higher_edu_x_professional'] = df_agg['edu_tier_Higher_pct'] * df_agg['is_professional_mean']
    df_agg['married_x_works_long']      = df_agg['is_married_mean'] * df_agg['works_long_mean']

    df_agg.fillna(0, inplace=True)
    df_agg = df_agg.reset_index()
    return df_agg

# =============================================================================
# STEP 2: LOAD & ENGINEER FEATURES
# =============================================================================
print("Loading data...")
train_raw = pd.read_csv(TRAIN_PATH)
test_raw  = pd.read_csv(TEST_PATH)

print("Building training features...")
train_feats = build_features(train_raw, has_label=True)

print("Building test features...")
test_feats  = build_features(test_raw, has_label=False)

feature_cols = [c for c in train_feats.columns if c not in ['bag_id', 'label']]
X      = train_feats[feature_cols]
y      = train_feats['label'].astype(int)
bag_ids_test = test_feats['bag_id']
X_test = test_feats[feature_cols]

print(f"Train shape: {X.shape} | Test shape: {X_test.shape}")
print(f"Class distribution: {dict(pd.Series(y).value_counts().sort_index())}")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# =============================================================================
# STEP 3A: BASELINE MODEL (Reference)
# Simple Random Forest on basic aggregated features
# =============================================================================
print("\n" + "="*50)
print("--- 1. BASELINE MODEL (Random Forest) ---")
print("="*50)
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=RANDOM_STATE)
oof_preds_base = np.zeros(len(X), dtype=int)
train_f1s_base, val_f1s_base = [], []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    
    rf_model.fit(X_tr, y_tr)
    oof_preds_base[va_idx] = rf_model.predict(X_va)
    
    tr_f1 = f1_score(y_tr, rf_model.predict(X_tr), average='macro')
    va_f1 = f1_score(y_va, oof_preds_base[va_idx], average='macro')
    train_f1s_base.append(tr_f1)
    val_f1s_base.append(va_f1)

base_oof_f1 = f1_score(y, oof_preds_base, average='macro')
base_gap = np.mean(train_f1s_base) - np.mean(val_f1s_base)
print(f"Baseline OOF Macro F1 : {base_oof_f1:.4f}")
print(f"Baseline Train-Val Gap: {base_gap:.4f}")

# =============================================================================
# STEP 3B: IMPROVED MODEL (Iterative Upgrade)
# Standard LightGBM on basic aggregated features
# =============================================================================
print("\n" + "="*50)
print("--- 2. IMPROVED MODEL (Standard LightGBM) ---")
print("="*50)

lgbm_standard = lgb.LGBMClassifier(
    n_estimators=500, learning_rate=0.05, 
    class_weight='balanced', random_state=RANDOM_STATE, verbose=-1, n_jobs=-1
)
oof_preds_imp = np.zeros(len(X), dtype=int)
train_f1s_imp, val_f1s_imp = [], []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    
    lgbm_standard.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_preds_imp[va_idx] = lgbm_standard.predict(X_va)
    
    tr_f1 = f1_score(y_tr, lgbm_standard.predict(X_tr), average='macro')
    va_f1 = f1_score(y_va, oof_preds_imp[va_idx], average='macro')
    train_f1s_imp.append(tr_f1)
    val_f1s_imp.append(va_f1)

imp_oof_f1 = f1_score(y, oof_preds_imp, average='macro')
imp_gap = np.mean(train_f1s_imp) - np.mean(val_f1s_imp)
print(f"Improved OOF Macro F1 : {imp_oof_f1:.4f}")
print(f"Improved Train-Val Gap: {imp_gap:.4f}")

# =============================================================================
# STEP 3C: BEST MODEL (Ultra-Regularized LightGBM - Final Submission)
# =============================================================================
print("\n" + "="*50)
print("--- 3. BEST MODEL (Ultra-Regularized LightGBM) ---")
print("="*50)

lgbm_params = {
    'n_estimators':    2000,
    'learning_rate':   0.03,
    'num_leaves':      15,
    'max_depth':       4,
    'min_data_in_leaf': 100,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.7,
    'bagging_freq':    5,
    'lambda_l1':       2.0,
    'lambda_l2':       2.0,
    'class_weight':    'balanced',
    'random_state':    RANDOM_STATE,
    'verbose':         -1,
    'n_jobs':          -1,
}

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
oof_preds      = np.zeros(len(X), dtype=int)
oof_probs      = np.zeros((len(X), 3))
test_probs     = np.zeros((len(X_test), 3))
train_f1s, val_f1s = [], []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    model = lgb.LGBMClassifier(**lgbm_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False),
                   lgb.log_evaluation(period=-1)]
    )

    oof_preds[va_idx]  = model.predict(X_va)
    oof_probs[va_idx]  = model.predict_proba(X_va)
    test_probs        += model.predict_proba(X_test) / N_FOLDS

    tr_f1 = f1_score(y_tr, model.predict(X_tr), average='macro')
    va_f1 = f1_score(y_va, oof_preds[va_idx],   average='macro')
    train_f1s.append(tr_f1)
    val_f1s.append(va_f1)
    print(f"  Fold {fold+1}: Train F1={tr_f1:.4f}  Val F1={va_f1:.4f}  Gap={tr_f1-va_f1:.4f}")

oof_macro_f1 = f1_score(y, oof_preds, average='macro')
avg_val_f1   = np.mean(val_f1s)
avg_gap      = np.mean(train_f1s) - avg_val_f1

print(f"\nOOF Macro F1 : {oof_macro_f1:.4f}")
print(f"Avg Val F1   : {avg_val_f1:.4f}")
print(f"Train-Val Gap: {avg_gap:.4f}")

print("\nPer-Class Report (BEST MODEL):")
print(classification_report(y, oof_preds, target_names=['lower', 'middle', 'upper']))

print("\n" + "="*50)
print("--- EVALUATION SUMMARY (Presentation Table) ---")
print("="*50)
print(f"{'Model':<35} | {'OOF Macro F1':<15} | {'Train-Val Gap':<15}")
print("-" * 70)
print(f"{'1. Baseline (Random Forest)':<35} | {base_oof_f1:<15.4f} | {base_gap:<15.4f}")
print(f"{'2. Improved (Standard LGBM)':<35} | {imp_oof_f1:<15.4f} | {imp_gap:<15.4f}")
print(f"{'3. Best (Ultra-Reg LGBM)':<35} | {oof_macro_f1:<15.4f} | {avg_gap:<15.4f}")
print("="*50)

# =============================================================================
# STEP 4: GENERATE SUBMISSION  (labels as integers: 0=lower, 1=middle, 2=upper)
# =============================================================================

final_preds = np.argmax(test_probs, axis=1)

submission = pd.DataFrame({
    'bag_id': bag_ids_test,
    'label':  final_preds
}).sort_values('bag_id').reset_index(drop=True)

out_path = os.path.join(OUT_DIR, 'submission.csv')
submission.to_csv(out_path, index=False)

print(f"\nSubmission saved to: {out_path}")
print(f"Shape            : {submission.shape}")
print(f"Label dist       : {dict(submission['label'].value_counts().sort_index())}")
print(f"  0=lower  1=middle  2=upper")
print(f"Sample:\n{submission.head()}")