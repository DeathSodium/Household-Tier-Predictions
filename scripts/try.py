import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import f1_score
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

print("1. Loading and Preprocessing Data...")
train = pd.read_csv('data/raw/Coderush-26-ML-Train.csv')
test = pd.read_csv('data/raw/Coderush-26-ML-test.csv')

# Map labels
label_map = {'lower': 0, 'middle': 1, 'upper': 2}
train['label'] = train['label'].map(label_map)

# Combine for uniform processing
train['is_train'] = 1
test['is_train'] = 0
test['label'] = -1 
df = pd.concat([train, test], ignore_index=True)

# =====================================================================
# ADVANCED INSTANCE-LEVEL FEATURE ENGINEERING
# =====================================================================
# 1. Base Derivations
df['age'] = 1994 - df['year_of_birth']
df['net_capital'] = df['capital_gain'] - df['capital_loss']

# 2. Financial & Effort Cross-Features (Crucial for Tree Splits)
df['wealth_velocity'] = df['net_capital_asset'] / (df['age'] + 1)
df['financial_stress'] = df['poverty_line_usd'] / (df['hours_per_week'] + 1)
df['effort_yield'] = df['education_num'] * df['hours_per_week']
df['is_capital_active'] = (df['capital_gain'] > 0) | (df['capital_loss'] > 0)
df['is_capital_active'] = df['is_capital_active'].astype(int)

# 3. Native Pandas Encoding (Zero Dependency on external libraries)
cols_to_drop = [
    'interview_mode', 'currency_code', 'year_of_birth',
    'annual_hours_est', # 1.0 correlated with hours_per_week
    'net_capital_asset', # 1.0 correlated with capital_gain
    'survey_year', # zero variance
    'processing_flag', # zero variance
    'poverty_line_usd' # zero variance
]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# Ordinal mappings
df['education_tier'] = df['education_tier'].map({'Primary': 0, 'Secondary': 1, 'Higher': 2})
edu_map = {
    'Preschool': 0, '1st-4th': 1, '5th-6th': 2, '7th-8th': 3, '9th': 4, '10th': 5, 
    '11th': 6, '12th': 7, 'HS-grad': 8, 'Some-college': 9, 'Assoc-voc': 10, 
    'Assoc-acdm': 11, 'Bachelors': 12, 'Masters': 13, 'Prof-school': 14, 'Doctorate': 15
}
df['education'] = df['education'].map(edu_map)
df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})

# Frequency Encoding for high cardinality to avoid explosion of columns
for col in ['native_country', 'occupation']:
    freq = df[col].value_counts() / len(df)
    df[f'{col}_freq'] = df[col].map(freq)
    df = df.drop(col, axis=1)

# Standard One-Hot for the rest
df = pd.get_dummies(df, columns=['relationship', 'race', 'workclass', 'marital_status'], drop_first=True)
for col in df.columns:
    if df[col].dtype == 'bool':
        df[col] = df[col].astype(int)

# Split back to Train / Test
X_train_full = df[df['is_train'] == 1].drop('is_train', axis=1).copy()
X_test_full = df[df['is_train'] == 0].drop(['is_train', 'label'], axis=1).copy()

# =====================================================================
# STAGE 1: INSTANCE-LEVEL WEAK LEARNER (Level 0)
# =====================================================================
print("\n2. STAGE 1: Training Instance Level Model (Weak Learners)...")
features = [c for c in X_train_full.columns if c not in ['label', 'bag_id']]
X = X_train_full[features]
y = X_train_full['label']
groups = X_train_full['bag_id']

# GroupKFold ensures no individuals from the same bag leak across train/val
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof_instance_probs = np.zeros((len(X_train_full), 3))
test_instance_probs = np.zeros((len(X_test_full), 3))

for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    # Powerful, deep tree to extract maximum signal from individuals
    model_l0 = lgb.LGBMClassifier(
        n_estimators=1500, learning_rate=0.03, max_depth=7, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, class_weight='balanced', random_state=42, n_jobs=-1
    )
    model_l0.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
    
    oof_instance_probs[val_idx] = model_l0.predict_proba(X_va)
    test_instance_probs += model_l0.predict_proba(X_test_full[features]) / 5

# Attach weak probabilities to original dataframes
for i in range(3):
    X_train_full[f'prob_{i}'] = oof_instance_probs[:, i]
    X_test_full[f'prob_{i}'] = test_instance_probs[:, i]

# =====================================================================
# STAGE 2: META-MODEL AGGREGATION & TRAINING (Level 1)
# =====================================================================
print("\n3. STAGE 2: Aggregating Topology & Training Meta-Model...")

def q25(x): return x.quantile(0.25)
def q75(x): return x.quantile(0.75)
def skew(x): return x.skew() if len(x) > 2 else 0

def bag_skew(x):
    return x.skew() if len(x)>2 else 0

# The Ultimate Statistical Mapping Dictionary
agg_funcs = {
    'prob_0': ['mean', 'max', 'min', 'std', q25, q75, bag_skew],
    'prob_1': ['mean', 'max', 'min', 'std', q25, q75, bag_skew],
    'prob_2': ['mean', 'max', 'min', 'std', q25, q75, bag_skew],
    'wealth_velocity': ['max', 'mean'],
    'education_num': ['max', 'mean'],
    'effort_yield': ['max', 'mean'],
    'age': ['min', 'max', 'mean']
}

# Apply aggregations to Train
train_meta = X_train_full.groupby('bag_id').agg({**agg_funcs, 'label': ['first']})
train_meta.columns = [f"{col[0]}_{col[1]}" if col[1] != 'first' else col[0] for col in train_meta.columns]
train_meta = train_meta.rename(columns={'label_<lambda_0>': 'label', 'label_first': 'label'}) 

X_meta = train_meta.drop('label', axis=1).fillna(0)
y_meta = train_meta['label'].astype(int)

# Apply aggregations to Test
test_meta = X_test_full.groupby('bag_id').agg(agg_funcs)
test_meta.columns = [f"{col[0]}_{col[1]}" for col in test_meta.columns]
test_meta = test_meta.fillna(0)

# Train the Meta Model
skf_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_meta_probs = np.zeros((len(X_meta), 3))
final_test_preds_prob = np.zeros((len(test_meta), 3))

for fold, (train_idx, val_idx) in enumerate(skf_meta.split(X_meta, y_meta)):
    X_tr, y_tr = X_meta.iloc[train_idx], y_meta.iloc[train_idx]
    X_va, y_va = X_meta.iloc[val_idx], y_meta.iloc[val_idx]
    
    # Meta model is heavily regularized to prevent overfitting on the synthetic probabilities
    model_l1 = lgb.LGBMClassifier(
        n_estimators=800, learning_rate=0.015, max_depth=4, num_leaves=15,
        class_weight='balanced', reg_lambda=3.0, reg_alpha=1.0, random_state=42, n_jobs=-1
    )
    model_l1.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(75, verbose=False)])
    
    oof_meta_probs[val_idx] = model_l1.predict_proba(X_va)
    final_test_preds_prob += model_l1.predict_proba(test_meta) / 5

# =====================================================================
# THRESHOLD OPTIMIZATION & FINAL OUTPUT
# =====================================================================
print("\n4. Optimizing Target Thresholds...")

def optimize_thresholds(y_true, y_probs):
    def f1_opt(weights):
        pred = np.argmax(y_probs * weights, axis=1)
        return -f1_score(y_true, pred, average='macro')
    # Use Nelder-Mead to solve the threshold optimization
    res = minimize(f1_opt, [1.0, 1.0, 1.0], method='Nelder-Mead')
    return res.x

best_weights = optimize_thresholds(y_meta, oof_meta_probs)
opt_preds_classes = np.argmax(oof_meta_probs * best_weights, axis=1)
final_f1 = f1_score(y_meta, opt_preds_classes, average='macro')

print("\n" + "="*50)
print(f"ULTIMATE STACKING MACRO F1 SCORE: {final_f1:.5f}")
print(f"Optimal Threshold Modifiers Applied: {np.round(best_weights, 3)}")
print("="*50)

# Generate final submission with exact same weights
final_predictions = np.argmax(final_test_preds_prob * best_weights, axis=1)

submission = pd.DataFrame({
    'bag_id': test_meta.index,
    'label': final_predictions
}).sort_values('bag_id').reset_index(drop=True)

submission.to_csv('submission_v4_ultimate.csv', index=False)
print("Saved final competitive predictions to 'submission_v4_ultimate.csv'")