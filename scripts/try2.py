import pandas as pd
import numpy as np
import category_encoders as ce
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import f1_score
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

print("Loading Data...")
train = pd.read_csv('data/raw/Coderush-26-ML-Train.csv')
test = pd.read_csv('data/raw/Coderush-26-ML-test.csv')

label_map = {'lower': 0, 'middle': 1, 'upper': 2}
train['label'] = train['label'].map(label_map)

train['is_train'] = 1
test['is_train'] = 0
test['label'] = -1 
df = pd.concat([train, test], ignore_index=True)

# =====================================================================
# UPGRADE 1: INSTANCE-LEVEL INTERACTION FEATURES
# =====================================================================
# Survey was in 1994. Let's calculate precise age and financial metrics.
df['age'] = 1994 - df['year_of_birth']

# Protect against divide-by-zero
df['wealth_accumulation_rate'] = df['net_capital_asset'] / (df['age'] + 1)
df['hours_per_education'] = df['hours_per_week'] * df['education_num']
df['capital_magnitude'] = df['capital_gain'] + df['capital_loss']

# Drop redundant and zero-variance features
cols_to_drop = [
    'interview_mode', 'currency_code', 'year_of_birth', 
    'annual_hours_est', # 1.0 correlated with hours_per_week
    'net_capital_asset', # 1.0 correlated with capital_gain
    'survey_year', # zero variance
    'processing_flag', # zero variance
    'poverty_line_usd' # zero variance
]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

df['education_tier'] = df['education_tier'].map({'Primary': 0, 'Secondary': 1, 'Higher': 2})
edu_map = {
    'Preschool': 0, '1st-4th': 1, '5th-6th': 2, '7th-8th': 3, '9th': 4, '10th': 5, 
    '11th': 6, '12th': 7, 'HS-grad': 8, 'Some-college': 9, 'Assoc-voc': 10, 
    'Assoc-acdm': 11, 'Bachelors': 12, 'Masters': 13, 'Prof-school': 14, 'Doctorate': 15
}
df['education'] = df['education'].map(edu_map)
df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})

bin_encoder = ce.BinaryEncoder(cols=['native_country', 'occupation'])
df = bin_encoder.fit_transform(df)

df = pd.get_dummies(df, columns=['relationship', 'race', 'workclass', 'marital_status'], drop_first=True)
for col in df.columns:
    if df[col].dtype == 'bool':
        df[col] = df[col].astype(int)

X_train_full = df[df['is_train'] == 1].drop('is_train', axis=1)
X_test_full = df[df['is_train'] == 0].drop(['is_train', 'label'], axis=1)

# =====================================================================
# STAGE 1: INSTANCE-LEVEL WEAK LEARNERS
# =====================================================================
print("--- STAGE 1: Generating Instance Probabilities ---")
features = [c for c in X_train_full.columns if c not in ['label', 'bag_id']]
X = X_train_full[features]
y = X_train_full['label']
groups = X_train_full['bag_id']

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof_instance_probs = np.zeros((len(X_train_full), 3))
test_instance_probs = np.zeros((len(X_test_full), 3))

for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    # Highly tuned hyperparameters for Level 0
    model_l0 = lgb.LGBMClassifier(
        n_estimators=1200, learning_rate=0.03, max_depth=6, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, class_weight='balanced', random_state=42
    )
    model_l0.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
    
    oof_instance_probs[val_idx] = model_l0.predict_proba(X_va)
    test_instance_probs += model_l0.predict_proba(X_test_full[features]) / 5

X_train_full['prob_0'] = oof_instance_probs[:, 0]
X_train_full['prob_1'] = oof_instance_probs[:, 1]
X_train_full['prob_2'] = oof_instance_probs[:, 2]

X_test_full['prob_0'] = test_instance_probs[:, 0]
X_test_full['prob_1'] = test_instance_probs[:, 1]
X_test_full['prob_2'] = test_instance_probs[:, 2]

# =====================================================================
# UPGRADE 2: BAG-LEVEL DISTRIBUTIONAL MAPPING
# =====================================================================
print("--- STAGE 2: Advanced Bag-Level Aggregation ---")

def bag_skew(x):
    return x.skew() if len(x)>2 else 0

# We now capture the Skewness and Standard Deviation of the probabilities
agg_funcs = {
    'prob_0': ['mean', 'max', 'min', 'std', bag_skew],
    'prob_1': ['mean', 'max', 'min', 'std', bag_skew],
    'prob_2': ['mean', 'max', 'min', 'std', bag_skew],
    # Also pass original strong numeric indicators up to the meta-model
    'capital_gain': ['max', 'mean'],
    'education_num': ['max', 'mean'],
    'wealth_accumulation_rate': ['max', 'mean']
}

train_meta = X_train_full.groupby('bag_id').agg({**agg_funcs, 'label': ['first']})
train_meta.columns = [f"{col[0]}_{col[1]}" if col[1] != 'first' else col[0] for col in train_meta.columns]
# Fill missing std and skew values
train_meta = train_meta.fillna(0)

X_meta = train_meta.drop('label', axis=1)
y_meta = train_meta['label'].astype(int)

test_meta = X_test_full.groupby('bag_id').agg(agg_funcs)
test_meta.columns = [f"{col[0]}_{col[1]}" for col in test_meta.columns]
test_meta = test_meta.fillna(0)

skf_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_meta_probs = np.zeros((len(X_meta), 3))
final_test_preds_prob = np.zeros((len(test_meta), 3))

for fold, (train_idx, val_idx) in enumerate(skf_meta.split(X_meta, y_meta)):
    X_tr, y_tr = X_meta.iloc[train_idx], y_meta.iloc[train_idx]
    X_va, y_va = X_meta.iloc[val_idx], y_meta.iloc[val_idx]
    
    # Meta Model heavily regularized (L2) to handle correlated aggregate features
    model_l1 = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.015, max_depth=4, 
        class_weight='balanced', reg_lambda=2.0, random_state=42
    )
    model_l1.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
    
    oof_meta_probs[val_idx] = model_l1.predict_proba(X_va)
    final_test_preds_prob += model_l1.predict_proba(test_meta) / 5

# =====================================================================
# UPGRADE 3: NELDER-MEAD THRESHOLD OPTIMIZATION
# =====================================================================
print("\n--- OPTIMIZING THRESHOLDS FOR MACRO F1 ---")

# Dynamically find the probability multipliers that yield the highest possible F1 Score
def optimize_thresholds(y_true, y_probs):
    def f1_opt(x):
        pred = np.argmax(y_probs * x, axis=1)
        return -f1_score(y_true, pred, average='macro')
    
    result = minimize(f1_opt, [1.0, 1.0, 1.0], method='Nelder-Mead')
    return result.x

best_weights = optimize_thresholds(y_meta, oof_meta_probs)
print(f"Optimal Probability Modifiers Discovered: {best_weights}")

# Apply optimized multipliers to the Out-Of-Fold predictions
opt_preds_classes = np.argmax(oof_meta_probs * best_weights, axis=1)
final_macro_f1 = f1_score(y_meta, opt_preds_classes, average='macro')

print("\n" + "="*50)
print(f"GRANDMASTER V3 MACRO F1 SCORE: {final_macro_f1:.5f}")
print("="*50)

# Apply the EXACT SAME multipliers to the Test Set predictions
final_predictions = np.argmax(final_test_preds_prob * best_weights, axis=1)

submission = pd.DataFrame({
    'bag_id': test_meta.index,
    'label': final_predictions
}).sort_values('bag_id').reset_index(drop=True)

submission.to_csv('submission_v3_optimized.csv', index=False)
print("Saved final optimized predictions to 'submission_v3_optimized.csv'")