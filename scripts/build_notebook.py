import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    # 0. Header & Imports
    nb.cells.append(nbf.v4.new_markdown_cell('# Coderush 2026: Economic Class Classification\n## Competition Submission Notebook\n\nThis notebook follows the required structure: Baseline, Improved, and Best models.'))
    
    nb.cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import entropy

# Configuration
RAW_TRAIN_PATH = "data/raw/Coderush-26-ML-Train.csv"
RANDOM_STATE = 42"""))

    # 1. Baseline Model
    nb.cells.append(nbf.v4.new_markdown_cell('## 1. Baseline Model\n\n**Approach**: Simple LightGBM with minimal features (means of numeric columns).'))
    nb.cells.append(nbf.v4.new_code_cell("""def get_baseline_features(df):
    num_cols = ["age", "education_num", "hours_per_week", "capital_gain", "capital_loss"]
    cat_cols = ["workclass", "occupation", "relationship"]
    
    # Simple aggregations
    df["age"] = df["survey_year"] - df["year_of_birth"]
    df_agg = df.groupby("bag_id").agg({col: "mean" for col in num_cols})
    
    for col in cat_cols:
        df_agg[f"{col}_mode"] = df.groupby("bag_id")[col].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else "None")
    
    if "label" in df.columns:
        df_agg["label"] = df.groupby("bag_id")["label"].first()
    
    return df_agg.reset_index()

raw_df = pd.read_csv(RAW_TRAIN_PATH)
baseline_df = get_baseline_features(raw_df)

le_map = {"lower": 0, "middle": 1, "upper": 2}
baseline_df["label"] = baseline_df["label"].map(le_map)

for col in baseline_df.columns:
    if baseline_df[col].dtype == "object" and col != "bag_id":
        le = LabelEncoder()
        baseline_df[col] = le.fit_transform(baseline_df[col].astype(str))

X_b = baseline_df.drop(["bag_id", "label"], axis=1)
y_b = baseline_df["label"]

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_b, y_b, test_size=0.2, stratify=y_b, random_state=RANDOM_STATE)

model_b = lgb.LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)
model_b.fit(X_train_b, y_train_b)

preds_b = model_b.predict(X_test_b)
print(f"Baseline Macro F1: {f1_score(y_test_b, preds_b, average='macro'):.4f}")"""))

    # 2. Improved Model
    nb.cells.append(nbf.v4.new_markdown_cell('## 2. Improved Model\n\n**Approach**: Tuned LightGBM with Advanced Features (Interaction terms, Skewness, Proportions).'))
    nb.cells.append(nbf.v4.new_code_cell("""# Using the advanced features we already generated
df_adv = pd.read_csv("data/processed/train_advanced.csv")
df_adv["label"] = df_adv["label"].map(le_map)

for col in df_adv.columns:
    if df_adv[col].dtype == "object" and col != "bag_id":
        le = LabelEncoder()
        df_adv[col] = le.fit_transform(df_adv[col].astype(str))

X_i = df_adv.drop(["bag_id", "label"], axis=1)
y_i = df_adv["label"]

X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X_i, y_i, test_size=0.2, stratify=y_i, random_state=RANDOM_STATE)

# Using our tuned parameters
best_lgbm_params = joblib.load("models/tuning/best_lgbm_params.joblib")
model_i = lgb.LGBMClassifier(**best_lgbm_params, n_estimators=500, random_state=RANDOM_STATE, verbose=-1)
model_i.fit(X_train_i, y_train_i)

preds_i = model_i.predict(X_test_i)
print(f"Improved Model Macro F1: {f1_score(y_test_i, preds_i, average='macro'):.4f}")"""))

    # 3. Best Model
    nb.cells.append(nbf.v4.new_markdown_cell('## 3. Best Model\n\n**Approach**: Stacking Ensemble (LightGBM + XGBoost + RandomForest -> Logistic Regression).\n**Why**: Combines the strengths of multiple architectures to reach maximum Macro F1.'))
    nb.cells.append(nbf.v4.new_code_cell("""# Preparing the Best Model (Stacking)
# We will use the Ultimate feature set
df_best = pd.read_csv("data/processed/train_ultimate.csv")
df_best["label"] = df_best["label"].map(le_map)

for col in df_best.columns:
    if df_best[col].dtype == "object" and col != "bag_id":
        le = LabelEncoder()
        df_best[col] = le.fit_transform(df_best[col].astype(str))

X_final = df_best.drop(["bag_id", "label"], axis=1)
y_final = df_best["label"]

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_final, y_final, test_size=0.2, stratify=y_final, random_state=RANDOM_STATE)

# Train Base Models
m1 = lgb.LGBMClassifier(**best_lgbm_params, n_estimators=500, random_state=42, verbose=-1)
m1.fit(X_train_f, y_train_f)

m2 = xgb.XGBClassifier(n_estimators=500, max_depth=7, learning_rate=0.03, random_state=42)
m2.fit(X_train_f, y_train_f)

m3 = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
m3.fit(X_train_f, y_train_f)

# Meta-Model (Stacking)
train_probs = np.hstack([m1.predict_proba(X_train_f), m2.predict_proba(X_train_f), m3.predict_proba(X_train_f)])
test_probs = np.hstack([m1.predict_proba(X_test_f), m2.predict_proba(X_test_f), m3.predict_proba(X_test_f)])

meta_model = LogisticRegression(max_iter=1000)
meta_model.fit(train_probs, y_train_f)

final_preds = meta_model.predict(test_probs)
print(f"Best Model (Stacking) Macro F1: {f1_score(y_test_f, final_preds, average='macro'):.4f}")

# Final Confusion Matrix
cm = confusion_matrix(y_test_f, final_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le_map.keys(), yticklabels=le_map.keys())
plt.title('Final Model Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()"""))

    with open('Coderush_2026_Submission.ipynb', 'w') as f:
        nbf.write(nb, f)
    print("Notebook 'Coderush_2026_Submission.ipynb' has been built successfully.")

if __name__ == "__main__":
    create_notebook()
