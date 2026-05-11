import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb

def train_final_stacking(split_dir, model_dir):
    print("Loading split datasets...")
    train_df = pd.read_csv(os.path.join(split_dir, "train_split.csv"))
    val_df = pd.read_csv(os.path.join(split_dir, "val_split.csv"))
    test_df = pd.read_csv(os.path.join(split_dir, "test_split.csv"))
    
    # Identify categorical columns
    cat_features = [col for col in train_df.columns if col.endswith('_mode')]
    
    # Label Encoding
    encoders = {}
    for col in cat_features:
        le = LabelEncoder()
        train_df[col] = train_df[col].astype(str)
        val_df[col] = val_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)
        
        train_df[col] = le.fit_transform(train_df[col])
        # Mapping unseen values in val/test to the first class
        val_df[col] = val_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        test_df[col] = test_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        
        val_df[col] = le.transform(val_df[col])
        test_df[col] = le.transform(test_df[col])
        encoders[col] = le
        
    X_train, y_train = train_df.drop(['bag_id', 'label'], axis=1), train_df['label']
    X_val, y_val = val_df.drop(['bag_id', 'label'], axis=1), val_df['label']
    X_test, y_test = test_df.drop(['bag_id', 'label'], axis=1), test_df['label']
    
    print("Step 1: Training Base Models with Tuned Parameters...")
    lgbm_best = joblib.load("models/tuning/best_lgbm_params.joblib")
    xgb_best = joblib.load("models/tuning/best_xgb_params.joblib")
    
    # 1. LightGBM
    model_lgb = lgb.LGBMClassifier(**lgbm_best, n_estimators=2000, random_state=42, verbose=-1)
    model_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=50)])
    
    # 2. XGBoost
    model_xgb = xgb.XGBClassifier(**xgb_best, n_estimators=1000, random_state=42)
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # 3. Random Forest
    model_rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    model_rf.fit(X_train, y_train)
    
    print("Step 2: Preparing Meta-Features...")
    # Get probabilities on Validation set to train Meta-Model
    val_probs_lgb = model_lgb.predict_proba(X_val)
    val_probs_xgb = model_xgb.predict_proba(X_val)
    val_probs_rf = model_rf.predict_proba(X_val)
    X_meta_val = np.hstack([val_probs_lgb, val_probs_xgb, val_probs_rf])
    
    # Train Meta-Model on Validation set probabilities
    meta_model = LogisticRegression(max_iter=1000, random_state=42)
    meta_model.fit(X_meta_val, y_val)
    
    print("Step 3: Evaluating on Hold-out Test Split...")
    # Get probabilities on Hold-out Test set
    test_probs_lgb = model_lgb.predict_proba(X_test)
    test_probs_xgb = model_xgb.predict_proba(X_test)
    test_probs_rf = model_rf.predict_proba(X_test)
    X_meta_test = np.hstack([test_probs_lgb, test_probs_xgb, test_probs_rf])
    
    final_preds = meta_model.predict(X_meta_test)
    
    f1 = f1_score(y_test, final_preds, average='macro')
    print(f"\nFinal Hold-out Test Macro F1: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, final_preds, target_names=['lower', 'middle', 'upper']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, final_preds))
    
    # Save everything
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump({
        'base_models': {'lgb': model_lgb, 'xgb': model_xgb, 'rf': model_rf},
        'meta_model': meta_model,
        'encoders': encoders
    }, os.path.join(model_dir, "final_stacking_package.joblib"))
    print(f"Final Stacking package saved to {model_dir}")

if __name__ == "__main__":
    split_dir = r"data/processed/split"
    final_model_dir = r"models/stacking"
    
    if os.path.exists(split_dir):
        train_final_stacking(split_dir, final_model_dir)
    else:
        print(f"Split data not found.")
