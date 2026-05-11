import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb

def train_stacking(train_path, model_dir):
    print("Loading aggregated training data for Stacking...")
    df = pd.read_csv(train_path)
    
    # Identify categorical columns
    cat_features = [col for col in df.columns if col.endswith('_mode')]
    
    # Label Encoding for categorical features
    encoders = {}
    for col in cat_features:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        
    X = df.drop(['bag_id', 'label'], axis=1)
    y = df['label']
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # To store OOF predictions (probabilities)
    n_classes = 3
    base_oof_probs = {
        'lgb': np.zeros((len(df), n_classes)),
        'xgb': np.zeros((len(df), n_classes)),
        'rf': np.zeros((len(df), n_classes))
    }
    
    all_base_models = {
        'lgb': [],
        'xgb': [],
        'rf': []
    }
    
    print("Step 1: Training Base Models and generating OOF features...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 1. LightGBM
        model_lgb = lgb.LGBMClassifier(
            n_estimators=1000, learning_rate=0.03, num_leaves=63, max_depth=7,
            random_state=42, verbose=-1
        )
        model_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                      callbacks=[lgb.early_stopping(stopping_rounds=50)], eval_metric='multi_logloss')
        base_oof_probs['lgb'][val_idx] = model_lgb.predict_proba(X_val)
        all_base_models['lgb'].append(model_lgb)
        
        # 2. XGBoost
        model_xgb = xgb.XGBClassifier(
            n_estimators=1000, learning_rate=0.03, max_depth=7,
            early_stopping_rounds=50,
            random_state=42, use_label_encoder=False, eval_metric='mlogloss'
        )
        model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        base_oof_probs['xgb'][val_idx] = model_xgb.predict_proba(X_val)
        all_base_models['xgb'].append(model_xgb)
        
        # 3. Random Forest
        model_rf = RandomForestClassifier(
            n_estimators=300, max_depth=12, random_state=42, n_jobs=-1
        )
        model_rf.fit(X_train, y_train)
        base_oof_probs['rf'][val_idx] = model_rf.predict_proba(X_val)
        all_base_models['rf'].append(model_rf)
        
        print(f"Fold {fold+1} completed.")

    # Prepare features for Meta-Model
    # Concatenate probabilities: (N, 3) + (N, 3) + (N, 3) -> (N, 9)
    X_meta = np.hstack([base_oof_probs['lgb'], base_oof_probs['xgb'], base_oof_probs['rf']])
    
    print("\nStep 2: Training Meta-Model (Logistic Regression)...")
    # Using 5-fold CV for Meta-Model too to get a realistic score
    meta_oof_preds = np.zeros(len(df))
    meta_models = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_meta, y)):
        X_meta_train, X_meta_val = X_meta[train_idx], X_meta[val_idx]
        y_meta_train, y_meta_val = y[train_idx], y[val_idx]
        
        meta_model = LogisticRegression(max_iter=1000, random_state=42)
        meta_model.fit(X_meta_train, y_meta_train)
        meta_oof_preds[val_idx] = meta_model.predict(X_meta_val)
        meta_models.append(meta_model)
        
    stacking_f1 = f1_score(y, meta_oof_preds, average='macro')
    print(f"\n--- STACKING RESULTS ---")
    print(f"Stacking (LGBM + XGB + RF -> LogReg) Macro F1: {stacking_f1:.4f}")
    
    # Save OOF for analysis
    oof_df = pd.DataFrame({
        'bag_id': df['bag_id'], 
        'true_label': y, 
        'pred': meta_oof_preds.astype(int)
    })
    oof_df.to_csv(os.path.join(r"data/processed", "stacking_oof_predictions.csv"), index=False)
    
    # Save everything
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump({
        'base_models': all_base_models,
        'meta_models': meta_models,
        'encoders': encoders
    }, os.path.join(model_dir, "stacking_package.joblib"))
    print(f"Stacking package saved to {model_dir}")

if __name__ == "__main__":
    train_aggregated_path = r"data/processed/train_aggregated.csv"
    stacking_model_dir = r"models/stacking"
    
    if os.path.exists(train_aggregated_path):
        train_stacking(train_aggregated_path, stacking_model_dir)
    else:
        print(f"Aggregated training data not found.")
