import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb

def train_ensemble(train_path, model_dir):
    print("Loading aggregated training data for ensemble...")
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
    oof_probs_lgb = np.zeros((len(df), n_classes))
    oof_probs_xgb = np.zeros((len(df), n_classes))
    oof_probs_rf = np.zeros((len(df), n_classes))
    
    all_models = {
        'lgb': [],
        'xgb': [],
        'rf': []
    }
    
    print("Starting Ensemble Cross-Validation training...")
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
        oof_probs_lgb[val_idx] = model_lgb.predict_proba(X_val)
        all_models['lgb'].append(model_lgb)
        
        # 2. XGBoost
        model_xgb = xgb.XGBClassifier(
            n_estimators=1000, learning_rate=0.03, max_depth=7,
            early_stopping_rounds=50,
            random_state=42, use_label_encoder=False, eval_metric='mlogloss'
        )
        model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        oof_probs_xgb[val_idx] = model_xgb.predict_proba(X_val)
        all_models['xgb'].append(model_xgb)
        
        # 3. Random Forest
        model_rf = RandomForestClassifier(
            n_estimators=300, max_depth=12, random_state=42, n_jobs=-1
        )
        model_rf.fit(X_train, y_train)
        oof_probs_rf[val_idx] = model_rf.predict_proba(X_val)
        all_models['rf'].append(model_rf)
        
        print(f"Fold {fold+1} completed.")

    # Ensemble: Simple average of probabilities
    oof_probs_ensemble = (oof_probs_lgb + oof_probs_xgb + oof_probs_rf) / 3
    oof_preds_ensemble = np.argmax(oof_probs_ensemble, axis=1)
    
    # Calculate scores
    lgb_f1 = f1_score(y, np.argmax(oof_probs_lgb, axis=1), average='macro')
    xgb_f1 = f1_score(y, np.argmax(oof_probs_xgb, axis=1), average='macro')
    rf_f1 = f1_score(y, np.argmax(oof_probs_rf, axis=1), average='macro')
    ensemble_f1 = f1_score(y, oof_preds_ensemble, average='macro')
    
    print(f"\n--- ENSEMBLE RESULTS ---")
    print(f"LGBM Macro F1: {lgb_f1:.4f}")
    print(f"XGBoost Macro F1: {xgb_f1:.4f}")
    print(f"RandomForest Macro F1: {rf_f1:.4f}")
    print(f"Ensemble (Voting) Macro F1: {ensemble_f1:.4f}")
    
    # Save OOF for analysis
    oof_df = pd.DataFrame({
        'bag_id': df['bag_id'], 
        'true_label': y, 
        'pred': oof_preds_ensemble
    })
    oof_df.to_csv(os.path.join(r"data/processed", "ensemble_oof_predictions.csv"), index=False)
    
    # Save models and encoders
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(all_models, os.path.join(model_dir, "ensemble_models.joblib"))
    joblib.dump(encoders, os.path.join(model_dir, "ensemble_encoders.joblib"))
    print(f"Ensemble models saved to {model_dir}")

if __name__ == "__main__":
    train_aggregated_path = r"data/processed/train_aggregated.csv"
    ensemble_model_dir = r"models/ensemble"
    
    if os.path.exists(train_aggregated_path):
        train_ensemble(train_aggregated_path, ensemble_model_dir)
    else:
        print(f"Aggregated training data not found.")
