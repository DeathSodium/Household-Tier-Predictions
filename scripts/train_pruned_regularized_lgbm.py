import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

def train_pruned_model(data_path, importance_path, output_dir):
    print("Training Pruned & Regularized Model...")
    df = pd.read_csv(data_path)
    df_imp = pd.read_csv(importance_path)
    
    # Keep Top 60 features
    top_features = df_imp.head(60)['feature'].tolist()
    
    # Label Handling
    le_map = {"lower": 0, "middle": 1, "upper": 2}
    if df["label"].dtype == object:
        df["label"] = df["label"].map(le_map)
    y = df['label'].astype(int)
    
    # Encoder for categorical modes (if in top features)
    encoders = {}
    for col in top_features:
        if col.endswith('_mode'):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            
    X = df[top_features]
    
    # 5-Fold OOF with Heavy Regularization
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(df))
    
    # Regularized Parameters
    reg_params = {
        'n_estimators': 1000,
        'learning_rate': 0.02,
        'num_leaves': 15,          # Much smaller (Prev: 31)
        'max_depth': 5,            # Shallow (Prev: 12)
        'min_data_in_leaf': 150,   # High (Prev: 20)
        'feature_fraction': 0.6,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'lambda_l1': 1.0,          # Added L1
        'lambda_l2': 1.0,          # Added L2
        'random_state': 42,
        'verbose': -1
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**reg_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(stopping_rounds=100)])
        
        oof_preds[val_idx] = model.predict(X_val)
        
    final_oof_f1 = f1_score(y, oof_preds, average='macro')
    print(f"\nPruned OOF Macro F1: {final_oof_f1:.4f}")
    
    # Final fit on all data
    final_model = lgb.LGBMClassifier(**reg_params)
    final_model.fit(X, y)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump({
        'model': final_model,
        'encoders': encoders,
        'features': top_features
    }, os.path.join(output_dir, "pruned_reg_lgbm.joblib"))
    print(f"Pruned model saved to {output_dir}")

if __name__ == "__main__":
    train_pruned_model(
        r"data/processed/train_ultimate.csv",
        r"data/processed/feature_importance_list.csv",
        r"models/pruned_lgbm"
    )
