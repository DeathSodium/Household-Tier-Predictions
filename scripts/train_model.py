import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import joblib

def train_model(train_path, model_dir):
    print("Loading aggregated training data...")
    df = pd.read_csv(train_path)
    
    # Identify categorical columns (the '_mode' ones)
    cat_features = [col for col in df.columns if col.endswith('_mode')]
    
    # Label Encoding for categorical features
    encoders = {}
    for col in cat_features:
        le = LabelEncoder()
        # Handle unseen labels in future test data by adding a 'missing' category
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        
    X = df.drop(['bag_id', 'label'], axis=1)
    y = df['label']
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(df))
    models = []
    
    print("Starting cross-validation training...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Optimized LightGBM model
        model = lgb.LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=7,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            importance_type='gain',
            verbose=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='multi_logloss',
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
        
        oof_preds[val_idx] = model.predict(X_val)
        models.append(model)
        
        fold_f1 = f1_score(y_val, oof_preds[val_idx], average='macro')
        print(f"Fold {fold+1} Macro F1: {fold_f1:.4f}")
        
    overall_f1 = f1_score(y, oof_preds, average='macro')
    print(f"\nOverall OOF Macro F1: {overall_f1:.4f}")
    
    # Save OOF predictions for realistic evaluation
    oof_df = pd.DataFrame({'bag_id': df['bag_id'], 'true_label': y, 'oof_pred': oof_preds})
    oof_df.to_csv(os.path.join(r"data/processed", "oof_predictions.csv"), index=False)
    
    # Save models and encoders
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(models, os.path.join(model_dir, "lgb_models.joblib"))
    joblib.dump(encoders, os.path.join(model_dir, "label_encoders.joblib"))
    print(f"Saved models and encoders to {model_dir}")

if __name__ == "__main__":
    train_aggregated_path = r"data/processed/train_aggregated.csv"
    models_directory = r"models"
    
    if os.path.exists(train_aggregated_path):
        train_model(train_aggregated_path, models_directory)
    else:
        print(f"Aggregated training data not found at {train_aggregated_path}")
