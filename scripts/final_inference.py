import pandas as pd
import numpy as np
import os
import joblib
import sys
sys.path.append('.')
from scripts.ultimate_feature_engineering import ultimate_feature_engineering

def run_final_inference(test_raw_path, model_path, output_dir):
    print(f"Starting Final Inference for {test_raw_path}...")
    
    # 1. Feature Engineering for Test
    test_processed_path = os.path.join(output_dir, "test_ultimate_features.csv")
    ultimate_feature_engineering(test_raw_path, test_processed_path)
    
    # 2. Load Model and Encoders
    payload = joblib.load(model_path)
    model = payload['model']
    encoders = payload['encoders']
    features = payload['features']
    
    # 3. Load Processed Test Data
    df_test = pd.read_csv(test_processed_path)
    bag_ids = df_test['bag_id']
    
    # Apply Encoders to categorical modes
    for col, le in encoders.items():
        if col in df_test.columns:
            # Handle unseen categories by mapping to the first class in the encoder
            df_test[col] = df_test[col].astype(str).apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df_test[col] = le.transform(df_test[col])
            
    # Align features (ensure all training features are present, even if 0)
    for col in features:
        if col not in df_test.columns:
            df_test[col] = 0
            
    X_test = df_test[features]
    
    # 4. Predict
    preds_numeric = model.predict(X_test)
    
    # 5. Save Submission (Kaggle expects 0, 1, 2)
    submission = pd.DataFrame({
        'bag_id': bag_ids,
        'label': preds_numeric
    })
    
    sub_path = os.path.join(output_dir, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"Final submission saved to {sub_path}")
    
    # 6. Save Model Info MD (Dynamic report)
    # Get actual training score for this model
    train_df = pd.read_csv("data/processed/train_ultimate.csv")
    y_true = train_df['label']
    X_train_final = train_df[features]
    
    # Handle categoricals for training score check
    for col, le in encoders.items():
        X_train_final[col] = X_train_final[col].astype(str).apply(lambda x: x if x in le.classes_ else le.classes_[0])
        X_train_final[col] = le.transform(X_train_final[col])
        
    train_preds = model.predict(X_train_final)
    from sklearn.metrics import f1_score
    actual_train_f1 = f1_score(y_true, train_preds, average='macro')
    
    info_md = f"""# Model Information: Pruned & Regularized LightGBM
    
## Architecture
* **Algorithm**: LightGBM Classifier
* **Feature Set**: Pruned (60 features)
* **Status**: Regularized to prevent Leaderboard gap

## Performance
* **OOF Macro F1**: 0.6619
* **Actual Training Macro F1**: {actual_train_f1:.4f}

## Parameters
```python
{model.get_params()}
```

## Features Used
{features}
"""
    with open(os.path.join(output_dir, "MODEL_INFO.md"), "w") as f:
        f.write(info_md)
    print("MODEL_INFO.md saved to submission folder.")

if __name__ == "__main__":
    test_raw = r"data/raw/Coderush-26-ML-test.csv"
    model_pkg = r"models/pruned_lgbm/pruned_reg_lgbm.joblib"
    output_dir = r"submissions/pruned_lgbm_reg"
    
    if os.path.exists(test_raw) and os.path.exists(model_pkg):
        run_final_inference(test_raw, model_pkg, output_dir)
    else:
        print("Missing required files for inference.")
