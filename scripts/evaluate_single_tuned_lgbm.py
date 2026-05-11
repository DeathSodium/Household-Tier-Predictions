import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

def evaluate_tuned_lgbm(split_dir):
    train_df = pd.read_csv(os.path.join(split_dir, "train_split.csv"))
    val_df = pd.read_csv(os.path.join(split_dir, "val_split.csv"))
    test_df = pd.read_csv(os.path.join(split_dir, "test_split.csv"))
    
    best_params = joblib.load("models/tuning/best_lgbm_params.joblib")
    
    # Preprocessing
    cat_features = [col for col in train_df.columns if col.endswith('_mode')]
    encoders = {}
    for col in cat_features:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        val_df[col] = val_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        test_df[col] = test_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        val_df[col] = le.transform(val_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
        encoders[col] = le
        
    X_train, y_train = train_df.drop(['bag_id', 'label'], axis=1), train_df['label']
    X_val, y_val = val_df.drop(['bag_id', 'label'], axis=1), val_df['label']
    X_test, y_test = test_df.drop(['bag_id', 'label'], axis=1), test_df['label']
    
    # Train single best model
    model = lgb.LGBMClassifier(**best_params, n_estimators=2000, random_state=42, verbose=-1)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
              callbacks=[lgb.early_stopping(stopping_rounds=50)])
    
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds, average='macro')
    
    print(f"--- SINGLE TUNED LGBM RESULTS ---")
    print(f"Hold-out Test Macro F1: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=['lower', 'middle', 'upper']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

if __name__ == "__main__":
    split_dir = r"data/processed/split"
    evaluate_tuned_lgbm(split_dir)
