import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def run_oof_validation(data_path, output_dir):
    print(f"Loading data from {data_path} for OOF validation...")
    df = pd.read_csv(data_path)
    
    # Label Handling
    le_map = {"lower": 0, "middle": 1, "upper": 2}
    if df["label"].dtype == object:
        df["label"] = df["label"].map(le_map)
    
    y = df['label'].astype(int)
    
    cat_features = [col for col in df.columns if col.endswith('_mode')]
    for col in cat_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        
    X = df.drop(['bag_id', 'label'], axis=1)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    train_scores = []
    val_scores = []
    oof_preds = np.zeros(len(df))
    
    print("Starting 5-Fold Cross-Validation...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Best Parameters (using our previously tuned ones)
        # For speed in this demo, I'll use standard tuned params
        model = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=31, max_depth=7, 
            random_state=42, verbose=-1
        )
        
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                  callbacks=[lgb.early_stopping(stopping_rounds=50)])
        
        # Predictions
        tr_preds = model.predict(X_train)
        v_preds = model.predict(X_val)
        
        tr_f1 = f1_score(y_train, tr_preds, average='macro')
        v_f1 = f1_score(y_val, v_preds, average='macro')
        
        train_scores.append(tr_f1)
        val_scores.append(v_f1)
        oof_preds[val_idx] = v_preds
        
        print(f"Fold {fold+1}: Train F1 = {tr_f1:.4f}, Val F1 = {v_f1:.4f}")
        
    avg_train = np.mean(train_scores)
    avg_val = np.mean(val_scores)
    print(f"\n--- CV RESULTS ---")
    print(f"Average Train Macro F1: {avg_train:.4f}")
    print(f"Average OOF Val Macro F1: {avg_val:.4f}")
    
    # Plotting
    plt.figure(figsize=(12, 5))
    
    # 1. Fold Performance Plot
    plt.subplot(1, 2, 1)
    folds = [f"Fold {i+1}" for i in range(5)]
    x = np.arange(len(folds))
    width = 0.35
    plt.bar(x - width/2, train_scores, width, label='Train F1', color='#3498db')
    plt.bar(x + width/2, val_scores, width, label='Val F1 (OOF)', color='#e74c3c')
    plt.axhline(avg_val, color='red', linestyle='--', label=f'Avg Val ({avg_val:.2f})')
    plt.ylabel('Macro F1 Score')
    plt.title('Performance per Fold')
    plt.xticks(x, folds)
    plt.legend()
    plt.ylim(0, 1.1)
    
    # 2. Confusion Matrix Heatmap
    plt.subplot(1, 2, 2)
    cm = confusion_matrix(y, oof_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=le_map.keys(), yticklabels=le_map.keys())
    plt.title('OOF Confusion Matrix (Total)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "oof_validation_plots.png")
    plt.savefig(plot_path)
    print(f"Plots saved to {plot_path}")
    plt.show()

if __name__ == "__main__":
    ultimate_path = r"data/processed/train_ultimate.csv"
    output_dir = r"artifacts"
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(ultimate_path):
        run_oof_validation(ultimate_path, output_dir)
    else:
        print(f"Data not found.")
