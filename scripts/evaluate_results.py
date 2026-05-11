import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_confusion_matrix(train_path, model_dir):
    print("Loading data and models for evaluation...")
    df = pd.read_csv(train_path)
    models = joblib.load(os.path.join(model_dir, "lgb_models.joblib"))
    encoders = joblib.load(os.path.join(model_dir, "label_encoders.joblib"))
    
    # Identify categorical columns
    cat_features = [col for col in df.columns if col.endswith('_mode')]
    
    # Apply encoders
    for col in cat_features:
        le = encoders[col]
        df[col] = df[col].astype(str)
        df[col] = le.transform(df[col])
        
    X = df.drop(['bag_id', 'label'], axis=1)
    y = df['label']
    
    # Generate ensemble predictions
    all_preds = []
    for model in models:
        all_preds.append(model.predict(X))
        
    # Majority vote for final prediction
    final_preds = pd.DataFrame(all_preds).mode().iloc[0].astype(int)
    
    # Confusion Matrix
    cm = confusion_matrix(y, final_preds)
    cr = classification_report(y, final_preds, target_names=['lower', 'middle', 'upper'])
    
    print("\n--- CLASSIFICATION REPORT ---")
    print(cr)
    
    print("\n--- CONFUSION MATRIX ---")
    print(cm)
    
    # Save confusion matrix to a file for the user
    output_file = r"C:\Users\mrkin\.gemini\antigravity\brain\6481ecbd-da3d-4e5c-8c9c-f4aad005b447\scratch\confusion_matrix.txt"
    with open(output_file, "w") as f:
        f.write("Classification Report:\n")
        f.write(cr)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))
    
    return cm, cr

if __name__ == "__main__":
    train_aggregated_path = r"data/processed/train_aggregated.csv"
    models_directory = r"models"
    
    if os.path.exists(train_aggregated_path):
        evaluate_confusion_matrix(train_aggregated_path, models_directory)
    else:
        print(f"Aggregated training data not found at {train_aggregated_path}")
