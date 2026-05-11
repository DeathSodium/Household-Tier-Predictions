import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import os

def show_oof_confusion_matrix(oof_path):
    if not os.path.exists(oof_path):
        print(f"OOF predictions not found at {oof_path}")
        return
        
    df = pd.read_csv(oof_path)
    y_true = df['true_label']
    y_pred = df['pred']
    
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, target_names=['lower', 'middle', 'upper'])
    
    print("\n--- REALISTIC (OOF) CLASSIFICATION REPORT ---")
    print(cr)
    
    print("\n--- REALISTIC (OOF) CONFUSION MATRIX ---")
    print(cm)
    
    # Prettier output for the user
    labels = ['lower', 'middle', 'upper']
    cm_df = pd.DataFrame(cm, index=[f"Actual {l}" for l in labels], columns=[f"Predicted {l}" for l in labels])
    print("\nConfusion Matrix Table:")
    print(cm_df)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="ensemble_oof_predictions.csv", help="OOF filename in data/processed")
    args = parser.parse_args()
    
    oof_predictions_path = os.path.join(r"data/processed", args.file)
    show_oof_confusion_matrix(oof_predictions_path)
