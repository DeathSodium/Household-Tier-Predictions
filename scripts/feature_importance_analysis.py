import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

def analyze_feature_importance(model_pkg_path, output_path):
    print(f"Analyzing Feature Importance from {model_pkg_path}...")
    
    # Load Model
    payload = joblib.load(model_pkg_path)
    model = payload['model']
    features = payload['features']
    
    # Get Importance (Gain is usually better for significance than split count)
    importances = model.feature_importances_
    importance_type = model.importance_type
    
    df_imp = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
    # Plot Top 40
    plt.figure(figsize=(10, 12))
    sns.barplot(x='importance', y='feature', data=df_imp.head(40))
    plt.title(f'Top 40 Features (Importance Type: {importance_type})')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Importance plot saved to {output_path}")
    
    # Identify low-importance features
    zero_imp = df_imp[df_imp['importance'] == 0]
    print(f"Found {len(zero_imp)} features with ZERO importance.")
    
    # Save the full importance list
    df_imp.to_csv("data/processed/feature_importance_list.csv", index=False)
    
    # Propose Top 50
    top_50 = df_imp.head(50)['feature'].tolist()
    print(f"\nTop 10 Features:")
    for i, f in enumerate(top_50[:10]):
        print(f"{i+1}. {f}")
        
    return top_50

if __name__ == "__main__":
    model_pkg = r"models/tuned_lgbm/tuned_lgbm_model.joblib"
    plot_out = r"artifacts/feature_importance.png"
    
    if os.path.exists(model_pkg):
        analyze_feature_importance(model_pkg, plot_out)
    else:
        print("Model package not found.")
