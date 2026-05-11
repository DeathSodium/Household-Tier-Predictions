"""
MIL Interpretability: Analyze Attention Weights.
Goal: Understand which household members the model prioritizes.
"""
import sys, os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from scripts_v2.v2_mil_attention import RobustAttentionMIL, collate_mil

def interpret_attention():
    print("="*60)
    print("MIL ATTENTION INTERPRETABILITY REPORT")
    print("="*60)
    
    # 1. Load Data & Model
    data_res = joblib.load('data/processed/train_instances.joblib')
    df = data_res['data']
    num_cols = data_res['num_cols']
    cat_cols = data_res['cat_cols']
    cat_sizes = [df[c].max() + 1 for c in cat_cols]
    
    device = torch.device("cpu")
    model = RobustAttentionMIL(len(num_cols), cat_sizes)
    model.load_state_dict(torch.load('models/v2/robust_mil_fold0.pt'))
    model.eval()
    
    # Inverse map for categoricals to make it readable
    encoders = data_res['encoders']
    
    # 2. Pick sample bags (one for each class)
    sample_bids = []
    for cls in ['lower', 'middle', 'upper']:
        sample_bids.append(df[df['label'] == cls]['bag_id'].iloc[0])
    
    results = []
    
    with torch.no_grad():
        for bid in sample_bids:
            group = df[df['bag_id'] == bid].copy()
            label = group['label'].iloc[0]
            
            # Prepare for model
            nums = torch.FloatTensor(group[num_cols].values).unsqueeze(0)
            cats = torch.LongTensor(group[cat_cols].values).unsqueeze(0)
            mask = torch.ones((1, len(group)))
            
            # Forward pass to get attention
            # We need to extract 'a' from the forward pass. 
            # I'll modify the forward pass slightly or use a hook.
            # Actually, I'll just re-implement the forward logic here for simplicity.
            embeddings = [model.embs[i](cats[:, :, i]) for i in range(len(model.embs))]
            x = torch.cat([nums] + embeddings, dim=2)
            h = model.feature_extractor(x)
            a = model.attention(h).squeeze(-1)
            a = torch.softmax(a, dim=1).squeeze(0).numpy()
            
            group['attention_weight'] = a
            
            # Decode categoricals for readability
            for col in cat_cols:
                group[col] = encoders[col].inverse_transform(group[col])
                
            # Unscale numerics (approximate)
            # group[num_cols] = data_res['scaler'].inverse_transform(group[num_cols])
            
            print(f"\nBag ID: {bid} | True Label: {label.upper()}")
            print("-" * 120)
            cols_to_show = ['age', 'education', 'occupation', 'relationship', 'sex', 'attention_weight']
            print(group[cols_to_show].sort_values('attention_weight', ascending=False).to_string(index=False))
            print("-" * 120)

if __name__ == '__main__':
    interpret_attention()
