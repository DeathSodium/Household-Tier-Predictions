import pandas as pd
import numpy as np
import os
from scipy.stats import entropy

def ultimate_feature_engineering(input_path, output_path):
    print(f"Generating Ultimate Feature Set...")
    df = pd.read_csv(input_path)
    
    # 1. Individual Level Derivations
    df['age'] = df['survey_year'] - df['year_of_birth']
    df['net_capital'] = df['capital_gain'] - df['capital_loss']
    df['cap_per_hour'] = df['net_capital'] / (df['hours_per_week'] + 1)
    df['edu_per_age'] = df['education_num'] / (df['age'] + 1)
    
    # Age Bins
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 45, 65, 120], labels=['young', 'adult', 'senior', 'elderly'])
    
    # 2. Numerical Aggregations (Expanding to 10 statistics)
    num_cols = ['age', 'education_num', 'hours_per_week', 'net_capital_asset', 'net_capital', 'cap_per_hour', 'edu_per_age']
    stats = ['mean', 'median', 'std', 'max', 'min', 'sum', 'skew', 'kurt'] # Added kurt
    
    agg_dict = {col: stats for col in num_cols}
    agg_dict['bag_id'] = ['count']
    
    if 'label' in df.columns:
        label_map = {'lower': 0, 'middle': 1, 'upper': 2}
        df['label_num'] = df['label'].map(label_map)
        agg_dict['label_num'] = ['first']
        
    df_agg = df.groupby('bag_id').agg(agg_dict)
    df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
    
    # Rename basic info
    rename_map = {'bag_id_count': 'bag_size'}
    if 'label_num_first' in df_agg.columns:
        rename_map['label_num_first'] = 'label'
    df_agg.rename(columns=rename_map, inplace=True)
    
    # 3. Categorical Diversity & Proportions
    cat_cols = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country', 'age_group']
    
    for col in cat_cols:
        # Diversity (Entropy)
        df_agg[f'{col}_entropy'] = df.groupby('bag_id')[col].apply(lambda x: entropy(x.value_counts(normalize=True)))
        # Uniqueness
        df_agg[f'{col}_nunique'] = df.groupby('bag_id')[col].nunique()
        # Mode
        df_agg[f'{col}_mode'] = df.groupby('bag_id')[col].agg(lambda x: str(x.mode().iloc[0]) if not x.mode().empty else "None")
        
        # Proportions for Top Values
        top_vals = df[col].value_counts().index[:6] # Top 6 now
        for val in top_vals:
            safe_val = str(val).replace(' ', '_').replace('-', '_').replace('.', '').replace('/', '_')
            df_agg[f'{col}_{safe_val}_pct'] = df.groupby('bag_id')[col].apply(lambda x: (x == val).mean())
            
    # 4. Specific Percentile Features (Q25, Q75)
    for col in ['age', 'education_num', 'hours_per_week']:
        df_agg[f'{col}_q25'] = df.groupby('bag_id')[col].quantile(0.25)
        df_agg[f'{col}_q75'] = df.groupby('bag_id')[col].quantile(0.75)
        df_agg[f'{col}_iqr'] = df_agg[f'{col}_q75'] - df_agg[f'{col}_q25']

    # 5. Final Cleaning
    df_agg.fillna(0, inplace=True)
    df_agg = df_agg.reset_index()
    
    df_agg.to_csv(output_path, index=False)
    print(f"Saved ULTIMATE features to {output_path}. Shape: {df_agg.shape}")

if __name__ == "__main__":
    raw_train = r"data/raw/Coderush-26-ML-Train.csv"
    output_train = r"data/processed/train_ultimate.csv"
    
    if os.path.exists(raw_train):
        ultimate_feature_engineering(raw_train, output_train)
