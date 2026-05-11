import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

def advanced_feature_engineering(input_path, output_path):
    print(f"Processing data with advanced features...")
    df = pd.read_csv(input_path)
    
    # 1. Feature Derivation & Interactions
    df['age'] = df['survey_year'] - df['year_of_birth']
    df['education_hours'] = df['education_num'] * df['hours_per_week']
    df['capital_diff'] = df['capital_gain'] - df['capital_loss']
    df['below_poverty'] = (df['net_capital_asset'] < df['poverty_line_usd']).astype(int)
    
    # 2. Aggregations
    num_cols = ['age', 'education_num', 'hours_per_week', 'net_capital_asset', 'capital_gain', 'capital_loss', 'survey_duration_mins', 'education_hours', 'capital_diff']
    cat_cols = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country', 'education_tier']
    
    # Numerical aggregations: Adding more moments
    agg_dict = {col: ['mean', 'median', 'std', 'max', 'min', 'sum', 'skew'] for col in num_cols}
    agg_dict['bag_size'] = ['first']
    agg_dict['below_poverty'] = ['mean', 'sum']
    
    if 'label' in df.columns:
        # Map labels for target encoding (temporary)
        label_map = {'lower': 0, 'middle': 1, 'upper': 2}
        df['label_num'] = df['label'].map(label_map)
        agg_dict['label_num'] = ['first']
        
    df_agg = df.groupby('bag_id').agg(agg_dict)
    df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
    
    # Rename key columns
    rename_map = {'bag_size_first': 'bag_size', 'below_poverty_mean': 'poverty_rate', 'below_poverty_sum': 'poverty_count'}
    if 'label_num_first' in df_agg.columns:
        rename_map['label_num_first'] = 'label'
    df_agg.rename(columns=rename_map, inplace=True)
    
    # 3. Categorical Proportions (Enhanced)
    for col in cat_cols:
        df_agg[f'{col}_nunique'] = df.groupby('bag_id')[col].nunique()
        df_agg[f'{col}_mode'] = df.groupby('bag_id')[col].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        
        # Calculate percentages for the top 5 values of each category
        top_vals = df[col].value_counts().index[:5]
        for val in top_vals:
            safe_val = str(val).replace(' ', '_').replace('-', '_').replace('.', '')
            df_agg[f'{col}_{safe_val}_pct'] = df.groupby('bag_id')[col].apply(lambda x: (x == val).mean())
            
    # 4. Target Encoding (Bag Level)
    # Since we are at the bag level already, and labels are at the bag level, 
    # we can't target encode individual bags with their own label.
    # We could do target encoding for categorical MODES across the whole dataset.
    # But let's skip that for now to avoid complexity and focus on distribution features.
    
    # 5. Final Cleaning
    df_agg.fillna(0, inplace=True)
    df_agg = df_agg.reset_index()
    
    df_agg.to_csv(output_path, index=False)
    print(f"Saved advanced features to {output_path}. Shape: {df_agg.shape}")

if __name__ == "__main__":
    raw_train = r"data/raw/Coderush-26-ML-Train.csv"
    output_train = r"data/processed/train_advanced.csv"
    
    if os.path.exists(raw_train):
        advanced_feature_engineering(raw_train, output_train)
