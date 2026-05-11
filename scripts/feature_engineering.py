import pandas as pd
import numpy as np
import os

def feature_engineering(input_path, output_path, is_train=True):
    print(f"Processing {'Training' if is_train else 'Testing'} data...")
    df = pd.read_csv(input_path)
    
    # 1. Basic Cleaning & Feature Derivation
    df['age'] = df['survey_year'] - df['year_of_birth']
    
    # Poverty Floor Feature: Is individual below poverty line?
    # (Assuming poverty_line_usd is a reference value to compare against net_capital_asset or similar)
    # However, poverty_line_usd in this dataset seems to be a constant or variable threshold.
    # Let's create a feature: net_capital_asset < poverty_line_usd
    df['below_poverty'] = (df['net_capital_asset'] < df['poverty_line_usd']).astype(int)
    
    if is_train:
        # Map labels to numeric
        label_map = {'lower': 0, 'middle': 1, 'upper': 2}
        df['label'] = df['label'].map(label_map)
    
    # 2. Define Aggregations
    num_cols = ['age', 'education_num', 'hours_per_week', 'net_capital_asset', 'capital_gain', 'capital_loss', 'survey_duration_mins']
    cat_cols = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country', 'education_tier']
    
    # Numerical aggregations: Adding 'min' and 'var' for more distribution info
    agg_dict = {col: ['mean', 'median', 'std', 'max', 'sum', 'min'] for col in num_cols}
    agg_dict['bag_size'] = ['first']
    agg_dict['below_poverty'] = ['mean'] # This gives the percentage of people below poverty line in the bag
    
    if is_train:
        agg_dict['label'] = ['first']
        
    # Group by bag_id and aggregate numericals
    df_agg = df.groupby('bag_id').agg(agg_dict)
    
    # Flatten multi-index columns
    df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
    
    # Clean up names
    rename_map = {'bag_size_first': 'bag_size', 'below_poverty_mean': 'poverty_rate'}
    if is_train: rename_map['label_first'] = 'label'
    df_agg.rename(columns=rename_map, inplace=True)
    
    # 3. Categorical Aggregations (per bag)
    for col in cat_cols:
        # Nunique (diversity in the bag)
        df_agg[f'{col}_nunique'] = df.groupby('bag_id')[col].nunique()
        
        # Mode (most common in the bag)
        df_agg[f'{col}_mode'] = df.groupby('bag_id')[col].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        
        # Proportions of top 3 most common values overall (to keep feature space manageable)
        top_vals = df[col].value_counts().index[:3]
        for val in top_vals:
            df_agg[f'{col}_{val}_pct'] = df.groupby('bag_id')[col].apply(lambda x: (x == val).mean())
        
    # 4. Fill NaNs (std can be NaN for bags of size 1)
    df_agg.fillna(0, inplace=True)
    
    # Reset index to have bag_id as a column
    df_agg = df_agg.reset_index()
    
    # Save processed data
    df_agg.to_csv(output_path, index=False)
    print(f"Saved aggregated data to {output_path}. Shape: {df_agg.shape}")

if __name__ == "__main__":
    raw_dir = r"data/raw"
    processed_dir = r"data/processed"
    
    train_input = os.path.join(raw_dir, "Coderush-26-ML-Train.csv")
    train_output = os.path.join(processed_dir, "train_aggregated.csv")
    
    test_input = os.path.join(raw_dir, "Coderush-26-ML-test.csv")
    test_output = os.path.join(processed_dir, "test_aggregated.csv")
    
    if os.path.exists(train_input):
        feature_engineering(train_input, train_output, is_train=True)
    
    if os.path.exists(test_input):
        feature_engineering(test_input, test_output, is_train=False)
