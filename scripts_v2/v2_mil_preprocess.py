"""
MIL Preprocessing: Prepare instance-level data for Attention Network.
Encodes categoricals and scales numerics at the individual level.
"""
import pandas as pd
import numpy as np
import os, joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_instances(input_path, output_path, is_train=True, scaler=None, encoders=None):
    print(f"Preprocessing instances: {input_path}")
    df = pd.read_csv(input_path)

    # 1. Feature Engineering (Individual level)
    df['age'] = df['survey_year'] - df['year_of_birth']
    
    # Select raw columns to use
    num_cols = ['age', 'education_num', 'hours_per_week', 'capital_gain', 'capital_loss', 'annual_hours_est']
    cat_cols = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']

    # 2. Encode Categoricals
    if encoders is None:
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        for col in cat_cols:
            le = encoders[col]
            # Handle unseen labels by mapping to a default/most frequent
            df[col] = df[col].astype(str).map(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])

    # 3. Scale Numerics
    if scaler is None:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        df[num_cols] = scaler.transform(df[num_cols])

    # 4. Save
    res = {
        'data': df,
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'scaler': scaler,
        'encoders': encoders
    }
    
    # Save a simplified CSV for reference, but we use the dict/joblib for training
    df.to_csv(output_path.replace('.joblib', '.csv'), index=False)
    joblib.dump(res, output_path)
    print(f"Saved to {output_path}")
    return res

if __name__ == '__main__':
    train_res = preprocess_instances('data/raw/Coderush-26-ML-Train.csv', 'data/processed/train_instances.joblib')
    preprocess_instances('data/raw/Coderush-26-ML-test.csv', 'data/processed/test_instances.joblib', 
                         is_train=False, scaler=train_res['scaler'], encoders=train_res['encoders'])
