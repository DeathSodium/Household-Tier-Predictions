"""
V2 Feature Engineering — Clean, stable, 47-feature bag-level aggregation.
Uses ALL 29 raw columns. No kurtosis/skewness on small bags.
"""
import pandas as pd
import numpy as np
import os
from scipy.stats import entropy

def v2_feature_engineering(input_path, output_path):
    print(f"[V2] Feature engineering: {input_path}")
    df = pd.read_csv(input_path)

    # === Individual-level derivations ===
    df['age'] = df['survey_year'] - df['year_of_birth']

    # === Aggregation dictionary ===
    agg = {}

    # 1. Core Numeric Stats — mean, std, median ONLY (stable on 3-7 members)
    for col in ['age', 'education_num', 'hours_per_week', 'net_capital_asset', 'annual_hours_est']:
        agg[col] = ['mean', 'std', 'median']

    # 2. Capital features
    agg['capital_gain'] = ['mean', 'std']
    agg['capital_loss'] = ['mean', 'std']
    agg['capital_activity_flag'] = ['mean']  # rate of capital activity in bag

    # 3. Bag size
    agg['bag_id'] = ['count']

    # Handle label
    has_label = 'label' in df.columns
    if has_label:
        label_map = {'lower': 0, 'middle': 1, 'upper': 2}
        df['label_enc'] = df['label'].map(label_map)
        agg['label_enc'] = ['first']

    # Run aggregation
    df_agg = df.groupby('bag_id').agg(agg)
    df_agg.columns = ['_'.join(c).strip() for c in df_agg.columns]

    # Rename
    renames = {'bag_id_count': 'bag_size'}
    if has_label:
        renames['label_enc_first'] = 'label'
    df_agg.rename(columns=renames, inplace=True)

    # 4. Education Tier proportions (3 clean categories)
    for tier in ['Primary', 'Secondary', 'Higher']:
        df_agg[f'edu_tier_{tier}_pct'] = df.groupby('bag_id')['education_tier'].apply(
            lambda x: (x == tier).mean()
        )

    # 5. Marital Status proportions (top 4)
    for val in ['Married-civ-spouse', 'Never-married', 'Divorced', 'Widowed']:
        safe = val.replace('-', '_').replace(' ', '_')
        df_agg[f'marital_{safe}_pct'] = df.groupby('bag_id')['marital_status'].apply(
            lambda x, v=val: (x == v).mean()
        )

    # 6. Occupation proportions (top 5)
    for val in ['Prof-specialty', 'Exec-managerial', 'Craft-repair', 'Sales', 'Adm-clerical']:
        safe = val.replace('-', '_').replace(' ', '_')
        df_agg[f'occ_{safe}_pct'] = df.groupby('bag_id')['occupation'].apply(
            lambda x, v=val: (x == v).mean()
        )

    # 7. Relationship proportions (top 4)
    for val in ['Husband', 'Not-in-family', 'Own-child', 'Wife']:
        safe = val.replace('-', '_').replace(' ', '_')
        df_agg[f'rel_{safe}_pct'] = df.groupby('bag_id')['relationship'].apply(
            lambda x, v=val: (x == v).mean()
        )

    # 8. Diversity — entropy for 4 key categoricals
    for col in ['education', 'occupation', 'marital_status', 'workclass']:
        df_agg[f'{col}_entropy'] = df.groupby('bag_id')[col].apply(
            lambda x: entropy(x.value_counts(normalize=True))
        )

    # 9. Demographics
    df_agg['sex_Male_pct'] = df.groupby('bag_id')['sex'].apply(lambda x: (x == 'Male').mean())
    df_agg['race_White_pct'] = df.groupby('bag_id')['race'].apply(lambda x: (x == 'White').mean())
    df_agg['race_Black_pct'] = df.groupby('bag_id')['race'].apply(lambda x: (x == 'Black').mean())

    # 10. Percentiles — IQR only for stable core numerics
    for col in ['age', 'education_num', 'hours_per_week']:
        q25 = df.groupby('bag_id')[col].quantile(0.25)
        q75 = df.groupby('bag_id')[col].quantile(0.75)
        df_agg[f'{col}_iqr'] = q75 - q25

    # === Final cleanup ===
    df_agg.fillna(0, inplace=True)
    df_agg = df_agg.reset_index()

    df_agg.to_csv(output_path, index=False)
    feature_count = len([c for c in df_agg.columns if c not in ['bag_id', 'label']])
    print(f"[V2] Saved to {output_path} | Shape: {df_agg.shape} | Features: {feature_count}")
    return df_agg

if __name__ == '__main__':
    v2_feature_engineering('data/raw/Coderush-26-ML-Train.csv', 'data/processed/train_v2.csv')
    v2_feature_engineering('data/raw/Coderush-26-ML-test.csv', 'data/processed/test_v2.csv')
