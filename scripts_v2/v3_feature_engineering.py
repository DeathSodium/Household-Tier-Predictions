"""
V3 Feature Engineering — Adds interaction features on top of v2's clean base.
Goal: Push past 0.70 OOF while keeping features interpretable.
"""
import pandas as pd
import numpy as np
import os
from scipy.stats import entropy

def v3_feature_engineering(input_path, output_path):
    print(f"[V3] Feature engineering: {input_path}")
    df = pd.read_csv(input_path)

    # === Individual-level derivations ===
    df['age'] = df['survey_year'] - df['year_of_birth']
    df['is_married'] = (df['marital_status'] == 'Married-civ-spouse').astype(int)
    df['is_high_edu'] = (df['education_num'] >= 13).astype(int)  # Bachelors+
    df['is_professional'] = df['occupation'].isin(['Prof-specialty', 'Exec-managerial']).astype(int)
    df['has_capital'] = (df['capital_gain'] > 0).astype(int)
    df['works_long'] = (df['hours_per_week'] > 40).astype(int)
    df['is_private'] = (df['workclass'] == 'Private').astype(int)

    # === Aggregation ===
    agg = {}

    # 1. Core Numeric — mean, std, median
    for col in ['age', 'education_num', 'hours_per_week', 'net_capital_asset', 'annual_hours_est']:
        agg[col] = ['mean', 'std', 'median']

    # 2. Capital
    agg['capital_gain'] = ['mean', 'std']
    agg['capital_loss'] = ['mean', 'std']
    agg['capital_activity_flag'] = ['mean']

    # 3. Binary indicator rates (NEW: bag-level rates of individual indicators)
    for col in ['is_married', 'is_high_edu', 'is_professional', 'has_capital', 'works_long', 'is_private']:
        agg[col] = ['mean']  # proportion of people in bag with this trait

    # 4. Bag size
    agg['bag_id'] = ['count']

    # Handle label
    has_label = 'label' in df.columns
    if has_label:
        label_map = {'lower': 0, 'middle': 1, 'upper': 2}
        df['label_enc'] = df['label'].map(label_map)
        agg['label_enc'] = ['first']

    df_agg = df.groupby('bag_id').agg(agg)
    df_agg.columns = ['_'.join(c).strip() for c in df_agg.columns]

    renames = {'bag_id_count': 'bag_size'}
    if has_label:
        renames['label_enc_first'] = 'label'
    df_agg.rename(columns=renames, inplace=True)

    # 5. Education Tier proportions
    for tier in ['Primary', 'Secondary', 'Higher']:
        df_agg[f'edu_tier_{tier}_pct'] = df.groupby('bag_id')['education_tier'].apply(
            lambda x: (x == tier).mean()
        )

    # 6. Marital Status proportions (top 4)
    for val in ['Married-civ-spouse', 'Never-married', 'Divorced', 'Widowed']:
        safe = val.replace('-', '_').replace(' ', '_')
        df_agg[f'marital_{safe}_pct'] = df.groupby('bag_id')['marital_status'].apply(
            lambda x, v=val: (x == v).mean()
        )

    # 7. Occupation proportions (top 5)
    for val in ['Prof-specialty', 'Exec-managerial', 'Craft-repair', 'Sales', 'Adm-clerical']:
        safe = val.replace('-', '_').replace(' ', '_')
        df_agg[f'occ_{safe}_pct'] = df.groupby('bag_id')['occupation'].apply(
            lambda x, v=val: (x == v).mean()
        )

    # 8. Relationship proportions (top 4)
    for val in ['Husband', 'Not-in-family', 'Own-child', 'Wife']:
        safe = val.replace('-', '_').replace(' ', '_')
        df_agg[f'rel_{safe}_pct'] = df.groupby('bag_id')['relationship'].apply(
            lambda x, v=val: (x == v).mean()
        )

    # 9. Diversity — entropy
    for col in ['education', 'occupation', 'marital_status', 'workclass']:
        df_agg[f'{col}_entropy'] = df.groupby('bag_id')[col].apply(
            lambda x: entropy(x.value_counts(normalize=True))
        )

    # 10. Demographics
    df_agg['sex_Male_pct'] = df.groupby('bag_id')['sex'].apply(lambda x: (x == 'Male').mean())
    df_agg['race_White_pct'] = df.groupby('bag_id')['race'].apply(lambda x: (x == 'White').mean())
    df_agg['race_Black_pct'] = df.groupby('bag_id')['race'].apply(lambda x: (x == 'Black').mean())

    # 11. IQR
    for col in ['age', 'education_num', 'hours_per_week']:
        q25 = df.groupby('bag_id')[col].quantile(0.25)
        q75 = df.groupby('bag_id')[col].quantile(0.75)
        df_agg[f'{col}_iqr'] = q75 - q25

    # === 12. INTERACTION FEATURES (NEW in v3) ===
    # These capture cross-feature relationships at the bag level
    df_agg['edu_x_hours'] = df_agg['education_num_mean'] * df_agg['hours_per_week_mean']
    df_agg['edu_x_age'] = df_agg['education_num_mean'] / (df_agg['age_mean'] + 1)
    df_agg['married_x_highedu'] = df_agg['is_married_mean'] * df_agg['is_high_edu_mean']
    df_agg['professional_x_highedu'] = df_agg['is_professional_mean'] * df_agg['is_high_edu_mean']
    df_agg['capital_x_hours'] = df_agg['capital_gain_mean'] * df_agg['hours_per_week_mean']
    df_agg['higher_edu_x_professional'] = df_agg['edu_tier_Higher_pct'] * df_agg['is_professional_mean']
    df_agg['married_x_works_long'] = df_agg['is_married_mean'] * df_agg['works_long_mean']

    # === Final cleanup ===
    df_agg.fillna(0, inplace=True)
    df_agg = df_agg.reset_index()

    df_agg.to_csv(output_path, index=False)
    feature_count = len([c for c in df_agg.columns if c not in ['bag_id', 'label']])
    print(f"[V3] Saved to {output_path} | Shape: {df_agg.shape} | Features: {feature_count}")
    return df_agg

if __name__ == '__main__':
    v3_feature_engineering('data/raw/Coderush-26-ML-Train.csv', 'data/processed/train_v3.csv')
    v3_feature_engineering('data/raw/Coderush-26-ML-test.csv', 'data/processed/test_v3.csv')
