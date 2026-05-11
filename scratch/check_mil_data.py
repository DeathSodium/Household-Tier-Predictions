import joblib
import pandas as pd

data = joblib.load('data/processed/train_instances.joblib')
df = data['data']
print("Columns:", df.columns.tolist())
print("\nFirst 10 rows:\n", df.head(10))

# Check label distribution per bag
bag_info = df.groupby('bag_id')['label'].first().value_counts()
print("\nBag Label Distribution:\n", bag_info)

# Check if any feature is suspiciously correlated with label
for col in data['num_cols'] + data['cat_cols']:
    corr = df[col].corr(df['label'].map({'lower':0, 'middle':1, 'upper':2}))
    print(f"Corr {col}: {corr:.4f}")
