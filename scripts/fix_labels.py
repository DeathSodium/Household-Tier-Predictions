"""Fix all existing submissions: convert integer labels (0,1,2) to strings (lower,middle,upper)"""
import pandas as pd, os, glob

decode = {0: 'lower', 1: 'middle', 2: 'upper'}

paths = glob.glob('submissions/**/*.csv', recursive=True) + glob.glob('submission_*.csv')
for path in paths:
    try:
        sub = pd.read_csv(path)
        if 'label' not in sub.columns:
            continue
        # Only fix if labels are integers
        if sub['label'].dtype in ['int64', 'int32', 'int']:
            sub['label'] = sub['label'].map(decode)
            sub.to_csv(path, index=False)
            print(f"Fixed: {path}")
            print(f"  -> labels now: {sorted(sub['label'].unique())}")
        else:
            print(f"OK (already strings): {path} -> {sorted(sub['label'].unique())}")
    except Exception as e:
        print(f"Skip {path}: {e}")
