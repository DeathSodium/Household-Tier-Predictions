import pandas as pd
from sklearn.model_selection import train_test_split
import os

def create_dataset_splits(input_path, output_dir):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # 1. First split: Separate 20% for Final Hold-out Test
    # 80% (train+val), 20% (test)
    train_val_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=42
    )
    
    # 2. Second split: Separate 25% of the remaining 80% for Validation
    # (0.25 * 0.8 = 0.20 of the original total)
    # 60% (train), 20% (val)
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.25, stratify=train_val_df['label'], random_state=42
    )
    
    print(f"Original shape: {df.shape}")
    print(f"Train shape: {train_df.shape} (60%)")
    print(f"Validation shape: {val_df.shape} (20%)")
    print(f"Hold-out Test shape: {test_df.shape} (20%)")
    
    # Save splits
    train_df.to_csv(os.path.join(output_dir, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val_split.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_split.csv"), index=False)
    print(f"Splits saved to {output_dir}")

if __name__ == "__main__":
    aggregated_path = r"data/processed/train_ultimate.csv"
    split_dir = r"data/processed/split"
    
    if os.path.exists(aggregated_path):
        create_dataset_splits(aggregated_path, split_dir)
    else:
        print(f"Aggregated data not found.")
