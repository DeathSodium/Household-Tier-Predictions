import pandas as pd
import numpy as np
import os
import joblib

def generate_submission(test_path, model_dir, output_path):
    print("Loading aggregated test data...")
    df = pd.read_csv(test_path)
    
    # Load models and encoders
    models = joblib.load(os.path.join(model_dir, "lgb_models.joblib"))
    encoders = joblib.load(os.path.join(model_dir, "label_encoders.joblib"))
    
    # Identify categorical columns
    cat_features = [col for col in df.columns if col.endswith('_mode')]
    
    # Apply encoders to test data
    for col in cat_features:
        le = encoders[col]
        df[col] = df[col].astype(str)
        
        # Handle unseen labels by mapping them to the first seen label (or a 'missing' category if it existed)
        # Here we just use a simple mapping for simplicity, but in production we'd be more careful.
        # Check if label in classes, if not, use the first class
        df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        df[col] = le.transform(df[col])
        
    X_test = df.drop(['bag_id'], axis=1)
    
    # Averaging predictions from all 5 models (using soft voting if desired, or hard voting)
    # We'll use hard voting for simplicity in the final labels
    all_preds = []
    for model in models:
        preds = model.predict(X_test)
        all_preds.append(preds)
        
    # Majority vote
    final_preds = pd.DataFrame(all_preds).mode().iloc[0].astype(int)
    
    # Prepare submission
    submission = pd.DataFrame({
        'bag_id': df['bag_id'],
        'label': final_preds
    })
    
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

if __name__ == "__main__":
    test_aggregated_path = r"data/processed/test_aggregated.csv"
    models_directory = r"models"
    submission_output = r"data/processed/submission.csv"
    
    if os.path.exists(test_aggregated_path):
        generate_submission(test_aggregated_path, models_directory, submission_output)
    else:
        print(f"Aggregated test data not found at {test_aggregated_path}")
