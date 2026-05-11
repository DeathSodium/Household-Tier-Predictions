"""
Unsupervised Learning Evaluation on Training Data.
Drops the labels, performs K-Means, maps clusters using economic logic, 
and then evaluates the generated predictions against the true labels.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, f1_score
import os

def evaluate_unsupervised():
    print("="*60)
    print("UNSUPERVISED CLUSTERING EVALUATION ON TRAINING DATA")
    print("="*60)

    # 1. Load Data
    train_path = 'data/processed/train_v2.csv'
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}")
        
    df_train = pd.read_csv(train_path)
    
    # Extract True Labels
    if 'label' not in df_train.columns:
        raise ValueError("Training data must contain 'label' for evaluation.")
    
    true_labels_str = df_train['label'].values
    true_labels = df_train['label'].values
    label_to_numeric = {'lower': 0, 'middle': 1, 'upper': 2}
    
    # 2. Exclude non-features and redundant features
    cols_to_drop = ['bag_id', 'label']
    for col in df_train.columns:
        if 'annual_hours_est' in col:
            cols_to_drop.append(col)
            
    features = [c for c in df_train.columns if c not in cols_to_drop]
    X_train = df_train[features].copy()
    
    print(f"Loaded train data: {X_train.shape[0]} bags, {X_train.shape[1]} features (Labels dropped).")

    # 3. Scale Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # 4. K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(X_scaled)
    
    # 5. Analyze Centroids to Map Labels
    centroids_orig = scaler.inverse_transform(kmeans.cluster_centers_)
    df_centroids = pd.DataFrame(centroids_orig, columns=features)
    
    # Wealth score calculation
    edu_scores = (df_centroids['education_num_mean'] - df_centroids['education_num_mean'].min()) / (df_centroids['education_num_mean'].max() - df_centroids['education_num_mean'].min())
    cap_scores = (df_centroids['capital_gain_mean'] - df_centroids['capital_gain_mean'].min()) / (df_centroids['capital_gain_mean'].max() - df_centroids['capital_gain_mean'].min())
    
    df_centroids['wealth_score'] = edu_scores + cap_scores
    sorted_clusters = df_centroids.sort_values('wealth_score').index.tolist()
    
    # Mapping logic
    cluster_to_label = {
        sorted_clusters[0]: 'lower',
        sorted_clusters[1]: 'middle',
        sorted_clusters[2]: 'upper'
    }
    
    print("\nCluster Mapping Logic (Learned from Training Data without labels):")
    for cid in range(3):
        mapped = cluster_to_label[cid]
        edu = df_centroids.loc[cid, 'education_num_mean']
        cap = df_centroids.loc[cid, 'capital_gain_mean']
        score = df_centroids.loc[cid, 'wealth_score']
        print(f"Cluster {cid} -> {mapped.upper()} | Edu: {edu:.2f}, CapGain: {cap:.2f}, WealthScore: {score:.2f}")

    # 6. Generate Predictions
    pred_labels = np.array([label_to_numeric[cluster_to_label[cid]] for cid in cluster_ids])
    
    # 7. Evaluate
    print("\n" + "="*60)
    print("EVALUATION SCORES")
    print("="*60)
    
    macro_f1 = f1_score(true_labels, pred_labels, average='macro')
    print(classification_report(true_labels, pred_labels, target_names=['lower', 'middle', 'upper']))
    print(f"Overall Macro F1 Score: {macro_f1:.4f}")

if __name__ == '__main__':
    evaluate_unsupervised()
