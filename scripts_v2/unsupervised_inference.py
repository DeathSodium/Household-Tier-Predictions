"""
Unsupervised Inference using K-Means Clustering on the Test Dataset.
Maps the 3 resulting clusters to 'lower', 'middle', 'upper' based on economic indicators.
"""
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def run_unsupervised_inference():
    print("="*60)
    print("UNSUPERVISED CLUSTERING INFERENCE")
    print("="*60)

    # 1. Load Data
    test_path = 'data/processed/test_v2.csv'
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at {test_path}")
        
    df_test = pd.read_csv(test_path)
    bag_ids = df_test['bag_id'].values
    
    # Exclude non-features and perfectly correlated redundant features
    cols_to_drop = ['bag_id', 'label']
    # Drop annual_hours_est aggregates because they are 1.0 correlated with hours_per_week
    for col in df_test.columns:
        if 'annual_hours_est' in col:
            cols_to_drop.append(col)
            
    features = [c for c in df_test.columns if c not in cols_to_drop]
    X_test = df_test[features].copy()
    
    print(f"Loaded test data: {X_test.shape[0]} bags, {X_test.shape[1]} features.")

    # 2. Scale Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_test)

    # 3. K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(X_scaled)
    
    # 4. Analyze Centroids to Map Labels
    # We invert the scaler to see the actual means of the clusters
    centroids_orig = scaler.inverse_transform(kmeans.cluster_centers_)
    df_centroids = pd.DataFrame(centroids_orig, columns=features)
    
    # We will use 'education_num_mean' and 'capital_gain_mean' to determine the economic hierarchy
    # The cluster with the lowest education/capital is 'lower', highest is 'upper'.
    
    # Calculate a simple "wealth score" for each cluster centroid
    # Standardizing these two indicators across the 3 centroids so we can add them equally
    edu_scores = (df_centroids['education_num_mean'] - df_centroids['education_num_mean'].min()) / (df_centroids['education_num_mean'].max() - df_centroids['education_num_mean'].min())
    cap_scores = (df_centroids['capital_gain_mean'] - df_centroids['capital_gain_mean'].min()) / (df_centroids['capital_gain_mean'].max() - df_centroids['capital_gain_mean'].min())
    
    df_centroids['wealth_score'] = edu_scores + cap_scores
    
    # Sort clusters by wealth score
    sorted_clusters = df_centroids.sort_values('wealth_score').index.tolist()
    
    # Mapping
    cluster_to_label = {
        sorted_clusters[0]: 'lower',
        sorted_clusters[1]: 'middle',
        sorted_clusters[2]: 'upper'
    }
    
    # numeric labels for kaggle: lower:0, middle:1, upper:2
    label_to_numeric = {'lower': 0, 'middle': 1, 'upper': 2}
    
    print("\nCluster Mapping Logic:")
    for cid in range(3):
        mapped = cluster_to_label[cid]
        edu = df_centroids.loc[cid, 'education_num_mean']
        cap = df_centroids.loc[cid, 'capital_gain_mean']
        score = df_centroids.loc[cid, 'wealth_score']
        print(f"Cluster {cid} -> {mapped.upper()} | Edu: {edu:.2f}, CapGain: {cap:.2f}, WealthScore: {score:.2f}")

    # 5. Generate Submission
    final_labels = [label_to_numeric[cluster_to_label[cid]] for cid in cluster_ids]
    
    sub = pd.DataFrame({
        'bag_id': bag_ids,
        'label': final_labels
    })
    
    out_dir = 'submissions/unsupervised_kmeans'
    os.makedirs(out_dir, exist_ok=True)
    sub_path = os.path.join(out_dir, 'submission.csv')
    sub.to_csv(sub_path, index=False)
    print(f"\nSubmission saved to {sub_path}")
    
    # 6. Generate Report
    report_path = os.path.join(out_dir, 'UNSUPERVISED_REPORT.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Unsupervised Inference Report\n\n")
        f.write("## Methodology\n")
        f.write("- **Algorithm:** K-Means Clustering (`n_clusters=3`)\n")
        f.write(f"- **Features Used:** {X_test.shape[1]} scaled bag-level features (redundant features like annual_hours_est dropped).\n")
        f.write("- **Mapping Logic:** Clusters mapped to `lower`, `middle`, `upper` based on their centroid's `education_num_mean` and `capital_gain_mean`.\n\n")
        
        f.write("## Cluster Mapping\n")
        for cid in range(3):
            mapped = cluster_to_label[cid]
            f.write(f"- **Cluster {cid}** -> **{mapped.upper()}** (Education: {df_centroids.loc[cid, 'education_num_mean']:.2f}, Capital Gain: {df_centroids.loc[cid, 'capital_gain_mean']:.2f})\n")
        
        f.write("\n## Label Distribution\n")
        dist = pd.Series([cluster_to_label[cid] for cid in cluster_ids]).value_counts()
        f.write("```\n")
        f.write(dist.to_string())
        f.write("\n```\n")
        
    print(f"Report saved to {report_path}")

if __name__ == '__main__':
    run_unsupervised_inference()
