import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys

warnings.filterwarnings('ignore')

# Import the feature engineering logic
sys.path.append('.')
from v2_lgbm_v3feats_notebook import build_features

# =============================================================================
# AESTHETIC CONFIGURATION ("The Periodics")
# =============================================================================
BG_COLOR = '#030303'
SURFACE_COLOR = '#0a0a0a'
GOLD = '#D4AF37'
CYAN = '#00d2ff'
PURPLE = '#b026ff'
TEXT_MAIN = '#e0e0e0'
TEXT_MUTED = '#888888'

plt.rcParams.update({
    'figure.facecolor': BG_COLOR,
    'axes.facecolor': SURFACE_COLOR,
    'axes.edgecolor': GOLD,
    'axes.labelcolor': TEXT_MAIN,
    'text.color': TEXT_MAIN,
    'xtick.color': TEXT_MUTED,
    'ytick.color': TEXT_MUTED,
    'grid.color': '#222222',
    'font.family': 'sans-serif',
    'font.sans-serif': ['IBM Plex Sans', 'Segoe UI', 'Arial']
})

# Custom Colormap for Correlation
from matplotlib.colors import LinearSegmentedColormap
colors = [CYAN, BG_COLOR, GOLD]
cmap = LinearSegmentedColormap.from_list("periodics", colors)

# =============================================================================
# DATA LOADING
# =============================================================================
def find_train_file():
    paths = [
        os.path.join('Dataset', 'Coderush-26-ML-Train.csv'),
        os.path.join('data', 'raw', 'Coderush-26-ML-Train.csv')
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Could not find training data.")

TRAIN_PATH = find_train_file()
df_raw = pd.read_csv(TRAIN_PATH)
print(f"Loaded training data from: {TRAIN_PATH}")

df_feat = build_features(df_raw, has_label=True)
print(f"Engineered features shape: {df_feat.shape}")

# Directories
OUT_DIR_ORIGINAL = os.path.join('visualizations', 'plots', 'original')
OUT_DIR_FEATURED = os.path.join('visualizations', 'plots', 'featured')
os.makedirs(OUT_DIR_ORIGINAL, exist_ok=True)
os.makedirs(OUT_DIR_FEATURED, exist_ok=True)

# =============================================================================
# PLOTTING ROUTINES
# =============================================================================

def save_plot(fig, directory, filename):
    filepath = os.path.join(directory, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor=BG_COLOR, edgecolor='none')
    print(f"Saved: {filepath}")
    plt.close(fig)

# 1. Original Dataset Correlation
print("Generating Original Correlation Matrix...")
numeric_raw = df_raw.select_dtypes(include=[np.number]).drop(columns=['bag_id'], errors='ignore')
corr_raw = numeric_raw.corr()

fig1, ax1 = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_raw, cmap=cmap, center=0, annot=True, fmt=".2f",
            cbar_kws={'shrink': 0.8}, ax=ax1, 
            linecolor=BG_COLOR, linewidths=0.5)

ax1.set_title("Original Dataset Correlation Matrix", fontsize=18, color=GOLD, pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
save_plot(fig1, OUT_DIR_ORIGINAL, '06_original_correlation.png')

# 2. Featured Dataset Correlation
print("Generating Featured Dataset Correlation Matrix...")
# Select numeric features. We drop bag_id for the correlation
numeric_feat = df_feat.select_dtypes(include=[np.number]).drop(columns=['bag_id'], errors='ignore')

# Because 57 features is too many for a clean heatmap with annotations,
# we will generate a large, unannotated heatmap to show the macro structure.
corr_feat = numeric_feat.corr()

fig2, ax2 = plt.subplots(figsize=(24, 20))
sns.heatmap(corr_feat, cmap=cmap, center=0, annot=False, 
            cbar_kws={'shrink': 0.8}, ax=ax2, 
            linecolor=BG_COLOR, linewidths=0.1)

ax2.set_title("Featured (MIL Topology) Dataset Correlation Matrix", fontsize=24, color=GOLD, pad=30)
plt.xticks(rotation=90, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
save_plot(fig2, OUT_DIR_FEATURED, '07_featured_correlation.png')

print("All correlation visualizations generated successfully!")
