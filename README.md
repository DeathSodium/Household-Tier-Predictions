# Household-Tier-Predictions: Topology-Aware Economic Stratification Engine

**Household-Tier-Predictions** is an advanced Multi-Instance Learning (MIL) framework designed to decode complex household economic hierarchies. By analyzing structural demographic patterns rather than simple averages, the engine identifies the hidden signatures of wealth and stability in large-scale census data.

## 🚀 Key Features

### 1. Topology-Aware Feature Engineering
*   **Income Diversity Entropy**: Utilizes Shannon Entropy to measure the diversity of a household's occupation and workclass streams, identifying resilient, multi-income Upper-tier structures.
*   **Multiplicative Interaction Mapping**: Explicitly models high-leverage wealth signatures like `Capital Gain x Working Hours` and `Education Tier x Occupation` to capture non-linear economic signals.
*   **Structural Demographics**: Captures generational variance and household size dependencies, exploiting the "Demographic Paradox" where higher-tier households exhibit distinct size and age distributions.

### 2. Multi-Instance Learning (MIL) Architecture
*   **Intelligent Bag-Level Aggregation**: Transforms granular individual-level data into coherent household signatures without destroying the family unit's structural integrity.
*   **Probabilistic Threshold Optimization**: Dynamically optimizes decision boundaries to maintain high precision on minority classes (Upper Tier) within imbalanced distributions.

### 3. Defense-Grade Validation & Regularization
*   **Leakage-Proof Validation**: Employs `StratifiedGroupKFold` strictly partitioned by `bag_id` to ensure family members are never split between training and validation folds, preventing "cheating" through intra-household memorization.
*   **Structural Constraints**: Implements high-penalty regularization (`min_data_in_leaf=100`, `lambda_l1=2.0`) to physically prevent the decision trees from "fingerprinting" specific families, guaranteeing extreme stability on unseen data.

## 🛠️ Technology Stack
*   **Modeling Engine**: LightGBM (Gradient Boosting Machine)
*   **Data Engineering**: Pandas, NumPy, Scipy (Entropy & Topology calculation)
*   **Evaluation Framework**: Scikit-Learn (Macro F1 optimization)
*   **Visualization Suite**: Matplotlib, Seaborn (Styled with "Scientific Nobility" aesthetics)

## 📋 Installation & Usage

### Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn lightgbm matplotlib seaborn scipy
   ```
3. Run the primary modeling engine:
   ```bash
   python v2_lgbm_v3feats_notebook.py
   ```

## 🛡️ The Problem-Solver Logic

### The Demographic Paradox
Traditional models fail because they ignore family size. We identified that Upper-tier households are significantly larger (more members per household) than Lower-tier households. Our engine exploits this structural reality to improve prediction accuracy across the entire stratification spectrum.

### Generalization over Noise-Chasing
We rejected automated hyperparameter tuning (like Optuna) in favor of **Domain-Informed Structural Regularization**. By manually enforcing a minimum of 100 households per leaf, we ensured the model learned broad, universal economic rules rather than memorizing noisy individual training rows.

### Balanced Sensitivity
Optimized strictly for **Macro F1-Score**. This forces the engine to be just as sensitive to the 10% minority (Upper Class) as it is to the 58% majority (Lower Class), preventing the "Majority Class Bias" that plagues standard classification models.

---
*Developed by Team "The Periodics" for the CodeRush ML Competition.*
