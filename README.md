# Household Economic Tier Prediction

This project implements a high-performance machine learning pipeline to predict household economic stratification using census-derived data.

## The Problem
The primary challenge is to predict the economic tier (**Lower**, **Middle**, or **Upper**) of a household based on the individual data of its members. This task presents several technical hurdles:

1.  **Multi-Instance Learning (MIL)**: The target label is assigned at the "bag" (household) level, but the data is provided at the "instance" (individual) level. Standard aggregation techniques (like simple averaging) often destroy the structural relationship between family members.
2.  **Structural Imbalance**: While Lower-tier households dominate the "bag" counts (58%), Upper-tier households tend to be significantly larger. This creates a "Demographic Paradox" where the individual-level distribution is completely different from the household-level distribution.
3.  **The Overfitting Trap**: In demographic datasets, decision trees easily "fingerprint" or memorize specific families, leading to high training accuracy that fails to generalize to new households on the private leaderboard.

## The Solution
We developed a **Topology-Aware MIL Pipeline** designed for maximum stability and generalization.

*   **Topology-Aware Feature Engineering**: We moved beyond simple statistics to create features that describe the *shape* of the household, such as `occupation_entropy` (income diversity), `capital_x_hours` (leveraged wealth), and `age_variance` (generational structure).
*   **Leakage-Proof Validation**: We implemented `StratifiedGroupKFold` grouped strictly by `bag_id`. This ensures that family members from the same household are never split between training and validation sets, preventing the model from "cheating" via data leakage.
*   **Ultra-Regularized LightGBM**: We applied extreme physical constraints to our model, including `min_data_in_leaf=100` and heavy L1/L2 penalties (`lambda_l1=2.0`). This forces the model to only learn rules that apply to broad groups of households.

## Why This Approach?
*   **MIL Focus**: Wealth is a collective household attribute. Our MIL approach preserves the internal family structure that simple averaging would destroy.
*   **Generalization over Optimization**: We chose manual structural regularization over automated tuning frameworks like Optuna. This was a strategic decision to prevent the model from "chasing noise" in the validation folds and instead force it to learn robust economic patterns.
*   **Macro F1 Optimization**: Since 58% of the data is Tier 1, Accuracy is a deceptive metric. We optimized exclusively for **Macro F1** to ensure the model correctly identifies the rare Tier 3 (Upper) households with high precision and recall.

---
*Developed by Team "The Periodics" for the CodeRush ML Competition.*
