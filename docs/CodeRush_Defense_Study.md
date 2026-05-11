# CodeRush Machine Learning Defense Manual: "The Periodics"

**Project Title:** Household Economic Stratification via Topology-Aware Multi-Instance Learning
**Objective:** Provide an airtight defense of the methodologies, feature engineering choices, and model architectures chosen for the final submission.

---

## 1. The Core Problem: Multi-Instance Learning (MIL)

**What we did:** We framed this competition not as a standard classification problem, but as a **Multi-Instance Learning (MIL)** problem.
**Why we did it:** The dataset provides individual-level data (multiple people per household), but the target label (economic tier) is assigned at the household (`bag_id`) level. 

**Why not standard aggregation?** 
If you naively average the features of a household (e.g., mean income, mean age), you destroy the structural reality of the family. A household with one high-earning tech executive and three unemployed dependents might have the exact same "average income" and "average age" as a household with four minimum-wage workers. Naive aggregation treats them identically. We needed the model to understand the *topology* of the household.

---

## 2. Demographic Discovery: The Individual vs. Bag Paradox

**What we discovered:** 
* At the **Household (Bag) level**, the classes are severely imbalanced: ~58% Lower, ~32% Middle, ~10% Upper.
* At the **Individual level**, the imbalance flips: ~28% Lower, ~40% Middle, ~32% Upper.

**Why this matters for our strategy:**
This proves mathematically that **Upper and Middle-class households have significantly more members per household than Lower-class households.** The physical size and structure of the family is a massive predictor of wealth in this dataset. This insight drove our entire Feature Engineering strategy.

---

## 3. Feature Engineering: Topology over Averages

**What we did:** We created custom "Topology-Aware" features rather than relying solely on `mean()`, `sum()`, or `std()`.

**Key Features & Defense:**
1. **`occupation_entropy` (Shannon Entropy of Occupations)**
   * *Why:* Measures the diversity of income streams. A household where everyone works in the same industry is vulnerable to sector shocks. A household with high entropy has diverse, stable income streams, correlating strongly with the Upper tier.
2. **`edu_x_hours` and `capital_x_hours` (Cross Features)**
   * *Why:* Linear models add features together. Decision trees split on them sequentially. But wealth is multiplicative. By explicitly multiplying `education` by `working hours`, we gave the model a direct mathematical shortcut to identify highly leveraged professionals. `capital_x_hours` identifies the rare households earning passive capital *while* working active labor—the definitive signature of Tier 3.
3. **`age_variance`**
   * *Why:* Captures multi-generational households (e.g., working adults + elderly + children) versus single-generation roommate situations, which have vastly different economic realities.

---

## 4. Pipeline Integrity: Preventing Data Leakage

**What we did:** Implemented strict `StratifiedGroupKFold` and in-loop threshold optimization.

**Why this technique?**
* **Group K-Fold by `bag_id`:** If you use standard cross-validation, a husband might end up in the training fold while his wife ends up in the validation fold. The model will just memorize the husband's exact age/income to cheat on the wife's prediction. Grouping by `bag_id` guarantees the entire household stays together, forcing the model to actually learn patterns, not memorize rows.
* **In-Loop Thresholding:** We optimized the probability cutoffs for F1 maximization *inside* the cross-validation loop. If you optimize thresholds on the whole dataset before evaluating, you bleed future data into the past, causing falsely high CV scores that crash on the private leaderboard.

---

## 5. Model Selection: The Overfitting Trap

**The Evolution of our Models:**

1. **Phase 1: Baseline Model (Random Forest / Simple Aggregation)**
   * *Why:* To establish a floor. It used basic means/modes. It achieved a 0.614 Macro F1 but failed to capture complex interactions.
   
2. **Phase 2: The Trap (Standard LightGBM + Complex Features)**
   * *What happened:* We fed our complex topology features into a standard LightGBM. The local CV score skyrocketed to 0.69+ (Train F1 was 0.95+). 
   * *Why we abandoned it:* A Train-Val gap of 0.30 is catastrophic overfitting. The decision trees were growing too deep and learning the exact, specific fingerprints of individual households rather than generalizable economic rules.

3. **Phase 3: The Best Model (Ultra-Regularized LightGBM)**
   * *What we did:* We applied extreme physical constraints to the model: `lambda_l1=2.0`, `lambda_l2=2.0`, and `min_data_in_leaf=100`.
   * *Why this technique?* By forcing `min_data_in_leaf=100`, we physically prevented the decision tree from making a rule unless it applied to at least 100 households. This completely stopped the model from memorizing specific families. We sacrificed some raw training accuracy to buy massive generalization stability on unseen data, locking in a robust 0.659 Macro F1.

---

## 6. Metric Choice: Why Macro F1?

**What we did:** We completely ignored Accuracy and optimized exclusively for Macro F1.

**Why not Accuracy?**
Because Tier 1 (Lower) makes up 58% of the households. A naive model that simply guesses "Lower" for every single family would achieve 58% accuracy while being completely useless, entirely failing to identify the 10% Upper-class families. Macro F1 calculates the F1 score for each class independently and averages them equally, forcing the model to respect the minority (Upper) class just as much as the majority (Lower) class.
