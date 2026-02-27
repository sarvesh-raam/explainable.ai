# Explainable AI (XAI) in Healthcare

## Project Overview
**Title:** Balancing Accuracy and Interpretability: A Post-hoc Explainable AI Framework for Structured Healthcare Data.

This research project investigates the reliability, stability, and clinical utility of post-hoc explainability methods (**SHAP** and **LIME**) when applied to black-box machine learning models in healthcare.

## Central Research Question
> *"Can post-hoc explainability methods produce faithful, stable, and clinically useful explanations for black-box healthcare models—while maintaining high predictive performance?"*

## Research Objectives
- **Empirical Comparison:** Evaluating the performance of Logistic Regression, Random Forest, and Gradient Boosting.
- **XAI Evaluation:** Comparing SHAP (Global & Local) vs. LIME (Local) explanations.
- **Stability Analysis:** Measuring the robustness of explanations against data perturbations (Gaussian noise).
- **Consistency Check:** Quantifying the agreement between different explanation methods.

## Methodology (The 5-Phase Plan)
1. **Phase 1: Dataset & Preprocessing:** Cleaning and engineering UCI Heart Disease & UCI Adult Income datasets.
2. **Phase 2: Model Training:** Building predictive engines using ensemble and linear methods.
3. **Phase 3: SHAP Analysis:** Global and local feature driver identification.
4. **Phase 4: LIME Analysis:** Local instance analysis and consistency comparisons.
5. **Phase 5: Stability Testing:** Quantifying the trustworthiness of explanations for publication.

## Directory Structure
- `data/`: Raw and processed healthcare datasets.
- `src/`: Python source code for preprocessing, modeling, and XAI.
- `notebooks/`: Experimental analysis and visualizations.
- `results/`: Research plots, tables, and publication-ready findings.

## Tech Stack
- **Language:** Python 3.x
- **Libraries:** scikit-learn, XGBoost, SHAP, LIME, pandas, NumPy, Matplotlib, Seaborn.
