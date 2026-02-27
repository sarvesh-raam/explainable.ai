import pandas as pd
import numpy as np
import os
import pickle
import shap
import matplotlib.pyplot as plt

def run_shap_analysis():
    # 1. Load data and models
    models_dir = 'results/models'
    processed_dir = 'data/processed'
    plots_dir = 'results/plots/shap'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load test data
    X_test = pd.read_csv(os.path.join(processed_dir, 'X_test_scaled.csv'))
    
    # Load models
    with open(os.path.join(models_dir, 'Random_Forest.pkl'), 'rb') as f:
        rf_model = pickle.load(f)
    
    with open(os.path.join(models_dir, 'Logistic_Regression.pkl'), 'rb') as f:
        lr_model = pickle.load(f)

    print("Models loaded. Starting SHAP analysis...")

    # --- SECTION A: Random Forest (Global) ---
    print("\n[1/3] Analyzing Random Forest (Global)...")
    # Using the modern Explainer API for more robust plotting
    explainer_rf = shap.Explainer(rf_model, X_test)
    shap_values_rf = explainer_rf(X_test, check_additivity=False)

    # Summary Plot (Global Importance) - using Beeswarm
    plt.figure(figsize=(10, 6))
    # For binary classification, we use class 1 (Positive)
    if len(shap_values_rf.shape) == 3: # (N, features, 2)
        shap_vals_to_plot = shap_values_rf[:, :, 1]
    else:
        shap_vals_to_plot = shap_values_rf

    shap.plots.beeswarm(shap_vals_to_plot, show=False)
    plt.title("SHAP Global Feature Importance (Random Forest)")
    plt.savefig(os.path.join(plots_dir, 'rf_summary_plot.png'), bbox_inches='tight')
    plt.close()

    # Waterfall Plot (Local Instance - First Patient in Test Set)
    print("[2/3] Generating Local Explanation (Waterfall Plot)...")
    # We need to use the explainer's .explanation object for Waterfall plots in newer SHAP versions
    # For RF, we calculate again via the newer API for convenience
    explainer_new = shap.Explainer(rf_model, X_test)
    shap_values_local = explainer_new(X_test, check_additivity=False)
    
    # For binary classification, we extract only the SHAP values for class 1 (Positive)
    # The shape is (samples, features, outputs)
    if len(shap_values_local.shape) == 3: # (N, features, 2)
        local_explanation = shap_values_local[0, :, 1]
    else:
        local_explanation = shap_values_local[0]
    
    plt.figure()
    shap.plots.waterfall(local_explanation, show=False)
    plt.savefig(os.path.join(plots_dir, 'rf_local_waterfall.png'), bbox_inches='tight')
    plt.close()

    # --- SECTION B: Logistic Regression (LinearExplainer) ---
    print("[3/3] Analyzing Logistic Regression (Consistency Check)...")
    explainer_lr = shap.LinearExplainer(lr_model, X_test)
    shap_values_lr = explainer_lr.shap_values(X_test)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_lr, X_test, show=False)
    plt.title("SHAP Feature Importance (Logistic Regression)")
    plt.savefig(os.path.join(plots_dir, 'lr_summary_plot.png'), bbox_inches='tight')
    plt.close()

    print(f"\nPhase 3 Plots saved successfully in: {plots_dir}")

if __name__ == "__main__":
    run_shap_analysis()
