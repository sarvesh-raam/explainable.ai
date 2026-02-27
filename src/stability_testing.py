import pandas as pd
import numpy as np
import os
import pickle
import shap
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

def check_stability():
    # 1. Load data and models
    models_dir = 'results/models'
    processed_dir = 'data/processed'
    plots_dir = 'results/plots/stability'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load test data and model
    X_test = pd.read_csv(os.path.join(processed_dir, 'X_test_scaled.csv'))
    with open(os.path.join(models_dir, 'Random_Forest.pkl'), 'rb') as f:
        rf_model = pickle.load(f)
        
    print("Starting Stability Testing (Phase 5)...")

    # 2. Select a base instance (Instance 0)
    base_instance = X_test.iloc[0].values.reshape(1, -1)
    
    # Setup SHAP explainer
    explainer = shap.TreeExplainer(rf_model)
    
    def get_shap_importance(instance):
        vals = explainer.shap_values(instance, check_additivity=False)
        # Class 1 (Positive) importance
        if isinstance(vals, list):
            return vals[1].flatten()
        return vals.flatten()

    original_importance = get_shap_importance(base_instance)
    
    # 3. Add Noise and Test Stability
    # We'll test different levels of Gaussian noise
    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5]
    stability_scores = []

    print("\nApplying perturbations and measuring explanation correlation...")
    
    for noise in noise_levels:
        correlations = []
        # Run multiple trials per noise level for better statistics
        for _ in range(20):
            # Create a noisy version of the same patient
            perturbed_instance = base_instance + np.random.normal(0, noise, base_instance.shape)
            
            # Get explanation for noisy patient
            perturbed_importance = get_shap_importance(perturbed_instance)
            
            # Measure similarity using Spearman Correlation (Rank stability)
            corr, _ = spearmanr(original_importance, perturbed_importance)
            correlations.append(corr)
            
        avg_corr = np.mean(correlations)
        stability_scores.append(avg_corr)
        print(f"Noise Level {noise}: Median Rank Correlation = {avg_corr:.4f}")

    # 4. Generate Stability Plot
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, stability_scores, marker='o', linestyle='-', color='teal', linewidth=2)
    plt.fill_between(noise_levels, np.array(stability_scores) - 0.05, np.array(stability_scores) + 0.05, alpha=0.1, color='teal')
    
    plt.title("Explanation Stability Analysis (SHAP + Random Forest)")
    plt.xlabel("Perturbation Level (Gaussian Noise Std Dev)")
    plt.ylabel("Rank Correlation with Original Explanation")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.1)
    
    plt.savefig(os.path.join(plots_dir, 'shap_stability_test.png'), bbox_inches='tight')
    plt.close()

    print(f"\nPhase 5 Complete! Stability results saved in: {plots_dir}")

if __name__ == "__main__":
    check_stability()
