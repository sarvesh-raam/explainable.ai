import pandas as pd
import numpy as np
import os
import pickle
from lime import lime_tabular
import matplotlib.pyplot as plt

def run_lime_analysis():
    # 1. Load data and models
    models_dir = 'results/models'
    processed_dir = 'data/processed'
    plots_dir = 'results/plots/lime'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load test data and full dataset for feature names and statistics
    X_test = pd.read_csv(os.path.join(processed_dir, 'X_test_scaled.csv'))
    
    # Load models
    with open(os.path.join(models_dir, 'Random_Forest.pkl'), 'rb') as f:
        rf_model = pickle.load(f)
    
    print("Models loaded. Starting LIME analysis...")

    # 2. Initialize LIME Explainer
    # LIME needs the training data distribution to perturb points correctly
    # Since we saved scaled data, we'll use X_test as a reference for the explainer
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_test),
        feature_names=X_test.columns.tolist(),
        class_names=['Low Risk', 'High Risk'],
        mode='classification'
    )

    # 3. Explain the first instance (same as we did in SHAP)
    print("\n[1/1] Generating Local LIME Explanation for Instance 0...")
    instance_idx = 0
    
    # Get the explanation specifically for the 'Positive' class (label 1)
    exp = explainer.explain_instance(
        data_row=X_test.iloc[instance_idx].values, # Use values for cleaner indexing
        predict_fn=rf_model.predict_proba,
        labels=(1,)
    )

    # 4. Save the explanation as a plot
    plt.figure()
    fig = exp.as_pyplot_figure(label=1) # Explicitly plot label 1
    plt.title(f"LIME Local Explanation (Instance {instance_idx})")
    plt.savefig(os.path.join(plots_dir, f'rf_lime_instance_{instance_idx}.png'), bbox_inches='tight')
    plt.close()

    print(f"\nPhase 4 LIME Plots saved successfully in: {plots_dir}")

if __name__ == "__main__":
    run_lime_analysis()
