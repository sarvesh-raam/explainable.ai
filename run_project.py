import sys
import os

# Ensure the 'src' directory is in the path so we can import our scripts
sys.path.append(os.path.join(os.getcwd(), 'src'))

print("====================================================")
print("   EXPLAINABLE AI (XAI) RESEARCH FRAMEWORK")
print("====================================================")

try:
    print("\n[PHASE 1] Cleaning Data...")
    from preprocessing import main as run_p1
    run_p1()

    print("\n[PHASE 2] Training Benchmark Models...")
    from model_training import train_models as run_p2
    run_p2()

    print("\n[PHASE 3] Generating SHAP Explanations...")
    from shap_analysis import run_shap_analysis as run_p3
    run_p3()

    print("\n[PHASE 4] Generating LIME Explanations...")
    from lime_analysis import run_lime_analysis as run_p4
    run_p4()

    print("\n[PHASE 5] Stress-Testing Stability...")
    from stability_testing import check_stability as run_p5
    run_p5()

    print("\n" + "="*52)
    print("      SUCCESS: ALL RESEARCH PHASES COMPLETED!")
    print("="*52)
    print("\nYou can now view all results in the 'results/' folder:")
    print(" - Benchmarks: results/models/")
    print(" - XAI Visuals: results/plots/shap/ and results/plots/lime/")
    print(" - Stability Proof: results/plots/stability/")
    print("="*52)

except Exception as e:
    print(f"\n[ERROR] An error occurred during the pipeline: {e}")
