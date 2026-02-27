import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_models():
    # 1. Load the preprocessed data
    data_path = os.path.join('data', 'processed', 'heart_disease_cleaned.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data not found at {data_path}. Please run preprocessing first.")
    
    df = pd.read_csv(data_path)
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 2. Split into Train/Test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples.")
    
    # 3. Scaling features (Crucial for Logistic Regression and consistent XAI)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert scaled arrays back to DataFrames to keep feature names (important for SHAP/LIME)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    # 4. Initialize Models
    models = {
        "Logistic_Regression": LogisticRegression(random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    # 5. Train and Evaluate
    trained_models = {}
    results = []
    
    print("\nTraining models...")
    for name, model in models.items():
        model.fit(X_train_scaled_df, y_train)
        y_pred = model.predict(X_test_scaled_df)
        acc = accuracy_score(y_test, y_pred)
        
        trained_models[name] = model
        results.append({"Model": name, "Accuracy": acc})
        
        print(f"{name} Accuracy: {acc:.4f}")
    
    # 6. Save Models, Scaler, and Splitted Data for Phase 3/4
    models_dir = 'results/models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Save models
    for name, model in trained_models.items():
        with open(os.path.join(models_dir, f'{name}.pkl'), 'wb') as f:
            pickle.dump(model, f)
            
    # Save the scaler (required for future predictions)
    with open(os.path.join(models_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
        
    # Save test data for evaluation in Phase 3/4
    processed_dir = os.path.join('data', 'processed')
    X_test_scaled_df.to_csv(os.path.join(processed_dir, 'X_test_scaled.csv'), index=False)
    y_test.to_csv(os.path.join(processed_dir, 'y_test.csv'), index=False)
    
    print(f"\nPhase 2 Complete! Models and Scaler saved in {models_dir}")

if __name__ == "__main__":
    train_models()
