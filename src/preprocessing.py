import pandas as pd
import numpy as np
import os

def load_data(file_path):
    """Loads the healthcare dataset."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded. Shape: {df.shape}")
    return df

def clean_data(df):
    """Basic cleaning: handle missing values and types."""
    # Check for '?' or other placeholders common in UCI datasets
    df = df.replace('?', np.nan)
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.any():
        print("Missing values detected:")
        print(missing_values[missing_values > 0])
        # For simplicity in this research, we'll drop rows with missing values
        # (Alternatively, could use median/mode imputation)
        df = df.dropna()
        print(f"Rows with missing values dropped. New shape: {df.shape}")
    else:
        print("No missing values detected.")
        
    return df

def preprocess_heart_disease(df):
    """Specific preprocessing for Heart Disease dataset."""
    # The 'target' column: 0 = No Disease, 1-4 = Presence of Disease
    # Often, research simplifies this to binary: 0 vs 1
    # Check current distribution
    print("Target distribution:")
    print(df['target'].value_counts())
    
    # Ensure binary target if needed (already seems binary in this version, but good practice)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    
    return df

def main():
    # Paths
    raw_data_path = os.path.join('data', 'heart_disease.csv')
    processed_dir = os.path.join('data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    # Execution
    df = load_data(raw_data_path)
    df = clean_data(df)
    df = preprocess_heart_disease(df)
    
    # Save processed data
    output_path = os.path.join(processed_dir, 'heart_disease_cleaned.csv')
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    main()
