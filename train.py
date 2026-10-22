"""
Reproducible training for the cardiac module.

Run:    python train.py
Output: models/cardiac/{lr,rf,xgb}.joblib, background.csv, metrics.json

Label note
----------
data/heart_disease.csv is the Kaggle copy of the UCI Cleveland dataset. Matching
all 303 rows against the original UCI file shows that `target == 1` means NO
heart disease (UCI num == 0). We therefore train on `disease = 1 - target`, so
class 1 of every model is "has heart disease" and predict_proba[:, 1] is risk.

The Kaggle copy also re-codes some categoricals (verified against UCI):
  cp      0 asymptomatic, 1 atypical angina, 2 non-anginal pain, 3 typical angina
  restecg 0 LV hypertrophy, 1 normal, 2 ST-T wave abnormality
  slope   0 downsloping, 1 flat, 2 upsloping
  thal    1 fixed defect, 2 normal, 3 reversible defect   (0 = missing in UCI)
  ca      0-3 vessels                                      (4 = missing in UCI)
"""

import json
import os
import platform

import joblib
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

SEED = 42
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, 'data', 'heart_disease.csv')
OUT_DIR = os.path.join(BASE, 'models', 'cardiac')

FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
NOMINAL = ['cp', 'restecg', 'slope', 'thal']


def load_data():
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    n0 = len(df)
    df = df.drop_duplicates()
    # ca == 4 and thal == 0 are missing values in the original UCI data
    df = df[(df['ca'] <= 3) & (df['thal'] != 0)].reset_index(drop=True)
    print(f'Loaded {n0} rows, kept {len(df)} after removing duplicates and missing-value placeholders')
    X = df[FEATURES].astype(float)
    y = (1 - df['target']).astype(int).rename('disease')
    return X, y


def build_models():
    numeric = [f for f in FEATURES if f not in NOMINAL]
    lr = Pipeline([
        ('prep', ColumnTransformer([
            ('num', StandardScaler(), numeric),
            ('cat', OneHotEncoder(handle_unknown='ignore'), NOMINAL),
        ])),
        ('clf', LogisticRegression(max_iter=2000, C=0.5)),
    ])
    # Tree models are trained on raw clinical units so SHAP values read naturally
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, max_features='sqrt',
                                random_state=SEED, n_jobs=-1)
    xgb = XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.9,
                        colsample_bytree=0.8, eval_metric='logloss', random_state=SEED)
    return {'lr': lr, 'rf': rf, 'xgb': xgb}


def evaluate(model, X, y):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        'accuracy': round(accuracy_score(y, pred), 4),
        'precision': round(precision_score(y, pred), 4),
        'recall': round(recall_score(y, pred), 4),
        'f1': round(f1_score(y, pred), 4),
        'roc_auc': round(roc_auc_score(y, prob), 4),
        'brier': round(brier_score_loss(y, prob), 4),
    }


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    metrics = {'n_train': len(X_train), 'n_test': len(X_test), 'models': {}}

    test_probs = []
    for key, model in build_models().items():
        cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        model.fit(X_train, y_train)
        m = evaluate(model, X_test, y_test)
        m['cv_auc_mean'] = round(cv_auc.mean(), 4)
        m['cv_auc_std'] = round(cv_auc.std(), 4)
        metrics['models'][key] = m
        test_probs.append(model.predict_proba(X_test)[:, 1])
        if 'n_jobs' in model.get_params():
            model.set_params(n_jobs=1)  # parallelism only slows down single-row predictions
        joblib.dump(model, os.path.join(OUT_DIR, f'{key}.joblib'))
        print(f'{key:>4}: test AUC {m["roc_auc"]:.3f}  acc {m["accuracy"]:.3f}  '
              f'CV AUC {m["cv_auc_mean"]:.3f} +/- {m["cv_auc_std"]:.3f}')

    ens = np.mean(test_probs, axis=0)
    metrics['ensemble'] = {
        'accuracy': round(accuracy_score(y_test, ens >= 0.5), 4),
        'roc_auc': round(roc_auc_score(y_test, ens), 4),
    }
    print(f' ens: test AUC {metrics["ensemble"]["roc_auc"]:.3f}  acc {metrics["ensemble"]["accuracy"]:.3f}')

    # Background sample for SHAP (training data only, so no test leakage)
    X_train.sample(n=min(100, len(X_train)), random_state=SEED).to_csv(
        os.path.join(OUT_DIR, 'background.csv'), index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(os.path.join(OUT_DIR, 'test_split.csv'), index=False)

    metrics['versions'] = {
        'python': platform.python_version(), 'scikit-learn': sklearn.__version__,
        'xgboost': xgboost.__version__, 'shap': shap.__version__,
    }
    with open(os.path.join(OUT_DIR, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f'Saved models and metrics to {os.path.relpath(OUT_DIR, BASE)}')


if __name__ == '__main__':
    main()
