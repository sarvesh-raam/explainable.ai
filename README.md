# explainable.ai — Cardiac Risk with Explainable ML

[![tests](https://github.com/sarvesh-raam/explainable.ai/actions/workflows/tests.yml/badge.svg)](https://github.com/sarvesh-raam/explainable.ai/actions/workflows/tests.yml)

**Live demo: https://explainable-ai-nr7p.onrender.com**
(click *High-risk sample* or *Low-risk sample* for an instant report)

A web app that estimates the probability of coronary heart disease from 13 routine clinical
measurements, then explains **why** using SHAP. It shows each factor's contribution in
percentage points, plain-language descriptions, model agreement, what-if scenarios and
suggested next steps.

> **Disclaimer:** this is an educational research prototype, not a medical device.

## How it works

| Step | What happens |
|------|--------------|
| Input | A guided chat collects 13 features (validated on the client and the server). |
| Prediction | Logistic Regression, Random Forest and XGBoost each predict a disease probability. The headline risk is their average. |
| Explanation | SHAP (`TreeExplainer`, Random Forest) splits the prediction into per-factor contributions in raw clinical units. Contributions add up exactly to the model output. |
| Report | Risk band, model agreement, factor chart, what-if for modifiable factors, recommendations, PDF export and a Q&A chat grounded in the report. |

### Model performance (held-out test set, n = 60)

| Model | ROC-AUC | Accuracy | Recall | 5-fold CV AUC (train) |
|-------|--------:|---------:|-------:|----------------------:|
| Logistic Regression | 0.958 | 0.867 | 0.821 | 0.893 ± 0.048 |
| Random Forest       | 0.955 | 0.867 | 0.821 | 0.895 ± 0.028 |
| XGBoost             | 0.939 | 0.850 | 0.786 | 0.875 ± 0.030 |
| **Ensemble**        | **0.954** | **0.850** | | |

The test set is small, so the cross-validated AUC (~0.89) is the more reliable estimate.

### Data notes

`data/heart_disease.csv` is the Kaggle copy of the [UCI Cleveland dataset](https://archive.ics.uci.edu/dataset/45/heart+disease).
Matching every row against the original UCI file shows that:

- **`target = 1` means *no* disease.** `train.py` trains on `disease = 1 - target`.
- Categories are re-coded relative to UCI: `cp` 0 = asymptomatic, 1 = atypical angina,
  2 = non-anginal, 3 = typical angina; `restecg` 0 = LV hypertrophy, 1 = normal, 2 = ST-T abnormality.
- `ca = 4` and `thal = 0` are missing values in UCI. Those 7 rows and 1 duplicate are dropped.

## Project structure

```
app.py                  Flask app: validation, prediction, SHAP, report Q&A
train.py                Reproducible training -> models/cardiac/
modules/cardiac/        Feature definitions, what-if and recommendation rules (config.json)
models/cardiac/         Trained models, SHAP background, metrics.json
templates/index.html    Single-page chat UI
tests/                  pytest suite
generate_paper_assets.py  Figures for the research paper (uses the original results/models)
```

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements-dev.txt
python train.py                 # optional: models are already committed
python app.py                   # http://localhost:5001
python -m pytest                # run tests
```

## Deploy

The app ships as a Docker image (`Dockerfile`). It listens on `$PORT` (default 7860) and exposes `/healthz`.

**Render (current host, free plan):**

1. Sign in at [render.com](https://render.com) with GitHub.
2. New → **Blueprint** → select this repo. `render.yaml` sets up the service on the free plan.
3. Every push to `main` redeploys automatically.

Free instances sleep after 15 minutes without traffic. An [UptimeRobot](https://uptimerobot.com) HTTP monitor
pings `/healthz` every 5 minutes to keep the demo awake. One always-on service fits within Render's
750 free instance hours per month.

**Any Docker host:**

```bash
docker build -t explainable-ai .
docker run -p 7860:7860 explainable-ai
```

## Limitations

- About 300 patients from a single hospital (1988). Results will not generalise without external validation.
- The what-if scenarios show model associations, not causal treatment effects.
- The chat is rule-based and only answers from the generated report.
