"""
Research Paper Asset Generator
================================
Generates all figures, tables, and visualizations needed for the
Explainable AI (XAI) for Cardiac Risk Prediction research paper.

Run: python generate_paper_assets.py
Output: results/paper_assets/
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.table import Table
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
import shap
import lime
import lime.lime_tabular

warnings.filterwarnings('ignore')

#  Paths 
BASE = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(BASE, "results", "paper_assets")
os.makedirs(ASSET_DIR, exist_ok=True)

MODELS_DIR = os.path.join(BASE, "results", "models")
DATA_PATH  = os.path.join(BASE, "data", "heart_disease.csv")

#  Palette 
BG        = "#0f172a"
CARD      = "#1e293b"
ACCENT1   = "#6366f1"   # indigo
ACCENT2   = "#22d3ee"   # cyan
ACCENT3   = "#f59e0b"   # amber
POSITIVE  = "#10b981"   # emerald
NEGATIVE  = "#f43f5e"   # rose
TEXT      = "#e2e8f0"
SUBTEXT   = "#94a3b8"

PALETTE   = [ACCENT1, ACCENT2, ACCENT3, POSITIVE, NEGATIVE, "#a78bfa", "#34d399"]

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    CARD,
    "axes.edgecolor":    SUBTEXT,
    "axes.labelcolor":   TEXT,
    "xtick.color":       SUBTEXT,
    "ytick.color":       SUBTEXT,
    "text.color":        TEXT,
    "grid.color":        "#2d3748",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
})

def save(fig, name, dpi=180):
    path = os.path.join(ASSET_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"    Saved  {os.path.relpath(path, BASE)}")
    return path


#  Load Data & Models 
print("\n[*]  Loading data and models ...")
df = pd.read_csv(DATA_PATH)
X  = df.drop("target", axis=1)
y  = df["target"]
feature_names = list(X.columns)

FEATURE_LABELS = {
    "age":      "Age (yrs)", "sex":   "Sex",        "cp":       "Chest Pain Type",
    "trestbps": "Resting BP",  "chol": "Cholesterol", "fbs":      "Fasting Blood Sugar",
    "restecg":  "Resting ECG", "thalach": "Max Heart Rate", "exang": "Ex. Angina",
    "oldpeak":  "ST Depression","slope": "ST Slope",  "ca":       "Major Vessels",
    "thal":     "Thalassemia",
}

scaler = pickle.load(open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb"))
rf     = pickle.load(open(os.path.join(MODELS_DIR, "Random_Forest.pkl"), "rb"))
lr     = pickle.load(open(os.path.join(MODELS_DIR, "Logistic_Regression.pkl"), "rb"))
xgb    = pickle.load(open(os.path.join(MODELS_DIR, "XGBoost.pkl"), "rb"))

X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_names)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

models = {"Random Forest": rf, "Logistic Regression": lr, "XGBoost": xgb}
model_colors = {"Random Forest": ACCENT1, "Logistic Regression": ACCENT2, "XGBoost": ACCENT3}


# ── 4 IMPORTANT GRAPHS ───────────────────────────────────────────────────────

# GRAPH 1: System Architecture
print("[G1/4] Generating System Architecture ...")
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
def box(ax, x, y, w, h, label, sublabel="", color=ACCENT1):
    rect = plt.Rectangle((x-w/2, y-h/2), w, h, color=color, alpha=0.8, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y+(0.1 if sublabel else 0), label, ha="center", va="center", fontweight="bold", color="white")
    if sublabel: ax.text(x, y-0.2, sublabel, ha="center", va="center", fontsize=8, color="#cbd5e1")

box(ax, 2, 5, 2.5, 0.8, "Patient Data", "13 Clinical Features", ACCENT1)
box(ax, 5, 5, 2.5, 0.8, "ML Models", "RF + LR + XGBoost", ACCENT2)
box(ax, 8, 5, 2.5, 0.8, "XAI Engine", "SHAP + LIME", POSITIVE)
box(ax, 5, 2, 4, 1, "Diagnostic Risk Report", "Risk Score + Explanation", NEGATIVE)
ax.annotate("", xy=(3.75, 5), xytext=(3.25, 5), arrowprops=dict(arrowstyle="->", color=SUBTEXT))
ax.annotate("", xy=(6.75, 5), xytext=(6.25, 5), arrowprops=dict(arrowstyle="->", color=SUBTEXT))
ax.annotate("", xy=(5, 2.5), xytext=(5, 4.6), arrowprops=dict(arrowstyle="->", color=SUBTEXT))
save(fig, "graph1_architecture.png")

# GRAPH 2: Model Performance (ROC)
print("[G2/4] Generating ROC Curves ...")
fig, ax = plt.subplots(figsize=(8, 6))
for name, model in models.items():
    prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    ax.plot(fpr, tpr, color=model_colors[name], lw=2, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
ax.plot([0, 1], [0, 1], "--", color=SUBTEXT)
ax.set_title("Performance: Receiver Operating Characteristic", pad=15)
ax.legend(loc="lower right")
save(fig, "graph2_performance_roc.png")

# GRAPH 3: Global Feature Importance (SHAP)
print("[G3/4] Generating SHAP Global Summary ...")
plt.clf()
# Use the TreeExplainer for RF as the primary global explanation
explainer_rf = shap.TreeExplainer(rf)
sv = explainer_rf.shap_values(X_test)
if isinstance(sv, list): sv = sv[1] # Use positive class probabilities

fig = plt.figure(figsize=(10, 6))
# Removing implicit title to fix the overlap seen in screenshots
shap.summary_plot(sv, X_test, feature_names=[FEATURE_LABELS.get(f, f) for f in feature_names], show=False)
plt.title("Global Feature Impact (SHAP Summary)", pad=20, color=TEXT)
plt.gca().patch.set_facecolor(CARD)
save(plt.gcf(), "graph3_global_importance.png")

# GRAPH 4: Local Explanation (Waterfall)
print("[G4/4] Generating Local Risk Breakdown ...")
plt.clf()
shap_exp = explainer_rf(X_test)
idx = int(np.where(y_test.values == 1)[0][0]) # First high-risk patient
single_exp = shap_exp[idx, :, 1] if len(shap_exp.shape)==3 else shap_exp[idx]
single_exp.feature_names = [FEATURE_LABELS.get(f, f) for f in feature_names]

fig, ax = plt.subplots(figsize=(10, 6))
shap.plots.waterfall(single_exp, show=False)
plt.title("Patient Case: Local Risk Factor Breakdown", pad=20)
save(plt.gcf(), "graph4_local_explanation.png")


# ── 2 TABLES ─────────────────────────────────────────────────────────────────

# TABLE 1: Performance Summary
print("[T1/2] Generating Performance Table ...")
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis("off")
data = [["Model", "Accuracy", "Precision", "Recall", "F1", "AUC"]]
for name, model in models.items():
    p = model.predict(X_test)
    pr = model.predict_proba(X_test)[:, 1]
    f, t, _ = roc_curve(y_test, pr)
    data.append([name, f"{accuracy_score(y_test, p):.3f}", f"{precision_score(y_test, p):.3f}", 
                 f"{recall_score(y_test, p):.3f}", f"{f1_score(y_test, p):.3f}", f"{auc(f, t):.3f}"])
tbl = ax.table(cellText=data, loc="center", cellLoc="center", bbox=[0,0,1,1])
tbl.auto_set_font_size(False); tbl.set_fontsize(10)
for (r,c), cell in tbl.get_celld().items():
    cell.set_edgecolor(SUBTEXT)
    cell.set_facecolor(CARD if r>0 else "#334155")
    cell.set_text_props(color="white")
save(fig, "table1_performance.png")

# TABLE 2: Feature Statistics
print("[T2/2] Generating Feature Stats Table ...")
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis("off")
stats = df.describe().round(2).T.reset_index()
stats.columns = ["Feature", "Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
stats["Feature"] = stats["Feature"].map(lambda x: FEATURE_LABELS.get(x, x))
tbl2 = ax.table(cellText=[stats.columns.tolist()] + stats.values.tolist(), loc="center", cellLoc="center", bbox=[0,0,1,1])
tbl2.auto_set_font_size(False); tbl2.set_fontsize(8)
for (r,c), cell in tbl2.get_celld().items():
    cell.set_edgecolor(SUBTEXT)
    cell.set_facecolor(CARD if r>0 else "#334155")
    cell.set_text_props(color="white")
save(fig, "table2_feature_stats.png")

print("\n[✔] Final assets generated: 4 Graphs, 2 Tables.")

   
 