from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import shap
import lime.lime_tabular
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io, base64, os, warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'results', 'models')
DATA_DIR   = os.path.join(BASE_DIR, 'data', 'processed')

with open(os.path.join(MODELS_DIR, 'Random_Forest.pkl'),       'rb') as f: rf_model  = pickle.load(f)
with open(os.path.join(MODELS_DIR, 'Logistic_Regression.pkl'), 'rb') as f: lr_model  = pickle.load(f)
with open(os.path.join(MODELS_DIR, 'XGBoost.pkl'),             'rb') as f: xgb_model = pickle.load(f)
with open(os.path.join(MODELS_DIR, 'scaler.pkl'),              'rb') as f: scaler    = pickle.load(f)

X_test_ref = pd.read_csv(os.path.join(DATA_DIR, 'X_test_scaled.csv'))

FEATURE_NAMES = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']

FEATURE_LABELS = {
    'age': 'Age', 'sex': 'Sex', 'cp': 'Chest Pain Type',
    'trestbps': 'Resting Blood Pressure (mmHg)', 'chol': 'Cholesterol (mg/dl)',
    'fbs': 'Fasting Blood Sugar > 120 mg/dl', 'restecg': 'Resting ECG Results',
    'thalach': 'Max Heart Rate Achieved', 'exang': 'Exercise Induced Angina',
    'oldpeak': 'ST Depression (OldPeak)', 'slope': 'Slope of Peak ST Segment',
    'ca': 'Major Vessels Coloured (0-4)', 'thal': 'Thalassemia Type',
}

MODELS_MAP  = {'rf': rf_model, 'lr': lr_model, 'xgb': xgb_model}
MODEL_NAMES = {'rf': 'Random Forest', 'lr': 'Logistic Regression', 'xgb': 'XGBoost'}
BG, CARD_BG = '#0f172a', '#1e293b'

# ── Plain-English descriptions per feature ─────────────────────────────────────
# ── Clinical Priority Mapping (Feature Groups) ──────────────────────────────────
CLINICAL_PRIORITY = {
    'cp': 'HIGH', 'oldpeak': 'HIGH', 'ca': 'HIGH', 'thal': 'HIGH',
    'trestbps': 'MED', 'chol': 'MED', 'thalach': 'MED', 'exang': 'MED',
    'age': 'LOW', 'sex': 'LOW', 'fbs': 'LOW', 'restecg': 'LOW', 'slope': 'LOW'
}

# ── Medical-Grade Feature Descriptions ──────────────────────────────────────────
def describe_feature(key, raw_val):
    v = raw_val
    if key == 'age':
        note = 'Older patients (55+) carry higher baseline systemic risk' if v >= 55 else 'Relatively lower age reduced baseline systemic risk'
        return f'Age ({int(v)} yrs): {note}'
    if key == 'sex':
        return 'Patient is male: Statistically higher relative risk in cardiac events' if v == 1 else 'Patient is female: Statistically lower relative risk'
    if key == 'cp':
        return {
            0: 'Asymptomatic: No reported pain, but clinical vigilance is required for "Silent MI" markers.',
            1: 'Non-anginal pain: Unlikely to be cardiac-related (non-specific symptoms).',
            2: 'Atypical angina: Intermediate suspicion — symptoms are not classic but significant.',
            3: 'Typical Angina: Strongest clinical indicator of potential obstructive coronary artery disease.'
        }.get(int(v), f'Chest pain type {int(v)}')
    if key == 'trestbps':
        note = 'Elevated BP (>140 mmHg) increases cardiac workload and shear stress' if v > 140 else 'Blood pressure within stable hemodynamic range'
        return f'Physiology: {note} ({int(v)} mmHg)'
    if key == 'chol':
        note = 'Hypercholesterolemia (>200 mg/dl): Increases probability of atherosclerosis and plaque' if v > 200 else 'Serum cholesterol levels are within managed limits'
        return f'Labs: {note} ({int(v)} mg/dl)'
    if key == 'fbs':
        note = 'potential Hyperglycemia (>120 mg/dl): A significant metabolic risk factor for heart disease' if v == 1 else 'Stable fasting glucose levels (<120 mg/dl)'
        return f'Metabolic: {note}'
    if key == 'restecg':
        return {
            0: 'ECG shows abnormal Repolarization: Suggests possible ischemia or electrolyte imbalance.',
            1: 'ECG is Normal: No immediate electrical abnormalities detected at rest.',
            2: 'Left Ventricular Hypertrophy: Heart muscle thickening suggests chronic strain/hypertension.'
        }.get(int(v), f'ECG: Type {int(v)}')
    if key == 'thalach':
        note = 'Reduced chronotropic response: Heart rate response to stress is suboptimal' if v < 140 else 'Normal chronotropic response: Heart achieved expected peak performance'
        return f'Vitals: {note} ({int(v)} bpm)'
    if key == 'exang':
        return 'Stress-induced ischemia: Positive for chest pain during exertion, a critical clinical marker' if v == 1 else 'Exercise Tolerance: Negative for stress-induced angina'
    if key == 'oldpeak':
        note = 'Significant ST depression (>1.5): Strong evidence of myocardial ischemia under stress' if v > 1.5 else 'Mild ST change: Borderline indicator, warrants correlation' if v > 0.5 else 'Negative ST segment: No significant ischemia detected.'
        return f'Ischemia Marker: {note} ({v:.1f} mm)'
    if key == 'slope':
        return {
            0: 'Downsloping ST: A pathological sign often associated with severe myocardial ischemia.',
            1: 'Flat ST: Often seen in chronic ischemic heart disease or metabolic strain.',
            2: 'Upsloping ST: Generally considered a physiological, normal response to stress.'
        }.get(int(v), f'ST-Slope: {int(v)}')
    if key == 'ca':
        return 'Major Vessel Narrowing: Narrowing in 0 major heart arteries detected (Excellent Prognosis)' if v == 0 else f'Vessel Obstruction: Narrowing detected in {int(v)} major arteries (Significant Risk)'
    if key == 'thal':
        return {
            0: 'Thalassemia: Not enough data available in this report.',
            1: 'Fixed Defect: Probable old infarction — blood flow is permanently reduced in areas.',
            2: 'Normal Blood Flow: Heart tissue receives optimal oxygen/blood distribution.',
            3: 'Reversible Defect: Ischemia during stress — blood flow drops only under workload.'
        }.get(int(v), f'Labs: Thalassemia Type {int(v)}')
    return f'{FEATURE_LABELS.get(key, key)}: {v}'

def get_risk_interpretation(prob_high):
    """Interpret the probability % into clinical suspicion levels."""
    if prob_high < 25: return "Very Low Suspicion", "Routine follow-up per standard guidelines."
    if prob_high < 55: return "Moderate Concern", "Active monitoring and risk factor management advised."
    if prob_high < 80: return "High Suspicion", "Strong clinical suspicion. Further diagnostic workup recommended."
    return "Strong Clinical Suspicion", "Urgent clinical review and comprehensive workup required."

def get_conversational_summary(prob_high, plain_items):
    """Generate a friendly chat-bot style explanation for general users."""
    if prob_high < 30:
        summary = "Great news! Based on the AI analysis, your cardiac risk markers are currently within a healthy range. "
    elif prob_high < 65:
        summary = "The AI has noticed some markers that warrant a bit of attention. It's not an emergency, but there are areas we can improve. "
    else:
        summary = "Our AI markers suggest a high level of concern. It’s important to discuss these findings with a healthcare professional soon. "

    factors = []
    for it in plain_items[:3]:
        if it['direction'] == 'raises risk':
            factors.append(f"Specifically, your **{it['label'].lower()}** is pushing your risk score up.")
        else:
            factors.append(f"On the positive side, your **{it['label'].lower()}** is helping keep your risk lower.")

    return summary + " ".join(factors) + " Would you like to know how to improve any of these specific areas?"

def get_recommendations(raw_values, prob_high):
    """Provide specific clinical 'next steps'."""
    steps = []
    if prob_high > 50:
        if raw_values['cp'] in [2, 3] or raw_values['oldpeak'] > 1.0:
            steps.append("Immediate: Conduct ECG Stress Test under cardiologist supervision.")
            steps.append("Secondary: Consider Cardiac Imaging (CT Angiography) if symptoms persist.")
        if raw_values['chol'] > 240 or raw_values['trestbps'] > 160:
            steps.append("Management: Intensive pharmacological management of BP and Lipids.")
    else:
        steps.append("Follow standard cardiac health guidelines and annual checkups.")
    return steps

def detect_interactions(raw_vals):
    """Find concerning patterns (Feature Interactions)."""
    insights = []
    # Pattern 1: Symptoms + ECG Ischemia
    if (raw_vals['cp'] in [2, 3]) and (raw_vals['oldpeak'] > 1.0):
        insights.append("Pattern Alert: Co-occurrence of classic symptoms and ST depression increases confidence in Ischemia.")
    # Pattern 2: Obesity/Vitals Cluster
    if raw_vals['chol'] > 240 and raw_vals['trestbps'] > 140:
        insights.append("Complexity Note: Elevated BP and High Cholesterol suggest high cardiovascular plaque burden.")
    return insights

def get_plain_english(model, model_key, input_df, raw_vals):
    """Return top factors with Clinical Rule Layer applied."""
    try:
        if model_key == 'lr':
            exp = shap.LinearExplainer(model, X_test_ref)
            vals = exp.shap_values(input_df)[0]
        else:
            exp  = shap.TreeExplainer(model)
            sv   = exp.shap_values(input_df)
            if isinstance(sv, list): vals = sv[1][0]
            elif hasattr(sv, 'shape') and len(sv.shape) == 3: vals = sv[0, :, 1]
            else: vals = sv[0]

        items = []
        for feat, val in zip(FEATURE_NAMES, vals):
            # ── Clinical Rule Override (Force Medical Logic) ──
            # Rule 1: 0 vessels MUST decrease risk
            if feat == 'ca' and float(raw_vals[feat]) == 0: val = min(val, -0.01)
            # Rule 2: Strong symptoms SHOULD increase risk
            if feat == 'cp' and float(raw_vals[feat]) == 3: val = max(val, 0.05)
            # Rule 3: High ST depression SHOULD increase risk
            if feat == 'oldpeak' and float(raw_vals[feat]) > 1.5: val = max(val, 0.05)

            items.append({
                'key': feat,
                'label': FEATURE_LABELS[feat],
                'shap_value': float(val),
                'priority': CLINICAL_PRIORITY.get(feat, 'LOW'),
                'direction': 'raises risk' if val > 0 else 'lowers risk',
                'description': describe_feature(feat, raw_vals[feat])
            })

        # Sort: High Priority first, then by absolute math impact
        p_map = {'HIGH': 3, 'MED': 2, 'LOW': 1}
        items.sort(key=lambda x: (p_map[x['priority']], abs(x['shap_value'])), reverse=True)

        return items[:6]
    except:
        return []


# ── Plot helpers ────────────────────────────────────────────────────────────────
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=BG, edgecolor='none', dpi=110)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return data

def style_ax(ax, fig):
    fig.patch.set_facecolor(BG); ax.set_facecolor(CARD_BG)
    ax.tick_params(colors='#94a3b8', labelsize=9)
    ax.xaxis.label.set_color('#cbd5e1'); ax.yaxis.label.set_color('#cbd5e1')
    ax.title.set_color('#f1f5f9')
    for sp in ax.spines.values(): sp.set_edgecolor('#334155')
    for t in ax.get_xticklabels() + ax.get_yticklabels(): t.set_color('#94a3b8')

def generate_shap_plot(model, model_key, input_df):
    plt.style.use('dark_background')
    try:
        if model_key == 'lr':
            sv = shap.LinearExplainer(model, X_test_ref)(input_df)
            local_sv = sv[0]
        else:
            sv = shap.Explainer(model, X_test_ref)(input_df, check_additivity=False)
            local_sv = sv[0, :, 1] if len(sv.shape) == 3 else sv[0]
        shap.plots.waterfall(local_sv, show=False)
        fig = plt.gcf(); fig.patch.set_facecolor(BG)
        for ax in fig.axes:
            ax.set_facecolor(CARD_BG)
            for t in ax.get_xticklabels() + ax.get_yticklabels(): t.set_color('#94a3b8')
        return fig_to_b64(fig)
    except:
        return generate_shap_bar(model, model_key, input_df)

def generate_shap_bar(model, model_key, input_df):
    try:
        if model_key == 'lr':
            vals = shap.LinearExplainer(model, X_test_ref).shap_values(input_df)[0]
        else:
            sv   = shap.TreeExplainer(model).shap_values(input_df)
            vals = sv[1][0] if isinstance(sv, list) else sv[0]
        labels  = [FEATURE_LABELS[f] for f in FEATURE_NAMES]
        colours = ['#ef4444' if v > 0 else '#22c55e' for v in vals]
        order   = np.argsort(np.abs(vals))[::-1]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh([labels[i] for i in order], [vals[i] for i in order], color=[colours[i] for i in order])
        ax.set_xlabel('SHAP Value'); ax.set_title(f'SHAP Feature Importance ({MODEL_NAMES[model_key]})')
        ax.legend(handles=[mpatches.Patch(color='#ef4444', label='Increases risk'),
                            mpatches.Patch(color='#22c55e', label='Decreases risk')],
                  facecolor=CARD_BG, labelcolor='#94a3b8')
        style_ax(ax, fig); return fig_to_b64(fig)
    except: return ""

def generate_lime_plot(model, input_df):
    try:
        exp = lime.lime_tabular.LimeTabularExplainer(
            np.array(X_test_ref), feature_names=[FEATURE_LABELS[f] for f in FEATURE_NAMES],
            class_names=['Low Risk', 'High Risk'], mode='classification'
        ).explain_instance(input_df.iloc[0].values, model.predict_proba, labels=(1,), num_features=13)
        fig = exp.as_pyplot_figure(label=1); fig.patch.set_facecolor(BG)
        for ax in fig.axes:
            ax.set_facecolor(CARD_BG); ax.tick_params(colors='#94a3b8')
            ax.title.set_color('#f1f5f9'); ax.xaxis.label.set_color('#cbd5e1')
            for t in ax.get_xticklabels() + ax.get_yticklabels(): t.set_color('#94a3b8')
            for sp in ax.spines.values(): sp.set_edgecolor('#334155')
        return fig_to_b64(fig)
    except: return ""

# ── Routes ──────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data      = request.json
        model_key = data.get('model', 'rf')

        raw_values     = [float(data[f]) for f in FEATURE_NAMES]
        raw_values_dict = dict(zip(FEATURE_NAMES, raw_values))
        input_df       = pd.DataFrame(scaler.transform(np.array(raw_values).reshape(1, -1)), columns=FEATURE_NAMES)

        predictions = {}
        for key, mdl in MODELS_MAP.items():
            prob = mdl.predict_proba(input_df)[0]
            predictions[key] = {
                'label': int(mdl.predict(input_df)[0]),
                'prob_low':  round(float(prob[0]) * 100, 1),
                'prob_high': round(float(prob[1]) * 100, 1),
                'name': MODEL_NAMES[key]
            }

        selected = MODELS_MAP[model_key]
        prob_high = predictions[model_key]['prob_high']
        susp_level, susp_note = get_risk_interpretation(prob_high)
        recommends = get_recommendations(raw_values_dict, prob_high)
        interactions = detect_interactions(raw_values_dict)

        plain = get_plain_english(selected, model_key, input_df, raw_values_dict)

        # Consensus verdict
        votes_high = sum(1 for p in predictions.values() if p['label'] == 1)
        if votes_high == 3:   consensus = "Unanimous Consensus: HIGH RISK"
        elif votes_high == 2: consensus = "Majority Consensus: HIGH RISK"
        elif votes_high == 1: consensus = "Majority Consensus: LOW RISK"
        else:                 consensus = "Unanimous Consensus: LOW RISK"

        return jsonify({
            'success': True,
            'predictions': predictions,
            'plain_explanations': plain,
            'interactions': interactions,
            'suspicion_level': susp_level,
            'suspicion_note': susp_note,
            'recommendations': recommends,
            'consensus': consensus,
            'shap_plot': generate_shap_plot(selected, model_key, input_df),
            'lime_plot': generate_lime_plot(selected, input_df),
            'chat_summary': get_conversational_summary(prob_high, plain),
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()})

if __name__ == '__main__':
    print("XAI Framework Dashboard running at http://localhost:5000")
    app.run(debug=True, port=5000)
   
 