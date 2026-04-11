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
import io, base64, os, warnings, json, logging
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODULES_DIR = os.path.join(BASE_DIR, 'modules')

# Store loaded models and configs in memory
MODULE_REGISTRY = {}

def load_modules():
    """Dynamically loads intelligence modules from the modules directory."""
    global MODULE_REGISTRY
    MODULE_REGISTRY = {}
    
    if not os.path.exists(MODULES_DIR):
        logger.warning(f"Modules directory not found at {MODULES_DIR}. Creating it...")
        os.makedirs(MODULES_DIR)
        return
        
    for mod_id in os.listdir(MODULES_DIR):
        mod_path = os.path.join(MODULES_DIR, mod_id)
        config_path = os.path.join(mod_path, 'config.json')
        
        if os.path.isdir(mod_path) and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # Load models for this module
                models = {}
                for key, relative_path in config['model_paths'].items():
                    abs_path = os.path.join(BASE_DIR, relative_path)
                    if not os.path.exists(abs_path):
                        logger.error(f"Model file not found: {abs_path}")
                        continue
                    with open(abs_path, 'rb') as f_model:
                        models[key] = pickle.load(f_model)
                
                # Load reference data
                ref_data_path = os.path.join(BASE_DIR, config['data_paths']['reference_data'])
                if not os.path.exists(ref_data_path):
                    logger.error(f"Reference data not found: {ref_data_path}")
                    continue
                    
                ref_df = pd.read_csv(ref_data_path)
                
                MODULE_REGISTRY[mod_id] = {
                    'config': config,
                    'models': models,
                    'ref_data': ref_df,
                    'feature_names': [f['id'] for f in config['features']]
                }
                logger.info(f"Successfully loaded intelligence module: {mod_id}")
            except Exception as e:
                logger.error(f"Failed to load module {mod_id}: {str(e)}")

    logger.info(f"Registry initialized with {len(MODULE_REGISTRY)} modules.")

# Initial load
load_modules()

BG, CARD_BG = '#0f172a', '#1e293b'

def get_risk_interpretation(prob_high):
    if prob_high < 25: return "Very Low Suspicion", "Routine follow-up per standard guidelines."
    if prob_high < 55: return "Moderate Concern", "Active monitoring and risk factor management advised."
    if prob_high < 80: return "High Suspicion", "Strong clinical suspicion. Further diagnostic workup recommended."
    return "Strong Clinical Suspicion", "Urgent clinical review and comprehensive workup required."

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=BG, edgecolor='none', dpi=110)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/modules')
def get_available_modules():
    return jsonify({mid: m['config'] for mid, m in MODULE_REGISTRY.items()})

@app.route('/predict/<module_id>', methods=['POST'])
def predict(module_id):
    if module_id not in MODULE_REGISTRY:
        return jsonify({'success': False, 'error': 'Module not found'})
    
    try:
        module = MODULE_REGISTRY[module_id]
        config = module['config']
        models = module['models']
        feature_names = module['feature_names']
        ref_data = module['ref_data']
        
        data = request.json
        raw_values = [float(data[f]) for f in feature_names]
        raw_values_dict = dict(zip(feature_names, raw_values))
        
        # Scaling
        input_df = pd.DataFrame(models['scaler'].transform(np.array(raw_values).reshape(1, -1)), columns=feature_names)

        # Predictions from all models
        predictions = {}
        model_keys = ['rf', 'lr', 'xgb']
        for k in model_keys:
            if k in models:
                prob = models[k].predict_proba(input_df)[0]
                predictions[k] = {
                    'label': int(models[k].predict(input_df)[0]),
                    'prob_high': round(float(prob[1]) * 100, 1),
                    'name': config['model_paths'].get(f"{k}_name", k.upper())
                }

        # Select main model for SHAP (default RF)
        main_model = models.get('rf', list(models.values())[0])
        prob_high = predictions.get('rf', list(predictions.values())[0])['prob_high']
        
        # SHAP Explanations
        explainer = shap.TreeExplainer(main_model)
        shap_values = explainer.shap_values(input_df)
        if isinstance(shap_values, list): vals = shap_values[1][0]
        else: vals = shap_values[0]

        # Map to features and descriptions (Simplified / All-Purpose)
        plain = []
        for feat_conf in config['features']:
            fid = feat_conf['id']
            idx = feature_names.index(fid)
            sv = float(vals[idx])
            plain.append({
                'key': fid,
                'label': feat_conf['label'],
                'shap_value': sv,
                'direction': 'raises risk' if sv > 0 else 'lowers risk',
                'description': f"{feat_conf['label']} level ({raw_values_dict[fid]}) {'contributes to' if sv > 0 else 'reduces'} the identified risk profile."
            })
        
        # Sort by importance
        plain.sort(key=lambda x: abs(x['shap_value']), reverse=True)

        # Consensus
        votes_high = sum(1 for p in predictions.values() if p['label'] == 1)
        consensus = f"Consensus: {'High' if votes_high >= 2 else 'Low'} Risk ({votes_high}/3 engines agree)"

        # SHAP Plot
        plt.clf()
        shap.plots.bar(explainer(input_df)[0,:,1] if len(explainer(input_df).shape)==3 else explainer(input_df)[0], show=False)
        fig = plt.gcf()
        fig.patch.set_facecolor(BG)
        for ax in fig.axes:
            ax.set_facecolor(CARD_BG)
            ax.tick_params(colors='#94a3b8')
        shap_plot = fig_to_b64(fig)

        susp_level, susp_note = get_risk_interpretation(prob_high)

        return jsonify({
            'success': True,
            'predictions': predictions,
            'plain_explanations': plain[:6],
            'consensus': consensus,
            'suspicion_level': susp_level,
            'suspicion_note': susp_note,
            'shap_plot': shap_plot,
            'module_name': config['name']
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()})

if __name__ == '__main__':
    load_modules()
    app.run(debug=True, port=5000)
