import json
import logging
import os
import re

import joblib
import numpy as np
import pandas as pd
import shap
from flask import Flask, jsonify, render_template, request

logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO'),
                    format='%(asctime)s %(levelname)s %(name)s: %(message)s')
log = logging.getLogger('xai')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODULES_DIR = os.path.join(BASE_DIR, 'modules')
MODEL_NAMES = {'lr': 'Logistic Regression', 'rf': 'Random Forest', 'xgb': 'XGBoost'}
DISCLAIMER = ('This is an educational research tool, not a medical device. '
              'It does not replace diagnosis by a qualified clinician.')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024
app.json.sort_keys = False


# ── Module loading ────────────────────────────────────────────────────────────

def load_module(module_id):
    """Load a module's config, models, SHAP explainer and metrics. Returns None if incomplete."""
    with open(os.path.join(MODULES_DIR, module_id, 'config.json'), encoding='utf-8') as f:
        config = json.load(f)
    model_dir = os.path.join(BASE_DIR, config.get('model_dir', ''))
    if not config.get('model_dir') or not os.path.isdir(model_dir):
        log.warning('Skipping module %r: model directory missing (run train.py)', module_id)
        return None

    models = {k: joblib.load(os.path.join(model_dir, f'{k}.joblib')) for k in MODEL_NAMES}
    background = pd.read_csv(os.path.join(model_dir, 'background.csv'))
    explainer = shap.TreeExplainer(models[config['explain_model']], background,
                                   model_output='probability' if config['explain_model'] == 'xgb' else 'raw')
    with open(os.path.join(model_dir, 'metrics.json'), encoding='utf-8') as f:
        metrics = json.load(f)
    # Warm up so the first user request doesn't pay SHAP/numba initialisation (~seconds)
    explainer(background.iloc[[0]])
    log.info('Loaded module %r', module_id)
    return {'config': config, 'models': models, 'explainer': explainer, 'metrics': metrics}


def load_all_modules():
    loaded = {}
    for module_id in sorted(os.listdir(MODULES_DIR)):
        if os.path.isfile(os.path.join(MODULES_DIR, module_id, 'config.json')):
            mod = load_module(module_id)
            if mod:
                loaded[module_id] = mod
    return loaded


MODULES = load_all_modules()


# ── Prediction & explanation ──────────────────────────────────────────────────

def validate_input(config, data):
    """Return (values, errors). Values are floats keyed by feature id, in config order."""
    values, errors = {}, {}
    if not isinstance(data, dict):
        return None, {'_': 'Expected a JSON object.'}
    for feat in config['features']:
        fid, raw = feat['id'], data.get(feat['id'])
        if raw is None or raw == '':
            errors[fid] = f"{feat['label']} is required."
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            errors[fid] = f"{feat['label']} must be a number."
            continue
        if not np.isfinite(val):
            errors[fid] = f"{feat['label']} must be a finite number."
        elif feat['type'] == 'select' and val not in [o['v'] for o in feat['opts']]:
            errors[fid] = f"{feat['label']} has an invalid option."
        elif feat['type'] == 'number' and not feat['min'] <= val <= feat['max']:
            errors[fid] = f"{feat['label']} must be between {feat['min']} and {feat['max']} {feat['unit']}."
        else:
            values[fid] = val
    return values, errors


def predict_risk(mod, rows):
    """Disease probability (0-100) for each row: (ensemble array, {model: array}, DataFrame)."""
    X = pd.DataFrame(rows, columns=[f['id'] for f in mod['config']['features']])
    per_model = {k: m.predict_proba(X)[:, 1] * 100 for k, m in mod['models'].items()}
    return np.mean(list(per_model.values()), axis=0), per_model, X


def describe(feat, val):
    if feat['type'] == 'select':
        return next(o['d'] for o in feat['opts'] if o['v'] == val)
    text = f"{val:g} {feat['unit']}"
    lo, hi = feat.get('normal', (None, None))
    if hi is not None and val > hi:
        text += ' (above the typical range)'
    elif lo is not None and val < lo:
        text += ' (below the typical range)'
    return text


def explain(mod, X, values):
    """Per-feature SHAP contributions (percentage points of risk) for the explanation model."""
    sv = mod['explainer'](X)
    contrib = sv.values[0, :, 1] if sv.values.ndim == 3 else sv.values[0]
    base = sv.base_values[0]
    base = float(base[1] if np.ndim(base) else base)
    drivers = []
    for feat, c in zip(mod['config']['features'], contrib):
        drivers.append({
            'key': feat['id'], 'label': feat['label'], 'value': values[feat['id']],
            'impact': round(float(c) * 100, 2),
            'direction': 'raises' if c > 0 else 'lowers',
            'description': describe(feat, values[feat['id']]),
        })
    drivers.sort(key=lambda d: abs(d['impact']), reverse=True)
    return {'model': MODEL_NAMES[mod['config']['explain_model']],
            'base_risk': round(base * 100, 1), 'drivers': drivers}


def what_if_rows(config, values):
    """Scenarios for modifiable factors that are currently above target."""
    return [(rule, {**values, rule['id']: rule['to']}) for rule in config.get('what_if', [])
            if values[rule['id']] > rule['when_above']]


CONDITION_OPS = {
    '==': lambda v, ref: v == ref,
    '>': lambda v, ref: v > ref,
    '<': lambda v, ref: v < ref,
    'in': lambda v, ref: v in ref,
}


def _cond(values, cond):
    fid, op, ref = cond
    return CONDITION_OPS[op](values[fid], ref)


def recommendations(config, values, risk):
    recs = []
    for rule in config.get('recommendations', []):
        in_band = rule.get('min_risk', 0) <= risk <= rule.get('max_risk', 100)
        if in_band and (not rule['any'] or any(_cond(values, c) for c in rule['any'])):
            recs.append(rule['text'])
    return recs


def risk_band(risk):
    if risk < 20: return 'Low'
    if risk < 50: return 'Moderate'
    if risk < 75: return 'High'
    return 'Very high'


def warnings_for(config, values):
    out = []
    for feat in config['features']:
        rng = feat.get('train_range')
        if rng and not rng[0] <= values[feat['id']] <= rng[1]:
            out.append(f"{feat['label']} ({values[feat['id']]:g} {feat['unit']}) is outside the range seen in "
                       f"training ({rng[0]}-{rng[1]}), so the estimate is less reliable.")
    return out


# ── Report Q&A (rule-based, grounded only in the report) ──────────────────────

def answer_question(question, report):
    q = question.lower()
    drivers = report.get('explanation', {}).get('drivers', [])
    if not drivers:
        return 'Please complete an assessment first, then ask me about the result.'
    up = [d for d in drivers if d['impact'] > 0][:3]
    down = [d for d in drivers if d['impact'] < 0][:3]

    for d in drivers:
        names = {d['key'], d['label'].lower()}
        if any(re.search(r'\b' + re.escape(n) + r'\b', q) for n in names):
            verb = 'raised' if d['impact'] > 0 else 'lowered'
            return (f"**{d['label']}** — {d['description']}. In this assessment it {verb} the estimated risk "
                    f"by about {abs(d['impact']):.1f} percentage points.")
    if re.search(r'\b(improve|reduce|lower|better|what if|change)\b', q):
        wi = report.get('what_if', [])
        if wi:
            lines = [f"• With {w['change']}, estimated risk would be **{w['risk']}%** ({w['delta']:+.1f} pts)."
                     for w in wi]
            return 'Here is what the models predict for modifiable factors:\n' + '\n'.join(lines)
        return ('None of the modifiable factors in this model (blood pressure, cholesterol, blood sugar) '
                'are currently elevated. Other factors such as age or test results cannot simply be changed.')
    if re.search(r'\b(agree|models?\b|consensus|confiden|sure\b|uncertain)', q):
        preds = report.get('predictions', {})
        parts = [f"{p['name']} {p['risk']}%" for p in preds.values()]
        spread = max(p['risk'] for p in preds.values()) - min(p['risk'] for p in preds.values()) if preds else 0
        note = 'The models broadly agree.' if spread < 15 else 'The models disagree noticeably, so treat the estimate with extra caution.'
        return f"Model estimates: {', '.join(parts)}. {note}"
    if re.search(r'\b(accura|reliab|trust|valid|performance|auc\b)', q):
        m = report.get('metrics', {})
        return (f"On a held-out test set the ensemble reached an ROC-AUC of **{m.get('roc_auc', '?')}** and accuracy of "
                f"**{m.get('accuracy', '?')}**. The dataset is small (about 300 patients from one hospital), so real-world "
                'performance may be lower.')
    if re.search(r'\b(shap\b|explain|how does|method)', q):
        return ('Each factor\'s impact comes from **SHAP**, which splits the model\'s prediction into contributions from '
                'every input. Starting from the average risk in the training data, each factor pushes the estimate up or down.')
    if re.search(r'\b(why|reason|factor|cause|driver)\b', q):
        txt = ''
        if up:
            txt += 'Factors raising the risk: ' + '; '.join(f"**{d['label']}** (+{d['impact']:.1f})" for d in up) + '. '
        if down:
            txt += 'Factors lowering it: ' + '; '.join(f"**{d['label']}** ({d['impact']:.1f})" for d in down) + '.'
        return txt
    return ('You can ask, for example: "why is the risk high?", "what can lower it?", "do the models agree?", '
            '"how accurate is this?", or about a specific factor like "cholesterol".')


# ── Routes ────────────────────────────────────────────────────────────────────

def public_config(config):
    keys = ('id', 'name', 'icon', 'description', 'features')
    return {k: config[k] for k in keys if k in config}


@app.after_request
def security_headers(resp):
    resp.headers['X-Content-Type-Options'] = 'nosniff'
    resp.headers['X-Frame-Options'] = 'SAMEORIGIN'
    resp.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return resp


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/favicon.ico')
def favicon():
    return '', 204


@app.route('/healthz')
def healthz():
    return jsonify({'status': 'ok', 'modules': list(MODULES)})


@app.route('/modules')
def list_modules():
    return jsonify({mid: public_config(m['config']) for mid, m in MODULES.items()})


@app.route('/predict/<module_id>', methods=['POST'])
def predict(module_id):
    mod = MODULES.get(module_id)
    if not mod:
        return jsonify({'success': False, 'error': 'Unknown module.'}), 404
    values, errors = validate_input(mod['config'], request.get_json(silent=True))
    if errors:
        return jsonify({'success': False, 'error': 'Invalid input.', 'fields': errors}), 400
    try:
        scenarios = what_if_rows(mod['config'], values)
        risks, per_model, X = predict_risk(mod, [values] + [row for _, row in scenarios])
        risk = float(risks[0])
        what_if = [{'change': rule['text'], 'risk': round(float(r), 1), 'delta': round(float(r) - risk, 1)}
                   for (rule, _), r in zip(scenarios, risks[1:])]
        ens_metrics = mod['metrics'].get('ensemble', {})
        return jsonify({
            'success': True,
            'module': module_id,
            'inputs': values,
            'risk': round(risk, 1),
            'band': risk_band(risk),
            'predictions': {k: {'name': MODEL_NAMES[k], 'risk': round(float(p[0]), 1)} for k, p in per_model.items()},
            'explanation': explain(mod, X.iloc[[0]], values),
            'what_if': what_if,
            'recommendations': recommendations(mod['config'], values, risk),
            'warnings': warnings_for(mod['config'], values),
            'metrics': ens_metrics,
            'disclaimer': DISCLAIMER,
        })
    except Exception:
        log.exception('Prediction failed for module %r', module_id)
        return jsonify({'success': False, 'error': 'Internal error while generating the report.'}), 500


@app.route('/chat', methods=['POST'])
def chat():
    body = request.get_json(silent=True) or {}
    message = str(body.get('message', ''))[:500]
    report = body.get('report') if isinstance(body.get('report'), dict) else {}
    try:
        return jsonify({'success': True, 'reply': answer_question(message, report)})
    except Exception:
        log.exception('Chat failed')
        return jsonify({'success': True, 'reply': "Sorry, I couldn't interpret that. Try asking about a specific factor."})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f'Explainable AI dashboard running at http://localhost:{port}')
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_DEBUG') == '1')
