import pytest

import app as xai

HEALTHY = dict(age=41, sex=0, cp=1, trestbps=120, chol=190, fbs=0, restecg=1,
               thalach=172, exang=0, oldpeak=0, slope=2, ca=0, thal=2)
SICK = dict(age=63, sex=1, cp=0, trestbps=150, chol=280, fbs=1, restecg=0,
            thalach=118, exang=1, oldpeak=2.6, slope=1, ca=2, thal=3)


@pytest.fixture
def client():
    return xai.app.test_client()


def predict(client, payload):
    return client.post('/predict/cardiac', json=payload)


def test_modules_lists_cardiac_without_internal_paths(client):
    mods = client.get('/modules').get_json()
    assert 'cardiac' in mods
    assert 'model_dir' not in mods['cardiac']
    assert len(mods['cardiac']['features']) == 13


def test_healthz(client):
    assert client.get('/healthz').get_json()['status'] == 'ok'


def test_risk_direction_is_clinically_sensible(client):
    # Guards against the label inversion in the Kaggle copy of the dataset
    low = predict(client, HEALTHY).get_json()
    high = predict(client, SICK).get_json()
    assert low['risk'] < 20 and low['band'] == 'Low'
    assert high['risk'] > 75
    for model in high['predictions'].values():
        assert model['risk'] > 50


def test_shap_contributions_are_additive(client):
    body = predict(client, SICK).get_json()
    exp, rf_risk = body['explanation'], body['predictions']['rf']['risk']
    total = exp['base_risk'] + sum(d['impact'] for d in exp['drivers'])
    assert total == pytest.approx(rf_risk, abs=0.5)


def test_drivers_are_sorted_and_described(client):
    drivers = predict(client, SICK).get_json()['explanation']['drivers']
    impacts = [abs(d['impact']) for d in drivers]
    assert impacts == sorted(impacts, reverse=True)
    assert all(d['description'] for d in drivers)


@pytest.mark.parametrize('field,value', [
    ('age', 'abc'), ('age', 500), ('cp', 7), ('ca', 4), ('chol', None), ('oldpeak', float('nan')),
])
def test_invalid_input_is_rejected(client, field, value):
    resp = predict(client, {**SICK, field: value})
    assert resp.status_code == 400
    body = resp.get_json()
    assert field in body['fields'] and 'trace' not in body


def test_non_json_body_is_rejected(client):
    assert client.post('/predict/cardiac', data='x').status_code == 400


def test_unknown_module(client):
    assert predict(client, {}).status_code == 400
    assert client.post('/predict/finance', json=SICK).status_code == 404


def test_out_of_training_range_warns(client):
    body = predict(client, {**SICK, 'chol': 650}).get_json()
    assert any('Cholesterol' in w for w in body['warnings'])


def test_chat_answers_from_report(client):
    report = predict(client, SICK).get_json()
    ask = lambda q: client.post('/chat', json={'message': q, 'report': report}).get_json()['reply']
    assert 'raising' in ask('why is it high?').lower()
    assert 'Cholesterol' in ask('tell me about cholesterol')
    assert 'ROC-AUC' in ask('how accurate is this?')
    assert 'Logistic Regression' in ask('do the models agree?')


def test_chat_without_report(client):
    reply = client.post('/chat', json={'message': 'why'}).get_json()['reply']
    assert 'complete an assessment' in reply
