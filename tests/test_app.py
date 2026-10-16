import pytest
from conftest import VALID
import app as appmod
from creditrisk import serving as S


def post(client, d):
    return client.post('/api/predict', json=d)


def test_api_valid(client):
    r = post(client, VALID); assert r.status_code == 200
    j = r.get_json()
    for k in ('probability', 'tier', 'tier_index', 'training_prevalence', 'reason_codes_borrower', 'reason_codes', 'notice', 'ebm_probability'):
        assert k in j
    assert 0 <= j['probability'] <= 1 and len(j['reason_codes_borrower']) <= 3
    assert 'interest_rate' not in [c['feature'] for c in j['reason_codes_borrower']]


def test_api_missing_field_is_400_not_zero_filled(client):
    d = dict(VALID); del d['debt_to_income']
    r = post(client, d); assert r.status_code == 400
    assert any("Missing required field 'debt_to_income'" in e for e in r.get_json()['errors'])


def test_api_unknown_category_rejected(client):
    r = post(client, dict(VALID, loan_purpose='yacht')); assert r.status_code == 400
    assert any("Unknown category for 'loan_purpose'" in e for e in r.get_json()['errors'])


def test_api_out_of_range_and_type_errors_listed_together(client):
    r = post(client, dict(VALID, fico_mean=1000, loan_amount='x')); assert r.status_code == 400
    assert len(r.get_json()['errors']) == 2


def test_api_not_json_and_non_object(client):
    assert client.post('/api/predict', data='x', content_type='text/plain').status_code == 400
    assert client.post('/api/predict', json=[1, 2]).status_code == 400


def test_api_schema_and_health(client):
    j = client.get('/api/schema').get_json()
    assert set(j['required']) == set(S.INPUT_FIELDS) and j['schema_hash'] == S.schema_hash() and len(j['tiers']) == 3
    h = client.get('/health').get_json(); assert h['status'] == 'ok'


@pytest.mark.parametrize("path", ['/', '/predict', '/dashboard', '/model-results'])
def test_pages_render_with_prototype_notice(client, path):
    r = client.get(path); assert r.status_code == 200 and b'Research prototype' in r.data


def test_old_hardcoded_bands_and_cutoff_are_gone(client):
    html = b''.join(client.get(p).data for p in ('/', '/predict', '/dashboard', '/model-results'))
    for bad in (b'High Risk', b'Low Risk', b'15%', b'50%'):
        assert bad not in html, bad


def test_form_valid_and_invalid(client):
    f = {k: str(v) for k, v in VALID.items()}
    r = client.post('/predict', data=f); assert r.status_code == 200 and b'Calibrated probability of default' in r.data
    assert b'interest rate' not in r.data.lower().split(b'top reasons')[1].split(b'reasons come from')[0]
    f['loan_amount'] = ''
    r = client.post('/predict', data=f); assert r.status_code == 400 and b'required' in r.data
    f['loan_amount'] = '12000'; f['loan_purpose'] = 'yacht'
    r = client.post('/predict', data=f); assert r.status_code == 400 and b'Unknown category' in r.data


def test_figures_whitelist(client):
    assert client.get('/figures/not_in_list.png').status_code == 404
    assert client.get('/figures/../app.py').status_code == 404


def test_app_refuses_to_start_without_model(tmp_path):
    with pytest.raises(S.ModelLoadError):
        appmod.create_app(deploy_dir=tmp_path)


def test_app_never_trains_or_loads_dataset():
    src = open(appmod.__file__).read()
    for bad in ('.fit(', 'read_csv(ROOT', 'accepted_2007', 'train_all_models'):
        assert bad not in src
