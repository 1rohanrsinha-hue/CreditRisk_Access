"""H24 (frozen at f60b319): serving parity on 1,000 held-out loans."""
import numpy as np, pandas as pd, pytest
from creditrisk import serving as S

D = S.DEPLOY_DIR
pytestmark = pytest.mark.skipif(not (D / S.BUNDLE_FILE).exists(), reason="models/deploy not trained (python scripts/train_deploy.py)")


@pytest.fixture(scope="module")
def setup():
    import app as appmod
    a = appmod.create_app(); a.config['TESTING'] = True
    P = pd.read_csv(D / "parity_sample.csv"); E = pd.read_csv(D / "parity_expected_in_memory.csv")
    return a.test_client(), a.config['BUNDLE'], P, E


def rows(P):
    cols = S.INPUT_FIELDS
    return [{c: (r[c] if isinstance(r[c], str) else float(r[c])) for c in cols if not (c == 'employment_length' and pd.isna(r[c]))}
            | ({'employment_length': 'not reported'} if pd.isna(r['employment_length']) else {}) for _, r in P.iterrows()]


def test_sample_size(setup):
    _, _, P, E = setup; assert len(P) == 1000 and len(E) == 1000


def test_reloaded_pipeline_matches_in_memory(setup):
    _, bundle, P, E = setup
    got = [S.score_one(bundle, d) for d in rows(P)]
    p = np.array([g['probability'] for g in got]); e = np.array([g['ebm_probability'] for g in got])
    assert np.max(np.abs(p - E.probability.values)) < 1e-9
    assert np.max(np.abs(e - E.ebm_probability.values)) < 1e-9
    assert (np.array([g['tier_index'] for g in got]) == E.tier_index.values).all()


def test_api_matches_library_call(setup):
    client, bundle, P, _ = setup
    worst = 0.0
    for d in rows(P):
        lib = S.score_one(bundle, d)['probability']
        r = client.post('/api/predict', json=d); assert r.status_code == 200
        worst = max(worst, abs(r.get_json()['probability'] - lib))
    assert worst < 1e-9


def test_vectorised_library_matches_per_loan(setup):
    _, bundle, P, _ = setup
    X = pd.DataFrame([S.validate_input(d)[0] for d in rows(P)])[S.FEATURES]
    pdv, _ = S.predict_frame(bundle, X)
    one = np.array([S.score_one(bundle, d)['probability'] for d in rows(P)])
    assert np.max(np.abs(pdv - one)) < 1e-9
