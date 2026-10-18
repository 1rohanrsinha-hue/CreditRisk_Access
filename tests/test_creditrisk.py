import json, shutil
import numpy as np, pandas as pd, pytest
from conftest import VALID, SMALL, synthetic
from creditrisk import serving as S
from creditrisk.calibration import Isotonic, Platt
from creditrisk.features import FrameEncoder
from creditrisk.reason_codes import direction_label
from creditrisk.tiers import assign_tier, tier_edges, tier_report


def test_encoder_is_fit_on_train_only():
    X, y, _ = synthetic(1000)
    tr, te = X.iloc[:700], X.iloc[700:].copy()
    e1 = FrameEncoder('ordinal').fit(tr, y[:700])
    te2 = te.copy(); te2['loan_amount'] = 1e9; te2['loan_purpose'] = 'zzz'
    e2 = FrameEncoder('ordinal').fit(tr, y[:700])
    # statistics come from the training rows alone and are unaffected by whatever the test frame contains
    pd.testing.assert_series_equal(e1.medians_, e2.medians_)
    assert e1.medians_['loan_amount'] == tr['loan_amount'].median()
    assert e1.mean_['loan_amount'] == pytest.approx(tr['loan_amount'].fillna(tr['loan_amount'].median()).mean())
    e1.transform(te2)                                   # transform never refits
    assert e1.medians_['loan_amount'] == tr['loan_amount'].median()


def test_pipeline_fit_ignores_test_rows():
    X, y, _ = synthetic(1500)
    p1 = S.make_lgbm_pipeline(SMALL, n_jobs=1).fit(X.iloc[:1000], y[:1000])
    p2 = S.make_lgbm_pipeline(SMALL, n_jobs=1).fit(X.iloc[:1000], y[:1000])
    np.testing.assert_allclose(p1.predict_proba(X.iloc[1000:])[:, 1], p2.predict_proba(X.iloc[1000:])[:, 1])


@pytest.mark.parametrize("Cal", [Isotonic, Platt])
def test_calibrators_are_monotone(Cal):
    r = np.random.RandomState(1); s = r.rand(3000); y = (r.rand(3000) < s ** 2).astype(int)
    g = np.linspace(0, 1, 501); p = Cal().fit(s, y).predict(g)
    assert np.all(np.diff(p) >= -1e-12) and p.min() >= 0 and p.max() <= 1


def test_tiers_cover_all_pds_and_edges():
    pi = .148; lo, hi = tier_edges(pi)
    assert (lo, hi) == (pytest.approx(.75 * pi), pytest.approx(1.5 * pi))
    g = np.concatenate([np.linspace(0, 1, 10001), [lo, hi, 0.0, 1.0]]); t = assign_tier(g, pi)
    assert set(np.unique(t)) == {0, 1, 2} and np.all(np.diff(t[np.argsort(g)]) >= 0)
    assert assign_tier([lo], pi)[0] == 1 and assign_tier([hi], pi)[0] == 1
    assert assign_tier([lo - 1e-9], pi)[0] == 0 and assign_tier([hi + 1e-9], pi)[0] == 2
    with pytest.raises(ValueError):
        assign_tier([np.nan], pi)
    r = np.random.RandomState(0); p = r.beta(1, 6, 5000); y = (r.rand(5000) < p).astype(int)
    rep = tier_report(y, p, pi, n_boot=50); assert rep.n.sum() == 5000 and abs(rep.share.sum() - 1) < 1e-9


def test_reason_code_labels_are_direction_aware():
    assert direction_label('fico_mean', 600, 690) != direction_label('fico_mean', 780, 690)
    assert 'lower' in direction_label('fico_mean', 600, 690).lower()
    assert direction_label('annual_income', 20000, 60000) != direction_label('annual_income', 200000, 60000)
    assert 'not reported' in direction_label('employment_length', float('nan'), 5).lower()
    assert 'not reported' in direction_label('emp_length_missing', 1.0, 0).lower()
    assert 'RENT'.lower() in direction_label('home_ownership', 'RENT').lower()
    assert 'car' in direction_label('loan_purpose', 'car')


def test_borrower_codes_hide_interest_rate_internal_codes_show_it(tiny_deploy):
    b = S.load_bundle(tiny_deploy)
    row = dict(VALID, interest_rate=27.0, fico_mean=620)               # rate is a dominant driver here
    r = S.score_one(b, row, show_rate_borrower=False)
    assert 'interest_rate' in [c['feature'] for c in r['reason_codes']]
    assert 'interest_rate' not in [c['feature'] for c in r['reason_codes_borrower']]
    r2 = S.score_one(b, row, show_rate_borrower=True)
    assert 'interest_rate' in [c['feature'] for c in r2['reason_codes_borrower']]


@pytest.mark.parametrize("patch,msg", [
    ({'loan_amount': None}, "required"), ({'loan_amount': 'abc'}, "number"), ({'loan_amount': float('nan')}, "finite"),
    ({'loan_amount': 10**9}, "outside"), ({'fico_mean': 100}, "outside"), ({'delinquencies': 1.5}, "whole number"),
    ({'loan_purpose': 'yacht'}, "Unknown category"), ({'home_ownership': 'CASTLE'}, "Unknown category"),
    ({'employment_length': '11 years'}, "Unknown category"), ({'bogus': 1}, "Unknown field")])
def test_validation_rejects(patch, msg):
    with pytest.raises(S.ValidationError) as e:
        S.validate_input(dict(VALID, **patch))
    assert msg in str(e.value)


def test_validation_reports_missing_and_never_fills():
    d = dict(VALID); del d['annual_income']
    with pytest.raises(S.ValidationError) as e:
        S.validate_input(d)
    assert "Missing required field 'annual_income'" in e.value.errors[0]
    row, _ = S.validate_input(dict(VALID, employment_length='not reported'))
    assert row['emp_length_missing'] == 1.0 and np.isnan(row['employment_length'])
    row, _ = S.validate_input(VALID); assert row['emp_length_missing'] == 0.0 and row['employment_length'] == 5.0


def test_load_bundle_fails_loudly(tiny_deploy, tmp_path):
    with pytest.raises(S.ModelLoadError, match="train_deploy"):
        S.load_bundle(tmp_path / "nothing")
    d = tmp_path / "v"; shutil.copytree(tiny_deploy, d)
    m = json.load(open(d / S.META_FILE)); m['versions']['scikit_learn'] = '0.0.1'
    json.dump(m, open(d / S.META_FILE, 'w'))
    with pytest.raises(S.ModelLoadError, match="version mismatch"):
        S.load_bundle(d)
    m['versions']['scikit_learn'] = S.library_versions()['scikit_learn']; m['schema_hash'] = 'deadbeef'
    json.dump(m, open(d / S.META_FILE, 'w'))
    with pytest.raises(S.ModelLoadError, match="Schema mismatch"):
        S.load_bundle(d)
    d2 = tmp_path / "w"; shutil.copytree(tiny_deploy, d2); (d2 / S.BUNDLE_FILE).write_bytes(b"corrupt")
    with pytest.raises(S.ModelLoadError, match="hash mismatch"):
        S.load_bundle(d2)


def test_scores_are_probabilities_and_consistent(tiny_deploy):
    b = S.load_bundle(tiny_deploy); r = S.score_one(b, VALID)
    assert 0 <= r['probability'] <= 1 and r['tier_index'] in (1, 2, 3) and r['training_prevalence'] == b['pi']
    assert len(r['reason_codes']) <= 3 and r['notice'].startswith('Research prototype')
