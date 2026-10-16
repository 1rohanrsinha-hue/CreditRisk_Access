import sys
from pathlib import Path
import numpy as np, pandas as pd, pytest
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from creditrisk import serving as S

VALID = dict(loan_amount=12000, interest_rate=13.5, annual_income=60000, fico_mean=690, debt_to_income=18.5,
             num_credit_lines=11, total_credit_lines=25, delinquencies=0, credit_inquiries=1, mortgage_accounts=1,
             public_records=0, collections=0, employment_length='5 years', loan_purpose='debt_consolidation',
             home_ownership='RENT')


def synthetic(n=4000, seed=0):
    r = np.random.RandomState(seed)
    X = pd.DataFrame(dict(
        loan_amount=r.uniform(1000, 35000, n), annual_income=r.lognormal(11, .5, n), debt_to_income=r.uniform(0, 40, n),
        delinquencies=r.poisson(.3, n).astype(float), num_credit_lines=r.poisson(10, n).astype(float),
        total_credit_lines=r.poisson(24, n).astype(float) + 1, loan_purpose=r.choice(S.PURPOSES, n),
        home_ownership=r.choice(['RENT', 'MORTGAGE', 'OWN'], n), public_records=r.poisson(.1, n).astype(float),
        credit_inquiries=r.poisson(1, n).astype(float), mortgage_accounts=r.poisson(1, n).astype(float),
        collections=r.poisson(.05, n).astype(float), employment_length=r.randint(0, 11, n).astype(float),
        interest_rate=r.uniform(6, 26, n), fico_mean=r.uniform(620, 800, n)))
    X.loc[r.rand(n) < .06, 'employment_length'] = np.nan
    X['emp_length_missing'] = X.employment_length.isna().astype(float)
    z = -2.6 + .12 * (X.interest_rate - 13) + .02 * X.debt_to_income - .004 * (X.fico_mean - 700)
    y = (r.rand(n) < 1 / (1 + np.exp(-z))).astype(int)
    meta = pd.DataFrame(dict(issue_d=['Jan-2012'] * n))
    return X[S.FEATURES], y, meta


SMALL = dict(n_estimators=40, learning_rate=0.1, num_leaves=8, min_child_samples=20, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.8, reg_lambda=0.1, max_depth=-1)


@pytest.fixture(scope="session")
def tiny_deploy(tmp_path_factory):
    d = tmp_path_factory.mktemp("deploy")
    X, y, meta = synthetic()
    bundle, metadata = S.fit_bundle(X, y, meta, 'isotonic', n_jobs=2, params=SMALL, log=lambda m: None)
    S.save_bundle(bundle, metadata, d)
    return d


@pytest.fixture(scope="session")
def client(tiny_deploy):
    import app as appmod
    a = appmod.create_app(deploy_dir=tiny_deploy, show_rate_borrower=False)
    a.config['TESTING'] = True
    return a.test_client()
