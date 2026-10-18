"""Deployment bundle: training, persistence, schema/version-validated loading, input validation
and scoring (calibrated PD, review tier, EBM reason codes).

The SAME fitted pipeline object that was trained is the one persisted and served: encoder, LightGBM and
calibrator are never re-created at request time. RESEARCH PROTOTYPE - not for credit decisions."""
import hashlib, json, math, os, platform, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

from .config import ROOT, RESULTS_DIR
from .features import FrameEncoder, EMP_ORDER, EMP_MAP, NOMINAL
from .calibration import Isotonic, Platt
from .reason_codes import direction_label
from .tiers import assign_tier, tier_edges, TIER_NAMES, TIER_DEFINITIONS

DEPLOY_DIR = ROOT / "models" / "deploy"
BUNDLE_FILE, META_FILE = "bundle.joblib", "metadata.json"
SEED = 42
FEATURES = ['loan_amount', 'annual_income', 'debt_to_income', 'delinquencies', 'num_credit_lines', 'total_credit_lines',
            'loan_purpose', 'home_ownership', 'public_records', 'credit_inquiries', 'mortgage_accounts', 'collections',
            'employment_length', 'emp_length_missing', 'interest_rate', 'fico_mean']
INPUT_FIELDS = [f for f in FEATURES if f != 'emp_length_missing']      # emp_length_missing is derived
PURPOSES = ['car', 'credit_card', 'debt_consolidation', 'educational', 'home_improvement', 'house', 'major_purchase',
            'medical', 'moving', 'other', 'renewable_energy', 'small_business', 'vacation', 'wedding']
HOME = ['ANY', 'MORTGAGE', 'NONE', 'OTHER', 'OWN', 'RENT']
NOT_REPORTED = ['not reported', 'n/a', '']
# name: (kind, lo, hi, integer?)  hard limits for input validation (outside = rejected, not clipped)
RANGES = {
    'loan_amount': ('number', 500, 40_000, False), 'annual_income': ('number', 1, 10_000_000, False),
    'debt_to_income': ('number', 0, 100, False), 'delinquencies': ('number', 0, 40, True),
    'num_credit_lines': ('number', 0, 90, True), 'total_credit_lines': ('number', 1, 200, True),
    'public_records': ('number', 0, 50, True), 'credit_inquiries': ('number', 0, 40, True),
    'mortgage_accounts': ('number', 0, 40, True), 'collections': ('number', 0, 20, True),
    'interest_rate': ('number', 5, 31, False), 'fico_mean': ('number', 300, 850, False),
}
DTYPES = {f: ('category' if f in NOMINAL else 'float64') for f in FEATURES}
NOTICE = "Research prototype - not for credit decisions."


class ModelLoadError(RuntimeError):
    pass


class ValidationError(ValueError):
    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def schema_hash():
    blob = json.dumps({'features': FEATURES, 'dtypes': DTYPES, 'ranges': RANGES, 'purposes': PURPOSES, 'home': HOME,
                       'emp': EMP_ORDER}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def library_versions():
    import sklearn, lightgbm, interpret, joblib
    return dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scikit_learn=sklearn.__version__,
                lightgbm=lightgbm.__version__, interpret=interpret.__version__, joblib=joblib.__version__)


def _git(*args):
    for g in ('git', '/Library/Developer/CommandLineTools/usr/bin/git'):
        try:
            out = subprocess.run([g, *args], cwd=ROOT, capture_output=True, text=True, timeout=20)
            if out.returncode == 0:
                return out.stdout.strip()
        except Exception:
            continue
    return 'unknown'


# ------------------------------------------------------------------------------ parameters
def median_params(model='LightGBM', path=None):
    """Deployment hyper-parameters: median over the 5 outer-fold Optuna winners for numeric parameters
    (integers rounded), mode for categorical ones (documented in the model card)."""
    P = json.load(open(path or RESULTS_DIR / "optuna_params.json"))
    ds = [v for k, v in P.items() if k.startswith(model + "|")]
    out = {}
    for k in ds[0]:
        vals = [d[k] for d in ds]
        if isinstance(vals[0], str):
            out[k] = max(set(vals), key=vals.count)
        elif isinstance(vals[0], int):
            out[k] = int(round(float(np.median(vals))))
        else:
            out[k] = float(np.median(vals))
    return out


def make_lgbm_pipeline(params, seed=SEED, n_jobs=4):
    from sklearn.pipeline import Pipeline
    from lightgbm import LGBMClassifier
    clf = LGBMClassifier(random_state=seed, verbose=-1, n_jobs=n_jobs, **params)
    return Pipeline([('enc', FrameEncoder(mode='category')), ('clf', clf)])


# ------------------------------------------------------------------------------ EBM companion
class EBMCompanion:
    """EBM (3D configuration, interactions=10) with exact additive contributions per original feature
    (interaction terms split equally between their two features)."""
    def __init__(self, seed=SEED, n_jobs=4, interactions=10):
        self.seed, self.n_jobs, self.interactions = seed, n_jobs, interactions

    def fit(self, X, y):
        from interpret.glassbox import ExplainableBoostingClassifier
        self.enc_ = FrameEncoder('catboost').fit(X, y)
        self.ebm_ = ExplainableBoostingClassifier(random_state=self.seed, n_jobs=self.n_jobs, interactions=self.interactions)
        self.ebm_.fit(self.enc_.transform(X), np.asarray(y))
        return self

    def predict_proba(self, X):
        return self.ebm_.predict_proba(self.enc_.transform(X))[:, 1]

    def contributions(self, X):
        T = self.ebm_.eval_terms(self.enc_.transform(X))
        names = list(self.ebm_.feature_names_in_)
        C = pd.DataFrame(0.0, index=range(len(X)), columns=names)
        for j, tf in enumerate(self.ebm_.term_features_):
            for f in tf:
                C[names[f]] += T[:, j] / len(tf)
        return C


# ------------------------------------------------------------------------------ fitting
def fit_bundle(X, y, meta, calibrator='isotonic', n_jobs=4, seed=SEED, params=None, fit_ebm=True, log=print):
    """X: raw feature frame (FEATURES columns), y: 0/1, meta: frame with issue_d. Returns (bundle, metadata)."""
    from sklearn.model_selection import StratifiedKFold
    if calibrator not in ('isotonic', 'platt'):
        raise ValueError("calibrator must be 'isotonic' or 'platt'")
    X = X[FEATURES]; y = pd.Series(np.asarray(y).astype(int), index=X.index)
    params = params or median_params('LightGBM')
    t0 = time.time()
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    for a, b in skf.split(X, y):                       # fresh pipeline per fold: calibrator never sees in-fold scores
        pipe = make_lgbm_pipeline(params, seed, n_jobs).fit(X.iloc[a], y.iloc[a])
        oof[b] = pipe.predict_proba(X.iloc[b])[:, 1]
    cal = (Isotonic() if calibrator == 'isotonic' else Platt()).fit(oof, y.values)
    log(f"[deploy] OOF calibration done {time.time()-t0:.0f}s")
    pipe = make_lgbm_pipeline(params, seed, n_jobs).fit(X, y)
    log(f"[deploy] final LightGBM fit {time.time()-t0:.0f}s")
    ebm = EBMCompanion(seed, n_jobs).fit(X, y) if fit_ebm else None
    log(f"[deploy] EBM fit {time.time()-t0:.0f}s")
    pi = float(y.mean())
    q = X[[f for f in FEATURES if f not in NOMINAL]].quantile([.001, .999])
    med = X[[f for f in FEATURES if f not in NOMINAL]].median()
    dates = pd.to_datetime(meta['issue_d'], format='%b-%Y', errors='coerce')
    bundle = dict(pipeline=pipe, calibrator=cal, ebm=ebm, features=FEATURES, dtypes=DTYPES, pi=pi,
                  medians={k: float(v) for k, v in med.items()}, q001={k: float(v) for k, v in q.loc[.001].items()},
                  q999={k: float(v) for k, v in q.loc[.999].items()}, calibrator_kind=calibrator, params=params)
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'), git_commit=_git('rev-parse', 'HEAD'),
                    git_branch=_git('rev-parse', '--abbrev-ref', 'HEAD'), git_dirty=bool(_git('status', '--porcelain', '--untracked-files=no')),
                    versions=library_versions(), schema_hash=schema_hash(), features=FEATURES, dtypes=DTYPES,
                    n_train=int(len(X)), prevalence=pi, calibrator=calibrator, lightgbm_params=params,
                    tier_edges_pd=list(tier_edges(pi)), training_issue_date_min=str(dates.min().date()),
                    training_issue_date_max=str(dates.max().date()), has_ebm=ebm is not None,
                    notice=NOTICE)
    return bundle, metadata


def save_bundle(bundle, metadata, out_dir=DEPLOY_DIR):
    import joblib
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_dir / BUNDLE_FILE, compress=3)
    metadata = dict(metadata, bundle_bytes=(out_dir / BUNDLE_FILE).stat().st_size,
                    bundle_sha256=hashlib.sha256((out_dir / BUNDLE_FILE).read_bytes()).hexdigest())
    json.dump(metadata, open(out_dir / META_FILE, 'w'), indent=2)
    return metadata


# ------------------------------------------------------------------------------ loading
def load_bundle(deploy_dir=DEPLOY_DIR, strict_versions=True):
    """Load and VALIDATE: files present, hash of the binary, schema hash, library versions. Raises ModelLoadError."""
    import joblib
    d = Path(deploy_dir)
    if not (d / BUNDLE_FILE).exists() or not (d / META_FILE).exists():
        raise ModelLoadError(f"No deployment model in {d}. Train one with: python scripts/train_deploy.py")
    meta = json.load(open(d / META_FILE))
    if hashlib.sha256((d / BUNDLE_FILE).read_bytes()).hexdigest() != meta.get('bundle_sha256'):
        raise ModelLoadError("bundle.joblib does not match metadata.json (hash mismatch); retrain or restore the files")
    if meta.get('schema_hash') != schema_hash():
        raise ModelLoadError(f"Schema mismatch: model trained with schema {meta.get('schema_hash')}, code has {schema_hash()}")
    cur = library_versions(); bad = []
    for k in ('scikit_learn', 'lightgbm', 'interpret', 'numpy', 'pandas'):
        if meta['versions'].get(k) != cur[k]:
            bad.append(f"{k}: trained {meta['versions'].get(k)}, installed {cur[k]}")
    if bad and strict_versions:
        raise ModelLoadError("Library version mismatch (" + "; ".join(bad) + "). Install the pinned versions or retrain.")
    bundle = joblib.load(d / BUNDLE_FILE)
    if list(bundle['features']) != FEATURES or bundle['dtypes'] != DTYPES:
        raise ModelLoadError("Feature list/dtypes in the bundle differ from the code schema")
    bundle['metadata'] = meta
    return bundle


# ------------------------------------------------------------------------------ validation
def _num(name, v, errors):
    if isinstance(v, bool) or v is None or (isinstance(v, str) and v.strip() == ''):
        errors.append(f"'{name}' is required and must be a number"); return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        errors.append(f"'{name}' must be a number, got {v!r}"); return None
    if not math.isfinite(x):
        errors.append(f"'{name}' must be a finite number"); return None
    return x


def validate_input(data):
    """Return (row dict with the 16 model features, list of warnings). Raises ValidationError with ALL problems.
    No defaults, no zero-filling, no substitution of unknown categories."""
    errors, warnings = [], []
    if not isinstance(data, dict):
        raise ValidationError(["Request body must be a JSON object of feature values"])
    unknown = sorted(set(data) - set(INPUT_FIELDS))
    if unknown:
        errors.append(f"Unknown field(s): {', '.join(unknown)}. Allowed: {', '.join(INPUT_FIELDS)}")
    row = {}
    for f in INPUT_FIELDS:
        if f not in data:
            errors.append(f"Missing required field '{f}'"); continue
        v = data[f]
        if f == 'loan_purpose':
            if v not in PURPOSES:
                errors.append(f"Unknown category for 'loan_purpose': {v!r}. Allowed: {', '.join(PURPOSES)}")
            else:
                row[f] = v
        elif f == 'home_ownership':
            if v not in HOME:
                errors.append(f"Unknown category for 'home_ownership': {v!r}. Allowed: {', '.join(HOME)}")
            else:
                row[f] = v
        elif f == 'employment_length':
            if v is None or (isinstance(v, str) and v.strip().lower() in NOT_REPORTED):
                row[f] = float('nan')                                  # explicit 'not reported' (model has a missing-indicator feature)
            elif isinstance(v, str) and v in EMP_MAP:
                row[f] = float(EMP_MAP[v])
            else:
                errors.append(f"Unknown category for 'employment_length': {v!r}. Allowed: {', '.join(EMP_ORDER)} or 'not reported'")
        else:
            _, lo, hi, integer = RANGES[f]
            x = _num(f, v, errors)
            if x is None:
                continue
            if x < lo or x > hi:
                errors.append(f"'{f}' = {x:g} is outside the accepted range [{lo}, {hi}]")
            elif integer and x != int(x):
                errors.append(f"'{f}' must be a whole number, got {x:g}")
            else:
                row[f] = x
    if errors:
        raise ValidationError(errors)
    row['emp_length_missing'] = 1.0 if math.isnan(row['employment_length']) else 0.0
    return row, warnings


def range_warnings(bundle, row):
    w = []
    for f, v in row.items():
        if f in bundle['q001'] and not (isinstance(v, float) and math.isnan(v)):
            if v < bundle['q001'][f] or v > bundle['q999'][f]:
                w.append(f"'{f}' = {v:g} is outside the 0.1%-99.9% range seen in training "
                         f"[{bundle['q001'][f]:g}, {bundle['q999'][f]:g}]; the prediction is an extrapolation")
    return w


def to_frame(row):
    return pd.DataFrame([{f: row[f] for f in FEATURES}], columns=FEATURES).astype({f: 'float64' for f in FEATURES if f not in NOMINAL})


# ------------------------------------------------------------------------------ scoring
def predict_frame(bundle, X):
    """Library-level scoring of a raw feature frame: calibrated PD and raw score (vectorised)."""
    X = X[FEATURES]
    raw = bundle['pipeline'].predict_proba(X)[:, 1]
    return bundle['calibrator'].predict(raw), raw


def reason_codes(bundle, X, k=3, exclude=(), contributions=None):
    """Top-k risk-increasing EBM contributions per row with direction-aware labels.
    employment_length and its missing-indicator are merged into one code."""
    C = (contributions if contributions is not None else bundle['ebm'].contributions(X[FEATURES])).copy()
    C['employment_length'] = C['employment_length'] + C['emp_length_missing']; C = C.drop(columns='emp_length_missing')
    out = []
    for i in range(len(C)):
        r = C.iloc[i].drop(labels=[e for e in exclude if e in C.columns])
        top = r[r > 0].sort_values(ascending=False).head(k)
        codes = []
        for f, c in top.items():
            v = X.iloc[i][f]
            codes.append(dict(feature=f, contribution=float(c),
                              label=direction_label(f, v, bundle['medians'].get(f))))
        out.append(codes)
    return out


def score_one(bundle, data, show_rate_borrower=False, k=3):
    """Validate, score one applicant. Returns the response dict (JSON-serialisable)."""
    row, warnings = validate_input(data)
    X = to_frame(row)
    pdv, raw = predict_frame(bundle, X)
    pi = bundle['pi']
    tier = int(assign_tier(pdv, pi)[0])
    resp = dict(probability=float(pdv[0]), raw_score=float(raw[0]), tier=TIER_NAMES[tier], tier_index=tier + 1,
                tier_definition=TIER_DEFINITIONS[tier], training_prevalence=pi,
                tier_edges_pd=list(tier_edges(pi)), calibrator=bundle['calibrator_kind'],
                warnings=warnings + range_warnings(bundle, row), notice=NOTICE)
    if bundle.get('ebm') is not None:
        C = bundle['ebm'].contributions(X)
        resp['ebm_probability'] = float(bundle['ebm'].predict_proba(X)[0])
        resp['reason_codes'] = reason_codes(bundle, X, k, contributions=C)[0]                      # internal: rate allowed
        resp['reason_codes_borrower'] = reason_codes(bundle, X, k, () if show_rate_borrower else ('interest_rate',), contributions=C)[0]
    return resp
