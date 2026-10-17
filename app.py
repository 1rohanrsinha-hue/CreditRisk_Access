"""Credit-risk research prototype (Flask) - v2.

Serves a PRE-TRAINED, calibrated model from models/deploy/ (see scripts/train_deploy.py). The app never trains
and never touches the dataset; dashboard and model-results pages read the cached result tables and figures
in experiments/results/ and experiments/figures/ (read-only).

    python app.py                          # development server
    gunicorn "app:create_app()"            # production (see Dockerfile)

Research prototype - not for credit decisions."""
import os
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from creditrisk import serving as S
from creditrisk.config import RESULTS_DIR, ROOT
from creditrisk.tiers import TIER_NAMES, TIER_DEFINITIONS

FIGURES_DIR = ROOT / "experiments" / "figures"
SHOWN_FIGURES = {
    'reliability_diagrams.png': "Reliability diagrams: raw vs isotonic calibrated scores (D2 sample, repeat 0; not the D3 headline set)",
    'tstar_vs_pi.png': "Youden-optimal threshold t* by calibration (D2 sample, mean ± sd over repeats; not the D3 headline set): after calibration it lies close to the prevalence pi",
    'profit_d3.png': "D3 (mature vintages), in-time: no threshold rule earns more than approving everyone at 0-3% discount",
    'capacity_v2.png': "Capacity-constrained approval: dollar vs count capacity",
    'drift_fixes.png': "Out-of-time calibration drift and recalibration fixes",
    'phase4_shap_top10.png': "Top-10 features by share of mean |attribution|",
    'phase4_shap_rankcorr.png': "Rank correlation of attributions across models",
    'phase4_term_vintage.png': "Default-risk ranking gain by test vintage and term",
}


def _env_flag(name, default=False):
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ('1', 'true', 'yes', 'on')


def _table(name, **filters):
    f = RESULTS_DIR / name
    if not f.exists():
        return None
    df = pd.read_csv(f)
    for k, v in filters.items():
        df = df[df[k] == v] if not isinstance(v, (list, tuple)) else df[df[k].isin(v)]
    return df


def _rows(df, cols, fmt=None):
    if df is None or df.empty:
        return []
    fmt = fmt or {}
    out = []
    for r in df[cols].itertuples(index=False):
        out.append([(fmt[c].format(v) if c in fmt and pd.notna(v) else v) for c, v in zip(cols, r)])
    return out


def create_app(deploy_dir=None, show_rate_borrower=None):
    """App factory. Loads the deployment bundle at startup; raises ModelLoadError with a clear message if it is
    missing or inconsistent (no silent retraining)."""
    app = Flask(__name__)
    app.config['SHOW_RATE_BORROWER'] = _env_flag('BORROWER_SHOW_RATE', False) if show_rate_borrower is None else show_rate_borrower
    app.config['MAX_CONTENT_LENGTH'] = 64 * 1024
    dd = Path(deploy_dir or os.environ.get('CREDIT_DEPLOY_DIR') or S.DEPLOY_DIR)
    bundle = S.load_bundle(dd)
    app.config['BUNDLE'] = bundle
    meta = bundle['metadata']

    @app.context_processor
    def inject():
        return dict(notice=S.NOTICE, meta=meta, pi=bundle['pi'])

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/health')
    def health():
        return jsonify(status='ok', model_created_utc=meta['created_utc'], git_commit=meta['git_commit'],
                       calibrator=meta['calibrator'], schema_hash=meta['schema_hash'])

    @app.route('/api/schema')
    def schema():
        fields = {}
        for f in S.INPUT_FIELDS:
            if f == 'loan_purpose':
                fields[f] = dict(type='category', allowed=S.PURPOSES)
            elif f == 'home_ownership':
                fields[f] = dict(type='category', allowed=S.HOME)
            elif f == 'employment_length':
                fields[f] = dict(type='category', allowed=S.EMP_ORDER + ['not reported'])
            else:
                _, lo, hi, integer = S.RANGES[f]
                fields[f] = dict(type='integer' if integer else 'number', min=lo, max=hi)
        return jsonify(fields=fields, required=S.INPUT_FIELDS, schema_hash=S.schema_hash(), notice=S.NOTICE,
                       tiers=[dict(name=n, definition=d) for n, d in zip(TIER_NAMES, TIER_DEFINITIONS)],
                       training_prevalence=bundle['pi'])

    @app.route('/api/predict', methods=['POST'])
    def api_predict():
        data = request.get_json(silent=True)
        if data is None:
            return jsonify(errors=["Request body must be JSON (Content-Type: application/json)"]), 400
        try:
            return jsonify(S.score_one(bundle, data, show_rate_borrower=app.config['SHOW_RATE_BORROWER']))
        except S.ValidationError as e:
            return jsonify(errors=e.errors), 400
        except Exception as e:                                   # pragma: no cover
            app.logger.exception("prediction failed")
            return jsonify(errors=[f"Internal error: {type(e).__name__}"]), 500

    @app.route('/predict', methods=['GET', 'POST'])
    def predict():
        ctx = dict(purposes=S.PURPOSES, home=S.HOME, emp=S.EMP_ORDER + ['not reported'], ranges=S.RANGES,
                   form={}, errors=[], result=None)
        if request.method == 'POST':
            form = {k: request.form.get(k) for k in S.INPUT_FIELDS if k in request.form}
            ctx['form'] = form
            try:
                res = S.score_one(bundle, form, show_rate_borrower=app.config['SHOW_RATE_BORROWER'])
                res['codes'] = res['reason_codes_borrower']          # borrower-facing view: interest_rate hidden unless flagged
                ctx['result'] = res
            except S.ValidationError as e:
                ctx['errors'] = e.errors
                return render_template('predict.html', **ctx), 400
        return render_template('predict.html', **ctx)

    @app.route('/dashboard')
    def dashboard():
        tiers = _table_deploy('tier_report.csv')
        metrics = _table_deploy('test_metrics.csv')
        return render_template('dashboard.html',
                               tier_rows=_rows(tiers, ['tier', 'definition', 'share', 'observed_default_rate', 'mean_pd', 'gap_obs_minus_pd'],
                                               {'share': '{:.3f}', 'observed_default_rate': '{:.3f}', 'mean_pd': '{:.3f}', 'gap_obs_minus_pd': '{:+.4f}'}),
                               metric_rows=_rows(metrics, ['model', 'scores', 'metric', 'value', 'ci_lo', 'ci_hi'],
                                                 {'value': '{:.4f}', 'ci_lo': '{:.4f}', 'ci_hi': '{:.4f}'}),
                               figures=[(n, SHOWN_FIGURES[n]) for n in ('reliability_diagrams.png', 'tstar_vs_pi.png') if (FIGURES_DIR / n).exists()])

    def _table_deploy(name):
        f = dd / name
        return pd.read_csv(f) if f.exists() else None

    @app.route('/model-results')
    def model_results():
        stats = _table('phase3c_stats.csv', dataset='D3', metric='ROC_AUC')
        opt = _table('phase3d_optuna_summary.csv')
        prof = _table('phase3b_profit_d3.csv', scenario='P2_3%', model='LightGBM')
        interp = _table('phase3d_interp_tests.csv', cell='with_rate')
        drift = _table('phase3c_drift.csv')
        if drift is not None:
            drift = drift.groupby(['T', 'fix']).ECE.mean().unstack(0).reset_index()
        return render_template(
            'model_results.html',
            stats_rows=_rows(stats, ['model', 'mean_value', 'mean_delta', 'ci_lo', 'ci_hi', 'equivalent'],
                             {'mean_value': '{:.4f}', 'mean_delta': '{:+.4f}', 'ci_lo': '{:+.4f}', 'ci_hi': '{:+.4f}'}),
            opt_rows=_rows(opt, ['model', 'AUC_default', 'AUC_tuned', 'dAUC', 'dAUC_lo', 'dAUC_hi'],
                           {c: '{:.4f}' for c in ['AUC_default', 'AUC_tuned', 'dAUC', 'dAUC_lo', 'dAUC_hi']}),
            profit_rows=_rows(prof, ['rule', 'profit_per_1000_mean', 'accept_rate_mean'], {'profit_per_1000_mean': '{:,.0f}', 'accept_rate_mean': '{:.3f}'}),
            interp_rows=_rows(interp, ['model', 'model_auc', 'best_auc', 'mean_delta', 'label'],
                              {'model_auc': '{:.4f}', 'best_auc': '{:.4f}', 'mean_delta': '{:+.4f}'}),
            drift_rows=[] if drift is None else [[r['fix']] + [f"{r[c]:.3f}" for c in drift.columns[1:]] for _, r in drift.iterrows()],
            drift_cols=[] if drift is None else [str(c) for c in drift.columns[1:]],
            figures=[(n, t) for n, t in SHOWN_FIGURES.items() if (FIGURES_DIR / n).exists()])

    @app.route('/figures/<name>')
    def figure(name):
        if name not in SHOWN_FIGURES:
            abort(404)
        return send_from_directory(FIGURES_DIR, name, max_age=3600)

    @app.errorhandler(404)
    def nf(e):
        return jsonify(errors=["Not found"]), 404

    return app


if __name__ == '__main__':
    try:
        application = create_app()
    except S.ModelLoadError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    application.run(host=os.environ.get('HOST', '127.0.0.1'), port=int(os.environ.get('PORT', '5000')), debug=False)
