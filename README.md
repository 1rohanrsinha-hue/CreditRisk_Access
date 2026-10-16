# Credit-risk research prototype (v2)

**Research prototype - not for credit decisions.** Companion code to a study of 10 machine-learning models for credit-default
prediction on Lending Club loans (2007-2018): calibration, decision thresholds, weighting, profit and explainability.
Findings, hypotheses and every decision are logged in `EXTENSION_TRACKER.md`. The conference-paper app is preserved
unchanged at tag `paper-v1-baseline`; this branch (`app-v2`) serves a pre-trained, calibrated model instead.

## What the app does
- `/predict` (form) and `POST /api/predict` (JSON): calibrated probability of default (tuned LightGBM + isotonic calibration),
  one of three **review tiers** anchored on the training prevalence pi (PD < 0.75 pi, 0.75-1.5 pi, > 1.5 pi), pi itself, and the
  top-3 **reason codes** from an interpretable EBM (direction-aware wording).
- Tiers are *review* tiers, **not accept/reject cutoffs**: on mature vintages no threshold rule earned more than approving everyone.
- Input is validated: all 15 fields required, ranges enforced, unknown categories rejected, nothing is defaulted or zero-filled.
  `employment_length` accepts `"not reported"` explicitly.
- `interest_rate` is the lender's own price. It may appear in the **internal** reason codes (`reason_codes` in the JSON) but is
  hidden from the **borrower-facing** view (`reason_codes_borrower`, and the HTML page) unless `BORROWER_SHOW_RATE=true`.
- `/dashboard` and `/model-results` read the cached study tables and figures in `experiments/results/` and `experiments/figures/`
  (read-only); the app never trains and never reads the dataset. `/health`, `/api/schema` are also available.

## Train the deployment model
Requires the Lending Club CSV (`accepted_2007_to_2018Q4.csv`, not in the repo) at the project root; the first run builds a parquet cache in `.cache/`.
```
python -m venv venv && source venv/bin/activate
pip install -r requirements-lock.txt          # full research environment (Python 3.13)
python scripts/train_deploy.py                # ~10-20 min; writes models/deploy/ (bundle.joblib, metadata.json, model_card.md, ...)
python scripts/train_deploy.py --calibrator platt --out models/deploy_platt   # Platt instead of isotonic
```
The bundle stores the fitted pipeline, calibrator, EBM, feature list and dtypes, training prevalence, library versions, training
date range, git commit and timestamp. `creditrisk.serving.load_bundle` validates schema, hash and library versions and fails loudly
on any mismatch; the app refuses to start without a valid bundle.

## Run
```
python app.py                                 # http://127.0.0.1:5000
gunicorn "app:create_app()"                   # production
curl -s localhost:5000/api/predict -H 'Content-Type: application/json' -d @example.json
```
Environment: `CREDIT_DEPLOY_DIR` (bundle directory), `BORROWER_SHOW_RATE` (default false), `HOST`, `PORT`.

## Test
```
pip install pytest
python -m pytest            # creditrisk unit tests, Flask route tests, and the serving-parity test (H24)
```
The parity test needs `models/deploy/` (it is skipped otherwise).

## Docker
```
docker compose up --build                     # http://localhost:8000, healthcheck on /health
```
Image: python slim, pinned `requirements-docker.txt`, non-root user, gunicorn. It contains `models/deploy/`, the templates, the
`creditrisk` package and the cached results, and excludes the dataset, caches, virtualenvs and legacy `models/*.pkl`.

## Recalibration schedule and monitoring
Calibration drifts out of time (study Phase 3C: raw ECE .041/.022/.034 for 2013-2015 test years, far above the in-time ECE).
**Refit the calibrator (or retrain) on recent labelled loans at least annually**, and monitor mean predicted PD against the realised
default rate per tier. Labels for a vintage mature slowly (36-60 months), so a live system will always calibrate on partly
resolved data.

## Limitations
Research prototype only; no fairness, stability or regulatory validation. Funded loans only (rejected applicants unobserved);
vintages 2010-2015 only; in-time test split (out-of-time AUC is about .016 lower). `interest_rate` is endogenous. Reason-code wording
is descriptive, not causal, and whether it satisfies adverse-action-notice requirements is a legal question outside the study.
See `models/deploy/model_card.md` for the model card.

## Repository layout
`creditrisk/` library (data, features, models, calibration, thresholds, serving, tiers, reason codes) | `experiments/` study scripts,
`results/` (CSV), `figures/` | `scripts/train_deploy.py` | `app.py`, `templates/` | `tests/` | `EXTENSION_TRACKER.md`.
