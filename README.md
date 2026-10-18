# Calibration, Base Rates and Decision Thresholds in Credit-Risk Classification

**Research prototype. Not for credit decisions.** This repository holds the code, cached results and paper assets for a
multi-dataset study of ten machine-learning credit-default models: calibration, the Youden-optimal threshold and the
base rate, class weighting, profit-based decisions, drift and explainability. It is the journal extension of a conference
paper; every analysis step, hypothesis (H1-H28) and decision is logged in [`EXTENSION_TRACKER.md`](EXTENSION_TRACKER.md).

Authors: Rohan Sinha and Kaliraj S. Licence: MIT (code). Data have their own terms (see below).

## Branches and tags
| ref | content |
|---|---|
| `main` | everything: study code, results, paper assets, data cards, tracker, and the research-prototype Flask app with its deployment bundle and tests |
| tag `paper-v2` | the exact code, results and prototype for the IEEE Access manuscript |
| tag `paper-v1-baseline` | the conference-paper app, frozen as submitted |
| `journal-extension`, `app-v2` | earlier working branches, merged into `main` and kept for history |

To use the version described in the paper:
```
git clone https://github.com/1rohanrsinha-hue/CreditRisk_Access.git
cd CreditRisk_Access
git checkout paper-v2
```

## Directory map
| path | what it is |
|---|---|
| `creditrisk/` | library: leak-free features and encoders, the 10 models, calibration, thresholds, profit definitions, repeated-split scoring, statistics, external-dataset loaders |
| `experiments/` | one script per study stage; `results/` holds the result CSVs, `figures/` the study figures |
| `paper_assets/` | tables (CSV and LaTeX) and figures (PNG and PDF) for the paper, `MANIFEST.md`, `manuscript_facts.md`, `verify.py` |
| `docs/data_cards/` | data cards for Lending Club, German Credit and Taiwan Default |
| `scripts/` | `reproduce_all.sh` (all 43 stages), `fetch_external.sh` (UCI downloads), `build_manuscript_facts.py` |
| `app.py`, `templates/`, `creditrisk/serving.py`, `creditrisk/tiers.py` | the research-prototype Flask app: serves the pre-trained calibrated model with review tiers and reason codes |
| `models/deploy/` | the deployment bundle, model card, tier report and serving-parity files |
| `scripts/train_deploy.py` | trains the deployment bundle (needs the Lending Club CSV) |
| `tests/`, `pytest.ini` | 39 tests: library, Flask routes and serving parity (H24) |
| `Dockerfile`, `docker-compose.yml`, `requirements-docker.txt` | container image for the app |
| `requirements-lock.txt`, `requirements-dev.txt` | pinned research environment (Python 3.13); `pytest` for the tests |

## Setup
Every command below is run from the repository root and uses the virtualenv's interpreter explicitly (`venv/bin/python`); if you
prefer, run `source venv/bin/activate` and use `python` instead.
```
python3.13 -m venv venv
venv/bin/python -m pip install -r requirements-lock.txt
venv/bin/python -m pip install -r requirements-dev.txt     # pytest, only needed for the tests
```
Results were produced with Python 3.13.5 on macOS (arm64, 10 cores); pandas 3.0.2 matters (see `scripts/reproduce_all.sh`).

## Data
| dataset | use | source and licence |
|---|---|---|
| Lending Club accepted loans 2007-2018Q4 (`accepted_2007_to_2018Q4.csv`) | D1, D2, D3 | **Not included in this repository.** Kaggle dataset "All Lending Club loan data" (wordsforthewise), https://www.kaggle.com/datasets/wordsforthewise/lending-club; file `accepted_2007_to_2018Q4.csv`, SHA-256 `3eae03c28fd9d2e8a076ebeb73507e8d4d0f44d90500decdb0936e0933d1f36a`; accessed March 2026; licence CC0 1.0 Universal (Public Domain Dedication), as shown on the Kaggle dataset page. Place the file at the repository root. If you only have the `.csv.gz`, run `venv/bin/python extract_dataset.py` first. |
| German Credit (UCI Statlog) | external check, n = 1,000 | UCI Machine Learning Repository, CC BY 4.0 |
| Default of Credit Card Clients, Taiwan (UCI) | external check, n = 30,000 | UCI Machine Learning Repository, CC BY 4.0 |

Details, file hashes and label coding are in `docs/data_cards/`. The two UCI files are downloaded by `scripts/fetch_external.sh` into `external_data/` (git-ignored).

## Reproduce the study
```
bash scripts/reproduce_all.sh --dry-run     # print the 43 stages and approximate runtimes
bash scripts/reproduce_all.sh               # run everything (many hours, resumable; needs the Lending Club CSV)
bash scripts/reproduce_all.sh --from 39     # start at a stage, e.g. the external datasets (stages 39-42)
```
Heavy stages checkpoint to `.cache/` and skip finished units. The raw Lending Club CSV is required for the Lending Club stages; the
external-dataset stages (39-42) only need network access to UCI.

**Re-running a stage.** Each finished stage leaves a marker in `.cache/repro_done/` and is skipped on the next run, even if its
inputs or code changed. After changing anything a stage depends on, re-run it with `FORCE=1` (for example
`FORCE=1 bash scripts/reproduce_all.sh --from 19`), or delete its marker file. Scoring checkpoints (`.cache/scores` and similar)
are separate; delete them as well if the data or the models changed.

**D3 scoring.** Stage 18 (`p3b_score_D3`) scores only the Random Forest / Extra Trees leaf sweep. The main D3 scores (10 models x
2 weightings x 10 repeats = 200 files, about 19 minutes on 5 workers) come from stage 19 (`p3b_score_D3_main`); the D3 analysis in
stage 20 needs both.

Without the raw data you can still check the shipped results:
```
venv/bin/python paper_assets/verify.py      # checks every table against the CSVs and the tracker, and recomputes key numbers from cached scores when they are present
```
(Cached scores in `.cache/` are not shipped: the independent recomputation checks that need them print `[skip]` when they are absent, and the table and tracker checks use the committed CSVs.)

## Research prototype (Flask app)
**Research prototype - not for credit decisions.** The app serves one pre-trained, calibrated model (tuned LightGBM with isotonic
calibration) and an interpretable EBM for reason codes. It does not train and never reads the dataset at request time.

What it does:
- `/predict` (form) and `POST /api/predict` (JSON): calibrated probability of default, one of three **review tiers** anchored on the
  training prevalence pi (PD < 0.75 pi, 0.75-1.5 pi, > 1.5 pi), pi itself, and the top-3 **reason codes** (direction-aware wording).
- The tiers are review tiers, **not accept/reject cutoffs**: on mature vintages no threshold rule earned more than approving everyone.
- Input is validated: all 15 fields are required, ranges are enforced, unknown categories are rejected, nothing is defaulted or
  zero-filled. `employment_length` accepts `"not reported"` explicitly.
- `interest_rate` is the lender's own price. It may appear in the internal reason codes (`reason_codes` in the JSON) but is hidden
  from the borrower-facing view (`reason_codes_borrower` and the HTML page) unless `BORROWER_SHOW_RATE=true`.
- `/dashboard` and `/model-results` read the cached study tables and figures in `experiments/results/` and `experiments/figures/`
  (read-only). `/health` and `/api/schema` are also available.

Train the deployment model (needs the Lending Club CSV at the repository root; the first run builds a parquet cache in `.cache/`):
```
venv/bin/python scripts/train_deploy.py        # about 10-20 minutes; writes models/deploy/ (bundle.joblib, metadata.json, model_card.md, ...)
venv/bin/python scripts/train_deploy.py --calibrator platt --out models/deploy_platt   # Platt instead of isotonic
```
`models/deploy/` is committed, so training is only needed to reproduce the bundle. The bundle stores the fitted pipeline,
calibrator, EBM, feature list and dtypes, training prevalence, library versions, training date range, git commit and timestamp;
`creditrisk.serving.load_bundle` validates schema, hash and library versions and fails loudly on any mismatch (the app refuses to
start without a valid bundle).

Run the app:
```
venv/bin/python app.py                          # http://127.0.0.1:5000  (/predict, /api/predict, /dashboard, /model-results, /health)
venv/bin/gunicorn "app:create_app()"            # production (gunicorn is in requirements-docker.txt, not in the research lock file)
curl -s localhost:5000/api/predict -H 'Content-Type: application/json' -d @example.json
```
Environment variables: `CREDIT_DEPLOY_DIR` (bundle directory), `BORROWER_SHOW_RATE` (default false), `HOST`, `PORT`.

Run the tests (39 tests: library unit tests, Flask route tests and the serving-parity test of H24; the parity test needs
`models/deploy/`, which is committed):
```
venv/bin/python -m pip install -r requirements-dev.txt   # pytest
venv/bin/python -m pytest
```

Docker: `docker compose up --build` serves the app on http://localhost:8000 with a healthcheck on `/health`. The image contains
`models/deploy/`, the templates, the `creditrisk` package and the cached results, and excludes the dataset, caches and virtualenvs.

**Recalibration and monitoring.** Calibration is worse out of time. The deployed configuration's expected calibration error (ECE)
on the new year was 0.0344 / 0.0179 / 0.0387 for test years 2013 / 2014 / 2015 (36-month loans in 2015), against in-time ECE of
0.0141 / 0.0074 / 0.0067 (absolute values; the ratio between them is fold-dependent, H25 not supported). Refit the calibrator (or
retrain) on recent labelled loans at least annually, and monitor mean predicted PD against the realised default rate per tier.
Labels for a vintage mature slowly (36-60 months), so a live system will always calibrate on partly resolved data.

**Limitations.** Research prototype only; no fairness, stability or regulatory validation. Funded loans only (rejected applicants
unobserved); vintages 2010-2015 only; in-time test split. `interest_rate` is endogenous. Reason-code wording is descriptive, not
causal, and whether it satisfies adverse-action-notice requirements is a legal question outside the study. See
`models/deploy/model_card.md` for the model card.

## Disclaimer
This is research code and an illustrative prototype. It has not been validated for lending, fairness, adverse-action-notice or any
other regulatory use, and nothing here is financial or legal advice. Results describe Lending Club loans issued 2007-2018 and two
public benchmark datasets; they may not transfer to other lenders or periods.

## Citation
See `CITATION.cff`.
