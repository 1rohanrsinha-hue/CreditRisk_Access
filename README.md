# Calibration, Base Rates and Decision Thresholds in Credit-Risk Classification

**Research prototype. Not for credit decisions.** This repository holds the code, cached results and paper assets for a
multi-dataset study of ten machine-learning credit-default models: calibration, the Youden-optimal threshold and the
base rate, class weighting, profit-based decisions, drift and explainability. It is the journal extension of a conference
paper; every analysis step, hypothesis (H1-H28) and decision is logged in [`EXTENSION_TRACKER.md`](EXTENSION_TRACKER.md).

Authors: Rohan Sinha and Kaliraj S. Licence: MIT (code). Data have their own terms (see below).

## Branches and tags
| ref | content |
|---|---|
| `journal-extension` (this branch) | study code, results, paper assets, data cards, tracker |
| `app-v2` | Flask app serving one pre-trained, calibrated model with review tiers and reason codes (`git checkout app-v2`) |
| tag `paper-v1-baseline` | the conference-paper app, frozen as submitted |

## Directory map
| path | what it is |
|---|---|
| `creditrisk/` | library: leak-free features and encoders, the 10 models, calibration, thresholds, profit definitions, repeated-split scoring, statistics, external-dataset loaders |
| `experiments/` | one script per study stage; `results/` holds the result CSVs, `figures/` the study figures |
| `paper_assets/` | tables (CSV and LaTeX) and figures (PNG and PDF) for the paper, `MANIFEST.md`, `manuscript_facts.md`, `verify.py` |
| `docs/data_cards/` | data cards for Lending Club, German Credit and Taiwan Default |
| `scripts/` | `reproduce_all.sh` (all stages), `fetch_external.sh` (UCI downloads), `build_manuscript_facts.py` |
| `app.py`, `templates/` | the legacy conference-paper app, kept unchanged for reference (superseded by `app-v2`) |
| `requirements-lock.txt` | pinned research environment (Python 3.13) |

## Setup
```
python3.13 -m venv venv
venv/bin/python -m pip install -r requirements-lock.txt
```
Results were produced with Python 3.13.5 on macOS (arm64, 10 cores); pandas 3.0.2 matters (see `scripts/reproduce_all.sh`).

## Data
| dataset | use | source and licence |
|---|---|---|
| Lending Club accepted loans 2007-2018Q4 (`accepted_2007_to_2018Q4.csv`) | D1, D2, D3 | **Not included in this repository.** Obtained from the Kaggle dataset "All Lending Club loan data" (wordsforthewise); licence not recorded - check the dataset page before reuse. Place the file at the repository root. If you only have the `.csv.gz`, run `python extract_dataset.py` first. |
| German Credit (UCI Statlog) | external check, n = 1,000 | UCI Machine Learning Repository, CC BY 4.0 |
| Default of Credit Card Clients, Taiwan (UCI) | external check, n = 30,000 | UCI Machine Learning Repository, CC BY 4.0 |

Details, file hashes and label coding are in `docs/data_cards/`. The two UCI files are downloaded by `scripts/fetch_external.sh` into `external_data/` (git-ignored).

## Reproduce the study
```
bash scripts/reproduce_all.sh --dry-run     # print the 42 stages and approximate runtimes
bash scripts/reproduce_all.sh               # run everything (many hours, resumable; needs the Lending Club CSV)
bash scripts/reproduce_all.sh --from 38     # start at a stage, e.g. the external datasets
```
Heavy stages checkpoint to `.cache/` and skip finished units. The raw Lending Club CSV is required for the Lending Club stages; the
external-dataset stages (38-41) only need network access to UCI. Without the raw data you can still check the shipped results:
```
venv/bin/python paper_assets/verify.py      # checks every table against the CSVs and the tracker, and recomputes key numbers from cached scores when they are present
```
(Cached scores in `.cache/` are not shipped: the independent recomputation checks that need them print `[skip]` when they are absent, and the table and tracker checks use the committed CSVs.)

## Run the Flask app
The deployable app lives on branch `app-v2`:
```
git checkout app-v2
python scripts/train_deploy.py        # needs the Lending Club CSV; writes models/deploy/ (about 10-20 minutes)
python app.py                         # http://127.0.0.1:5000  (/predict, /api/predict, /dashboard, /model-results, /health)
python -m pytest                      # 39 tests
```
See the README on `app-v2` for the API, input ranges, Docker and the recalibration schedule. The app reports a calibrated probability
of default and one of three review tiers; the tiers are **not** accept/reject cutoffs.

## Disclaimer
This is research code and an illustrative prototype. It has not been validated for lending, fairness, adverse-action-notice or any
other regulatory use, and nothing here is financial or legal advice. Results describe Lending Club loans issued 2007-2018 and two
public benchmark datasets; they may not transfer to other lenders or periods.

## Citation
See `CITATION.cff`.
