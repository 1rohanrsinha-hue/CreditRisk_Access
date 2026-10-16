"""Train and persist the deployment bundle (tuned LightGBM + OOF calibration + EBM companion).
    python scripts/train_deploy.py [--calibrator isotonic|platt] [--out models/deploy] [--no-ebm]
Data: the mature-vintage population D3 (667,024 strict-labelled loans: 2010-2013 any term + 2014/2015
36-month; it contains the 200k D3 development sample). 80/20 stratified split (seed 42): the model is
fit on the 80% (OOF calibration inside it) and every reported metric is on the untouched 20%.
Requires the cached dataset (.cache/lc_full.parquet; built by creditrisk.data.build_cache)."""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import train_test_split
from creditrisk.calibration import ece, reliability_curve
from creditrisk.features import build_frame
from creditrisk.repeats import d3_population
from creditrisk import serving as S
from creditrisk.stats import bootstrap_ci
from creditrisk.tiers import tier_report


def ci_row(name, fn, n):
    m, lo, hi = bootstrap_ci(fn, n, n_boot=1000, seed=0)
    return dict(metric=name, value=fn(np.arange(n)), ci_lo=lo, ci_hi=hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrator", default="isotonic", choices=["isotonic", "platt"])
    ap.add_argument("--out", default=str(S.DEPLOY_DIR)); ap.add_argument("--no-ebm", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=4)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    raw = d3_population().sort_index()
    X, y, meta = build_frame(raw, True, "fico")
    idx = np.arange(len(X))
    tr, te = train_test_split(idx, test_size=0.20, random_state=S.SEED, stratify=y.values)
    print(f"[train] population {len(X)}  train {len(tr)}  held-out test {len(te)}  prevalence {y.mean():.4f}", flush=True)
    bundle, metadata = S.fit_bundle(X.iloc[tr], y.iloc[tr], meta.iloc[tr], a.calibrator, a.n_jobs, fit_ebm=not a.no_ebm,
                                    log=lambda m: print(m, flush=True))
    # ---- held-out metrics (in memory, before persistence)
    Xte, yte = X.iloc[te], y.values[te]
    p_cal, p_raw = S.predict_frame(bundle, Xte)
    rows = []
    for tag, p in (("calibrated PD", p_cal), ("raw score", p_raw)):
        rows += [dict(model="LightGBM (tuned)", scores=tag, **ci_row("ROC-AUC", lambda i, p=p: roc_auc_score(yte[i], p[i]), len(te))),
                 dict(model="LightGBM (tuned)", scores=tag, **ci_row("Brier", lambda i, p=p: brier_score_loss(yte[i], p[i]), len(te))),
                 dict(model="LightGBM (tuned)", scores=tag, **ci_row("ECE", lambda i, p=p: ece(yte[i], p[i]), len(te)))]
    rows.append(dict(model="LightGBM (tuned)", scores="calibrated PD", **ci_row("PR-AUC", lambda i: average_precision_score(yte[i], p_cal[i]), len(te))))
    rows.append(dict(model="LightGBM (tuned)", scores="calibrated PD", **ci_row("LogLoss", lambda i: log_loss(yte[i], np.clip(p_cal[i], 1e-6, 1 - 1e-6)), len(te))))
    if bundle['ebm'] is not None:
        p_ebm = bundle['ebm'].predict_proba(Xte)
        rows.append(dict(model="EBM (companion)", scores="raw probability", **ci_row("ROC-AUC", lambda i: roc_auc_score(yte[i], p_ebm[i]), len(te))))
        rows.append(dict(model="EBM (companion)", scores="raw probability", **ci_row("ECE", lambda i: ece(yte[i], p_ebm[i]), len(te))))
    M = pd.DataFrame(rows); M.round(5).to_csv(out / "test_metrics.csv", index=False)
    T = tier_report(yte, p_cal, bundle['pi']); T.round(5).to_csv(out / "tier_report.csv", index=False)
    pc, oc = reliability_curve(yte, p_cal, 15)
    pd.DataFrame(dict(mean_pd=pc, observed=oc)).round(5).to_csv(out / "reliability_15bins.csv", index=False)
    print(M.round(4).to_string(index=False)); print(T.round(4).to_string(index=False))
    # ---- parity sample: 1,000 held-out loans that are VALID API inputs; in-memory predictions recorded before saving
    inv = {i: k for k, i in S.EMP_MAP.items()}; rng = np.random.RandomState(S.SEED); order = rng.permutation(len(te)); keep, excluded = [], 0
    for j in order:
        r = Xte.iloc[j]; d = {f: r[f] for f in S.INPUT_FIELDS}
        d['employment_length'] = 'not reported' if pd.isna(r['employment_length']) else inv[int(r['employment_length'])]
        d = {k: (float(v) if k not in ('loan_purpose', 'home_ownership', 'employment_length') else v) for k, v in d.items()}
        try:
            S.validate_input(d); keep.append((j, d))
        except S.ValidationError:
            excluded += 1
        if len(keep) == 1000:
            break
    P = pd.DataFrame([d for _, d in keep]); P.insert(0, 'heldout_row', [int(te[j]) for j, _ in keep])
    P['y_true'] = [int(yte[j]) for j, _ in keep]
    P.to_csv(out / "parity_sample.csv", index=False)
    mem = [S.score_one(bundle, d) for _, d in keep]
    pd.DataFrame(dict(probability=[m['probability'] for m in mem], ebm_probability=[m.get('ebm_probability', np.nan) for m in mem],
                      tier_index=[m['tier_index'] for m in mem])).to_csv(out / "parity_expected_in_memory.csv", index=False, float_format='%.17g')
    metadata.update(heldout_n=int(len(te)), parity_excluded_invalid_inputs=excluded, parity_n=len(keep),
                    train_seconds=round(time.time() - t0))
    meta_out = S.save_bundle(bundle, metadata, out)
    # ---- model card
    g = lambda m, s_='calibrated PD', mod='LightGBM (tuned)': M[(M.metric == m) & (M.scores == s_) & (M.model == mod)].iloc[0]
    f = lambda r: f"{r.value:.4f} [{r.ci_lo:.4f}, {r.ci_hi:.4f}]"
    tbl = "\n".join(f"| {r.tier} | {r.definition} | {r.share:.3f} | {r.observed_default_rate:.3f} [{r.obs_ci_lo:.3f}, {r.obs_ci_hi:.3f}] | {r.mean_pd:.3f} | {r.gap_obs_minus_pd:+.4f} [{r.gap_ci_lo:+.4f}, {r.gap_ci_hi:+.4f}] |" for r in T.itertuples())
    card = f"""# Model card: calibrated LightGBM credit-default PD (research prototype)

**Not for credit decisions.** This is a research prototype that accompanies a journal study. It has not been validated for any
lending decision, and whether any output satisfies adverse-action-notice requirements is a legal question outside the study.

## Intended use
Illustrating calibrated probability of default (PD) and review tiers on public Lending Club data. Tiers are *review* tiers, not
accept/reject cutoffs: on mature vintages no threshold rule earned more than approving everyone (study finding H5).

## Data
- Lending Club accepted loans 2007-2018Q4; strict label (Fully Paid = 0, Charged Off/Default = 1).
- Mature population D3: issue years 2010-2013 (any term) and 2014-2015 (36-month only), {metadata['n_train'] + metadata['heldout_n']:,} loans, prevalence {metadata['prevalence']:.4f} (training split).
- Training split {metadata['n_train']:,} loans (issued {metadata['training_issue_date_min']} to {metadata['training_issue_date_max']}); held-out test split {metadata['heldout_n']:,} (stratified random 20%, seed 42).
- Funded loans only: rejected applicants are not observed.

## Model
- Tuned LightGBM (unweighted), hyper-parameters = median over the 5 outer folds of the nested Optuna search (integers rounded): `{json.dumps(metadata['lightgbm_params'])}`.
- Calibration: {metadata['calibrator']} regression fitted on 5-fold out-of-fold scores of the training split (fresh pipeline per fold). Platt is available with `--calibrator platt`.
- Companion: Explainable Boosting Machine (10 interactions); reason codes are its exact additive term contributions (interactions split equally between their two features).
- Features (16, application-time): {', '.join(S.FEATURES)}. `interest_rate` is the lender's own risk price (see limitations). `fico_mean` is the mean of the reported FICO range.

## Held-out performance (test split, 95% bootstrap CIs, 1,000 resamples, in-time)
| metric | value |
|---|---|
| ROC-AUC (calibrated PD) | {f(g('ROC-AUC'))} |
| PR-AUC | {f(g('PR-AUC'))} |
| Brier (calibrated / raw) | {f(g('Brier'))} / {f(g('Brier', 'raw score'))} |
| ECE, 15 equal-frequency bins (calibrated / raw) | {f(g('ECE'))} / {f(g('ECE', 'raw score'))} |
| EBM ROC-AUC / ECE | {f(g('ROC-AUC', 'raw probability', 'EBM (companion)'))} / {f(g('ECE', 'raw probability', 'EBM (companion)'))} |

## Review tiers (calibrated PD anchored on prevalence pi = {metadata['prevalence']:.4f}; edges {metadata['tier_edges_pd'][0]:.4f} and {metadata['tier_edges_pd'][1]:.4f})
| tier | definition | share | observed default rate [95% CI] | mean PD | observed - PD [95% CI] |
|---|---|---|---|---|---|
{tbl}

## Limitations
- **Research prototype, not for credit decisions;** no fairness, stability or regulatory validation.
- **In-time test split.** Out-of-time (Phase 3) AUC fell by about .016 and calibration drifted: raw ECE .041/.022/.034 and isotonic-on-previous-year ECE .022/.036/.022 for test years 2013/2014/2015 (study Phase 3C), versus an in-time calibrated ECE of {g('ECE').value:.4f} for this model, i.e. about {0.022/g('ECE').value:.0f}-{0.036/g('ECE').value:.0f}x higher out of time (5-model means from the study; this exact model was not run out of time). **Recalibrate (refit the calibrator on recent labelled loans) at least annually**, and monitor mean PD against the realised default rate.
- **Funded loans only; vintages 2010-2015 only.** Late vintages (2016+) are censored and excluded; the model has not seen 60-month loans issued after 2013.
- **interest_rate is endogenous** (set by the lender from its own risk assessment); removing it costs about .048 AUC and changes capacity-ranking conclusions. Reason codes shown to borrowers hide it by default (config flag).
- Reason-code wording describes the value relative to the training median; it is not causal.
- Inputs outside the accepted ranges are rejected; inputs outside the 0.1%-99.9% training range are scored with a warning.

## Reproduction
Created {metadata['created_utc']} from commit `{metadata['git_commit']}` (branch {metadata['git_branch']}; working tree dirty at training time: {metadata['git_dirty']}). Library versions: {json.dumps(metadata['versions'])}. Schema hash {metadata['schema_hash']}. Regenerate with `python scripts/train_deploy.py` (about {metadata['train_seconds']//60} min).
"""
    (out / "model_card.md").write_text(card)
    print(json.dumps({k: meta_out[k] for k in ('bundle_bytes', 'n_train', 'prevalence', 'git_commit')}), f"total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
