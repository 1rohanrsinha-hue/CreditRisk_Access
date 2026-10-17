"""Phase 7 / task C: Phase-2 protocol on external datasets (resumable per (dataset, repeat) unit).

  --stage score   : raw OOF + test scores per (repeat, weighting, model) -> .cache/external/
  --stage analyze : calibrators x {0.5, Youden} -> experiments/results/phase3e_external*.csv
H26-H28 are pre-registered in EXTENSION_TRACKER.md (Phase 7 block + clarifications)."""
import argparse, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as sst
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score, f1_score
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from creditrisk.config import CACHE_DIR, RESULTS_DIR
from creditrisk.external import LOADERS, TEST_SIZE, N_REPEATS, EXT_MODELS, fit_ext
from creditrisk.calibration import make_calibrators, apply_cal, prob_metrics, Isotonic
from creditrisk import thresholds as T
from stats_study import nb, holm

BASE_SEED = 42
CACHE = CACHE_DIR / "external"
WEIGHTINGS = ('none', 'balanced')
BOOSTERS = ['LightGBM', 'XGBoost', 'CatBoost', 'Gradient Boosting']
MARGIN = 0.005


def split(y, ds, rep):
    idx = np.arange(len(y))
    tv, te = train_test_split(idx, test_size=TEST_SIZE[ds], random_state=BASE_SEED + rep, stratify=y)
    return tv, te


def spath(ds, rep, model, w):
    d = CACHE / ds; d.mkdir(parents=True, exist_ok=True)
    return d / f"r{rep:02d}__{model.replace(' ', '_')}__{w}.npz"


def score_unit(ds, rep, n_jobs=2):
    X, y, spec = LOADERS[ds]()
    tv, te = split(y, ds, rep); seed = BASE_SEED + rep
    Xtv, ytv = X.iloc[tv], y.iloc[tv]
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for w in WEIGHTINGS:
        for model in EXT_MODELS:
            p = spath(ds, rep, model, w)
            if p.exists():
                continue
            t0 = time.time(); oof = np.zeros(len(tv))
            for a, b in skf.split(Xtv, ytv):
                m = fit_ext(model, Xtv.iloc[a], ytv.iloc[a], spec, w, seed, n_jobs)
                oof[b] = m.predict_proba(Xtv.iloc[b])[:, 1]
            m = fit_ext(model, Xtv, ytv, spec, w, seed, n_jobs)
            te_s = m.predict_proba(X.iloc[te])[:, 1]
            tmp = str(p) + ".tmp.npz"
            np.savez_compressed(tmp, oof=oof, te=te_s, idx_tv=tv, idx_te=te, secs=np.array([time.time() - t0]))
            os.replace(tmp, p)
            print(f"[score] {ds} rep={rep} {model} w={w} {time.time()-t0:.1f}s", flush=True)


def load(ds, rep, model, w):
    z = np.load(spath(ds, rep, model, w)); return {k: z[k] for k in z.files}


def analyse_unit(ds, rep, y):
    rows = []
    for w in WEIGHTINGS:
        for model in EXT_MODELS:
            z = load(ds, rep, model, w)
            y_tv, y_te = y[z['idx_tv']], y[z['idx_te']]; pi = float(y_tv.mean())
            so_raw, st_raw = z['oof'], z['te']
            for cname, cal in make_calibrators(so_raw, y_tv).items():
                so, st = apply_cal(cal, so_raw), apply_cal(cal, st_raw)
                t_star = T.youden_threshold(y_tv, so)
                pm = prob_metrics(y_te, st)
                r = dict(dataset=ds, rep=rep, model=model, weighting=w, calib=cname, pi=pi, t_star=t_star,
                         abs_dev_pi=abs(t_star - pi), abs_dev_half=abs(t_star - 0.5),
                         ROC_AUC=roc_auc_score(y_te, st), ROC_AUC_raw=roc_auc_score(y_te, st_raw),
                         PR_AUC=average_precision_score(y_te, st), Brier=pm['Brier'], ECE=pm['ECE'],
                         n_train=len(y_tv), n_test=len(y_te))
                for lab, t in (('0.5', 0.5), ('tuned', t_star)):
                    pr = (st >= t).astype(int)
                    r[f'Recall_{lab}'] = recall_score(y_te, pr, zero_division=0)
                    r[f'Precision_{lab}'] = precision_score(y_te, pr, zero_division=0)
                    r[f'F1_{lab}'] = f1_score(y_te, pr, zero_division=0)
                rows.append(r)
    return pd.DataFrame(rows)


def boot_tstar(ds, y, w='none', n_boot=1000, seed=0):
    """Refit isotonic + Youden on resampled (OOF score, label) pairs of repeat 0."""
    rng = np.random.default_rng(seed); out = []
    for model in EXT_MODELS:
        z = load(ds, 0, model, w); yt, s = y[z['idx_tv']], z['oof']; n = len(yt); pi = float(yt.mean())
        ts = []
        for _ in range(n_boot):
            i = rng.integers(0, n, n)
            if yt[i].min() == yt[i].max():
                continue
            iso = Isotonic().fit(s[i], yt[i])
            ts.append(T.youden_threshold(yt[i], iso.predict(s[i])))
        ts = np.array(ts); lo, hi = np.quantile(ts, [.025, .975])
        out.append(dict(dataset=ds, model=model, weighting=w, pi=pi, t_star_boot_mean=ts.mean(), ci_lo=lo, ci_hi=hi,
                        pi_in_ci=bool(lo <= pi <= hi), n_boot=len(ts)))
    return pd.DataFrame(out)


def tests(ds, C):
    """H28: N-B corrected TOST of boosters vs best (raw AUC, weighting none)."""
    n_te, n_tr = C.n_test.iloc[0], C.n_train.iloc[0]; ratio = n_te / n_tr
    a = C[(C.calib == 'raw') & (C.weighting == 'none')].pivot(index='rep', columns='model', values='ROC_AUC_raw')
    best = a.mean().idxmax(); rows = []
    res = {m: nb((a[m] - a[best]).values, ratio=ratio) for m in EXT_MODELS if m != best}
    for grp, ms in (('booster', [m for m in BOOSTERS if m != best]), ('other', [m for m in EXT_MODELS if m != best and m not in BOOSTERS])):
        if not ms:
            continue
        ph = holm([res[m]['p'] for m in ms]); pt = holm([res[m]['tost_p'] for m in ms])
        for i, m in enumerate(ms):
            r = res[m]; eq = bool(pt[i] < .05)
            ex0 = r['ci_lo'] > 0 or r['ci_hi'] < 0
            verdict = 'EQUIVALENT' if eq else ('NOT EQUIVALENT' if ex0 else 'INCONCLUSIVE')
            rows.append(dict(dataset=ds, group=grp, model=m, best=best, mean_auc=a[m].mean(), best_auc=a[best].mean(),
                             sd_auc=a[m].std(ddof=1), n_rep=len(a), nb_ratio=ratio, **r, p_holm=ph[i], tost_p_holm=pt[i], result=verdict))
    return pd.DataFrame(rows)


def verdicts(ds, C, B, TT):
    n = int(C.n_train.iloc[0] + C.n_test.iloc[0]); out = []
    iso = C[(C.calib == 'isotonic')]
    for w in WEIGHTINGS:
        g = iso[iso.weighting == w].groupby('model').abs_dev_pi.mean()
        ok = int((g <= .02).sum()); mean = float(g.mean())
        out.append(dict(dataset=ds, hyp='H26' + ('' if w == 'none' else ' (secondary, balanced)'), n=n, models_ok=ok, stat=mean,
                        detail=f"mean |t*-pi| over models={mean:.4f}; models with repeat-mean <= .02: {ok}/6",
                        result=('SUPPORTED' if (ok >= 5 and mean <= .02) else 'NOT SUPPORTED') if n >= 20000 else 'n/a (n<20,000: see bootstrap row)'))
    if n < 20000:
        frac = float(B.pi_in_ci.mean())
        out.append(dict(dataset=ds, hyp='H26 (bootstrap, n<20,000)', n=n, models_ok=int(B.pi_in_ci.sum()), stat=frac,
                        detail=f"bootstrap 95% CI of t* contains pi for {int(B.pi_in_ci.sum())}/6 models (fraction {frac:.2f})",
                        result='SUPPORTED' if frac > .5 else 'NOT SUPPORTED'))
    raw = C[C.calib == 'raw']
    t = raw.pivot_table(index=['model', 'rep'], columns='weighting', values='t_star').reset_index()
    t['move'] = (t['none'] - .5).abs() - (t['balanced'] - .5).abs()
    pm = t.groupby('model').move.mean(); c1 = float(t.move.mean()); k1 = int((pm > 0).sum())
    tuned = iso.groupby(['model', 'weighting']).Recall_tuned.mean().unstack()
    gap = float((tuned['balanced'] - tuned['none']).abs().mean()); lim = .04 if n < 5000 else .02
    rt = raw.groupby(['model', 'weighting']).Recall_tuned.mean().unstack(); gap_raw = float((rt['balanced'] - rt['none']).abs().mean())
    ok = (c1 > 0 and k1 >= 5) and gap < lim
    out.append(dict(dataset=ds, hyp='H27', n=n, models_ok=k1, stat=gap,
                    detail=(f"clause1: mean move toward .5 = {c1:+.4f} ({k1}/6 models positive; raw t* none {raw[raw.weighting=='none'].t_star.mean():.3f}, balanced {raw[raw.weighting=='balanced'].t_star.mean():.3f}); "
                            f"clause2: mean |tuned recall bal-none| (isotonic) = {gap:.4f} (limit {lim}); supplementary raw-tuned gap = {gap_raw:.4f}"),
                    result='SUPPORTED' if ok else 'NOT SUPPORTED'))
    b = TT[TT.group == 'booster']
    if len(b):
        if (b.result == 'EQUIVALENT').all():
            res = 'SUPPORTED'
        elif (b.result == 'NOT EQUIVALENT').any():
            res = 'NOT SUPPORTED'
        else:
            res = 'INCONCLUSIVE'
        if n < 20000 and res == 'SUPPORTED':
            res = 'SUPPORTED (n<20,000)'
        out.append(dict(dataset=ds, hyp='H28', n=n, models_ok=int((b.result == 'EQUIVALENT').sum()), stat=float(b.mean_auc.max() - b.best_auc.iloc[0]),
                        detail=f"best={b.best.iloc[0]} (AUC {b.best_auc.iloc[0]:.4f}); " + "; ".join(f"{r.model}: {r.result} (d={r.mean_delta:+.4f}, CI [{r.ci_lo:+.4f},{r.ci_hi:+.4f}], Holm TOST p={r.tost_p_holm:.3f}, Holm p of the difference={r.p_holm:.3f})" for r in b.itertuples()),
                        result=res))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["score", "analyze"], required=True)
    ap.add_argument("--dataset", choices=list(LOADERS), required=True)
    ap.add_argument("--worker", type=int, default=0); ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--n-jobs", type=int, default=2)
    a = ap.parse_args(); ds = a.dataset; reps = range(N_REPEATS[ds])
    if a.stage == "score":
        for rep in reps:
            if rep % a.n_workers == a.worker:
                score_unit(ds, rep, a.n_jobs)
        print(f"[done] {ds} worker {a.worker}/{a.n_workers}", flush=True); return
    _, y, _ = LOADERS[ds](); y = y.values
    C = pd.concat([analyse_unit(ds, r, y) for r in reps], ignore_index=True)
    B = boot_tstar(ds, y); TT = tests(ds, C); V = pd.DataFrame(verdicts(ds, C, B, TT))
    RESULTS_DIR.mkdir(exist_ok=True)
    for name, df in (('', C), ('_tstar_ci', B), ('_tests', TT), ('_verdicts', V)):
        f = RESULTS_DIR / f"phase3e_external{name}_{ds}.csv"; df.to_csv(f, index=False)
    pd.set_option('display.width', 250, 'display.max_colwidth', 400)
    print(V[['hyp', 'result', 'detail']].to_string(index=False))


if __name__ == "__main__":
    main()
