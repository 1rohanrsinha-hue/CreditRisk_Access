"""Cheap analysis pass over cached scores: calibrators x threshold rules x
test-set metrics (+ profit). Result per (dataset, repeat) is cached as CSV."""
import numpy as np, pandas as pd

from .config import ANALYSIS_DIR
from .models import MODEL_NAMES
from .repeats import get_dataset, load_score, score_path
from .calibration import make_calibrators, apply_cal, prob_metrics
from .evaluate import metrics_at
from . import thresholds as T

RULES = ['fixed_0.5', 'youden', 'f1_max', 'g_mean', 'recall_0.70', 'recall_0.80',
         'prevalence', 'elkan', 'profit_max']


def analyse_repeat(ds, rep, models=None, weightings=('none', 'balanced'), leaves=(None,),
                   with_profit=True):
    """Return (rules_df, cal_df) for one repeat."""
    X, y, profit = get_dataset(ds)
    yv, pv = y.values, profit.values
    rules, cals = [], []
    for model in (models or MODEL_NAMES):
        for w in weightings:
            for leaf in leaves:
                if leaf is not None and model not in ('Random Forest', 'Extra Trees'):
                    continue
                if not score_path(ds, rep, model, w, leaf).exists():
                    continue
                z = load_score(ds, rep, model, w, leaf)
                tv, te = z['idx_tv'], z['idx_te']
                y_tv, y_te, p_tv, p_te = yv[tv], yv[te], pv[tv], pv[te]
                pi = float(y_tv.mean())
                s_oof, s_te = z['oof'].astype(float), z['te'].astype(float)
                cal_objs = make_calibrators(s_oof, y_tv)
                # Elkan costs from train+val only (never test)
                c_fn = float(-p_tv[y_tv == 1].mean()) if with_profit else np.nan
                c_fp = float(p_tv[y_tv == 0].mean()) if with_profit else np.nan
                for cname, cal in cal_objs.items():
                    so, st = apply_cal(cal, s_oof), apply_cal(cal, s_te)
                    base = dict(dataset=ds, rep=rep, model=model, weighting=w,
                                leaf=leaf or 0, calib=cname)
                    pm = prob_metrics(y_te, st)
                    jm, flat = T.youden_flatness(y_tv, so)
                    t_y = T.youden_threshold(y_tv, so)
                    from sklearn.metrics import roc_auc_score
                    cals.append({**base, **pm, 'ROC_AUC': roc_auc_score(y_te, st),
                                 'ROC_AUC_raw': roc_auc_score(y_te, s_te),
                                 't_star': t_y, 'pi': pi, 'J_max_oof': jm, 'flat_width': flat,
                                 'frac_score0': float((s_te <= 1e-9).mean())})
                    thr = {'fixed_0.5': 0.5, 'youden': t_y,
                           'f1_max': T.f1_max_threshold(y_tv, so),
                           'g_mean': T.gmean_threshold(y_tv, so),
                           'recall_0.70': T.recall_target_threshold(y_tv, so, 0.70),
                           'recall_0.80': T.recall_target_threshold(y_tv, so, 0.80),
                           'prevalence': T.prevalence_threshold(y_tv)}
                    if cname != 'raw' and with_profit:
                        thr['elkan'] = T.elkan_threshold(c_fp, c_fn)
                    if with_profit:
                        thr['profit_max'] = T.profit_max_threshold(so, p_tv)
                    for rname, t in thr.items():
                        r = {**base, 'rule': rname, 'threshold': t, **metrics_at(y_te, st, t)}
                        if with_profit:
                            r.update(T.policy_outcome(st, t, y_te, p_te))
                        rules.append(r)
    return pd.DataFrame(rules), pd.DataFrame(cals)


def analyse(ds, reps, tag="", **kw):
    """Cached across repeats: .cache/analysis/{ds}{tag}_r{rep}_{rules|cal}.csv"""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    R, C = [], []
    for rep in reps:
        fr = ANALYSIS_DIR / f"{ds}{tag}_r{rep:02d}_rules.csv"
        fc = ANALYSIS_DIR / f"{ds}{tag}_r{rep:02d}_cal.csv"
        if fr.exists() and fc.exists():
            R.append(pd.read_csv(fr)); C.append(pd.read_csv(fc)); continue
        r, c = analyse_repeat(ds, rep, **kw)
        if len(r):
            r.to_csv(fr, index=False); c.to_csv(fc, index=False)
            R.append(r); C.append(c)
    return pd.concat(R, ignore_index=True), pd.concat(C, ignore_index=True)
