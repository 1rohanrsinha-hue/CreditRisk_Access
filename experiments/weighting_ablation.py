"""Phase 2 / task C: weighting x threshold ablation (+ RF/ET min_samples_leaf).

  --stage score   : compute + checkpoint raw OOF/test scores (resumable, parallel
                    via --worker i --n-workers n)
  --stage analyze : build experiments/results/phase2_weighting*.csv
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from creditrisk.config import RESULTS_DIR
from creditrisk.models import MODEL_NAMES
from creditrisk.repeats import run_scoring
from creditrisk.study import analyse
from creditrisk.stats import summarize

METRICS = ['Recall', 'Precision', 'F1', 'MCC', 'ROC_AUC', 'PR_AUC', 'Brier', 'threshold']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["score", "analyze", "leaf"], required=True)
    ap.add_argument("--dataset", choices=["D1", "D2"], required=True)
    ap.add_argument("--repeats", type=int, default=None, help="default D2=10, D1=5")
    ap.add_argument("--leaf-sweep", action="store_true")
    ap.add_argument("--leaf-repeats", type=int, default=3)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--n-jobs", type=int, default=2)
    a = ap.parse_args()
    n_rep = a.repeats or (10 if a.dataset == "D2" else 5)
    reps = list(range(n_rep))
    if a.stage == "score":
        t0 = time.time()
        if a.leaf_sweep:
            run_scoring(a.dataset, list(range(a.leaf_repeats)), models=['Random Forest', 'Extra Trees'],
                        leaves=(5, 20), n_jobs=a.n_jobs, worker=(a.worker, a.n_workers))
        else:
            run_scoring(a.dataset, reps, n_jobs=a.n_jobs, worker=(a.worker, a.n_workers))
        print(f"[done] scoring {a.dataset} worker {a.worker}/{a.n_workers} in {time.time()-t0:.0f}s")
        return
    if a.stage == "leaf":
        R, C = analyse(a.dataset, list(range(a.leaf_repeats)), tag="_leaf",
                       models=['Random Forest', 'Extra Trees'], leaves=(None, 5, 20), with_profit=False)
        x = R[R.rule.isin(['fixed_0.5', 'youden']) & (R.calib == 'raw')]
        piv = x.pivot_table(index=['model', 'weighting', 'leaf', 'rep'], columns='rule',
                            values=['Recall', 'threshold', 'Precision', 'F1']).reset_index()
        piv.columns = ['_'.join([c for c in col if c]) for col in piv.columns]
        c = C[C.calib == 'raw'][['model', 'weighting', 'leaf', 'rep', 'ROC_AUC', 'Brier', 'frac_score0', 't_star']]
        m = piv.merge(c, on=['model', 'weighting', 'leaf', 'rep'])
        m['leaf'] = m.leaf.replace({0: 1})
        out = summarize(m, ['model', 'weighting', 'leaf'],
                        ['Recall_fixed_0.5', 'Recall_youden', 'threshold_youden', 'F1_youden',
                         'ROC_AUC', 'Brier', 'frac_score0'])
        out.to_csv(RESULTS_DIR / "phase2_leaf_sweep.csv", index=False)
        print(out.round(3).to_string(index=False))
        return
    # ---- analyze
    R, _ = analyse(a.dataset, reps, with_profit=(a.dataset == "D2"))
    R = R[R.rule.isin(['fixed_0.5', 'youden']) & (R.calib == 'raw')].copy()
    R['thr_rule'] = R.rule.map({'fixed_0.5': '0.5', 'youden': 'tuned'})
    S = summarize(R, ['dataset', 'model', 'weighting', 'thr_rule'], METRICS)
    S.to_csv(RESULTS_DIR / f"phase2_weighting_{a.dataset}.csv", index=False)
    # spread across models in each cell (per repeat, then mean/sd across repeats)
    sp = (R.groupby(['dataset', 'weighting', 'thr_rule', 'rep'])
           .agg(recall_spread=('Recall', lambda v: v.max() - v.min()),
                auc_spread=('ROC_AUC', lambda v: v.max() - v.min()),
                f1_spread=('F1', lambda v: v.max() - v.min()),
                recall_mean=('Recall', 'mean')).reset_index())
    SP = summarize(sp, ['dataset', 'weighting', 'thr_rule'],
                   ['recall_spread', 'auc_spread', 'f1_spread', 'recall_mean'])
    SP.to_csv(RESULTS_DIR / f"phase2_weighting_spread_{a.dataset}.csv", index=False)
    print(SP.round(3).to_string(index=False))
    print(S[S.thr_rule == '0.5'].pivot(index='model', columns='weighting', values='Recall_mean').round(3))


if __name__ == "__main__":
    main()
