"""Phase 3C / task D: calibration under drift. Fixes evaluated on OOT folds."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from creditrisk.config import RESULTS_DIR, FIGURES_DIR
from creditrisk.calibration import Isotonic, Platt, prob_metrics
from creditrisk.plotting import style, SERIES, MUTED, INK
import matplotlib.pyplot as plt
import oot_scores as OS
from oot_study import FOLDS

FIXES = ['raw', 'iso_T-1 (baseline)', 'platt_T-1', 'iso_trailing6m', 'iso+prior_EM', 'iso+prior_oracle']


def prior_shift(p, pi_old, pi_new):
    a = pi_new / pi_old; b = (1 - pi_new) / (1 - pi_old)
    return a * p / (a * p + b * (1 - p))


def saerens_em(p, pi_old, iters=200, tol=1e-7):
    pi = pi_old
    for _ in range(iters):
        q = prior_shift(p, pi_old, pi)
        new = q.mean()
        if abs(new - pi) < tol:
            pi = new; break
        pi = new
    return pi


def main():
    t = OS.oot_table(); y, month = t['y'], t['month']
    rows = []
    for T in FOLDS:
        for model in OS.MAIN5:
            for seed in (42, 43, 44):
                z = OS.load(T, model, seed)
                ca, te = z['idx_ca'], z['idx_te']
                s_ca, s_te = z['s_ca'].astype(float), z['s_te'].astype(float)
                y_ca, y_te = y[ca], y[te]
                pi_cal, pi_T = y_ca.mean(), y_te.mean()
                iso = Isotonic().fit(s_ca, y_ca); p_iso = iso.predict(s_te)
                m6 = month[ca] >= 7
                p6 = Isotonic().fit(s_ca[m6], y_ca[m6]).predict(s_te)
                pi_hat = saerens_em(p_iso, pi_cal)
                out = {'raw': s_te, 'iso_T-1 (baseline)': p_iso,
                       'platt_T-1': Platt().fit(s_ca, y_ca).predict(s_te),
                       'iso_trailing6m': p6,
                       'iso+prior_EM': prior_shift(p_iso, pi_cal, pi_hat),
                       'iso+prior_oracle': prior_shift(p_iso, pi_cal, pi_T)}
                for f, p in out.items():
                    pm = prob_metrics(y_te, p)
                    rows.append(dict(T=T, model=model, seed=seed, fix=f, **pm, mean_p_minus_pi=float(p.mean() - pi_T),
                                     pi_cal=pi_cal, pi_test=pi_T, pi_hat=pi_hat if 'EM' in f else np.nan))
    D = pd.DataFrame(rows); D.round(5).to_csv(RESULTS_DIR / "phase3c_drift.csv", index=False)
    pd.set_option('display.width', 250)
    S = D.groupby(['T', 'fix'])[['ECE', 'Brier', 'LogLoss', 'mean_p_minus_pi']].mean()
    base = S.xs('iso_T-1 (baseline)', level='fix').ECE
    S['ECE_reduction_vs_iso%'] = [100 * (1 - S.loc[(T_, f), 'ECE'] / base[T_]) for (T_, f) in S.index]
    print(S.round(4).to_string())
    print("\npi_cal, pi_test, pi_hat (EM) by fold:", D.groupby('T')[['pi_cal', 'pi_test']].first().round(4).to_dict('index'),
          D[D.fix == 'iso+prior_EM'].groupby('T').pi_hat.mean().round(4).to_dict())
    red = S['ECE_reduction_vs_iso%'].unstack(0)
    print("\nECE reduction (%) vs iso-on-T-1:\n", red.round(1).to_string())
    cheap = ['platt_T-1', 'iso_trailing6m', 'iso+prior_EM']
    ok = {f: int((red.loc[f] >= 30).sum()) for f in cheap}
    print("folds with >=30% reduction:", ok, "->", "H14 SUPPORTED" if max(ok.values()) >= 2 else "H14 REJECTED")
    style()
    fig, ax = plt.subplots(figsize=(8.5, 4.2)); w = 0.13
    for i, f in enumerate(FIXES):
        v = [S.loc[(T_, f), 'ECE'] for T_ in FOLDS]
        ax.bar(np.arange(3) + (i - 2.5) * w, v, w * 0.9, color=SERIES[i], label=f)
    ax.set_xticks(range(3)); ax.set_xticklabels([f'T={T_}' for T_ in FOLDS]); ax.set_ylabel('ECE on year T (mean of 5 models × 3 seeds)')
    ax.set_title('Calibration under drift: ECE of recalibration fixes'); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "drift_fixes.png"); plt.close(fig)


if __name__ == "__main__":
    main()
