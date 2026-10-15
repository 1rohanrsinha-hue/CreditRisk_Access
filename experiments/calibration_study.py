"""Phase 2 / task D: calibration + the Youden-threshold theory (H1-H3).
Needs cached scores (run weighting_ablation.py --stage score first)."""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from creditrisk.config import RESULTS_DIR, FIGURES_DIR
from creditrisk.calibration import make_calibrators, apply_cal, reliability_curve
from creditrisk.models import MODEL_NAMES
from creditrisk.repeats import get_dataset, load_score
from creditrisk.stats import summarize, boot_youden
from creditrisk.study import analyse
from creditrisk.plotting import style, SERIES, MUTED, INK
import matplotlib.pyplot as plt

REPR = ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'XGBoost', 'CatBoost', 'Neural Network']
COLS = ['Brier', 'LogLoss', 'ECE', 'ROC_AUC', 'ROC_AUC_raw', 't_star', 'pi', 'dev_pi', 'dev_half',
        'flat_width', 'J_max_oof']


def verdict(frac, strong=0.8, weak=0.5):
    return "SUPPORTED" if frac >= strong else ("PARTIAL" if frac >= weak else "NOT SUPPORTED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["D1", "D2"], default="D2")
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--tol", type=float, default=0.05, help="|t*-target| tolerance for hypotheses")
    a = ap.parse_args()
    ds = a.dataset
    reps = list(range(a.repeats or (10 if ds == "D2" else 5)))
    _, C = analyse(ds, reps, with_profit=(ds == "D2"))
    C['dev_pi'] = C.t_star - C.pi
    C['dev_half'] = C.t_star - 0.5
    S = summarize(C, ['dataset', 'model', 'weighting', 'calib'], COLS)
    # AUC invariance sanity check
    C['auc_change'] = C.ROC_AUC - C.ROC_AUC_raw
    chk = C.groupby('calib').auc_change.agg(['min', 'max']).round(5)
    print("AUC change vs raw by calibrator (Platt must be ~0; isotonic can lose ties):\n", chk)
    # bootstrap CI for t* on the reference repeat (rep 0)
    X, y, _p = get_dataset(ds)
    ci = []
    for model in MODEL_NAMES:
        for w in ('none', 'balanced'):
            z = load_score(ds, 0, model, w)
            y_tv = y.values[z['idx_tv']]
            cals = make_calibrators(z['oof'].astype(float), y_tv)
            for cn, cal in cals.items():
                bs = boot_youden(y_tv, apply_cal(cal, z['oof'].astype(float)), a.n_boot, seed=1)
                ci.append(dict(dataset=ds, model=model, weighting=w, calib=cn,
                               t_star_rep0_boot_lo=np.quantile(bs, .025),
                               t_star_rep0_boot_hi=np.quantile(bs, .975),
                               t_star_rep0_boot_sd=bs.std()))
    S = S.merge(pd.DataFrame(ci), on=['dataset', 'model', 'weighting', 'calib'], how='left')
    S.to_csv(RESULTS_DIR / f"phase2_calibration_{ds}.csv", index=False)

    # ---------------- hypotheses (per-variant means across repeats)
    tol = a.tol
    cal = S[S.calib != 'raw']
    h1 = (cal.dev_pi_mean.abs() <= tol).mean()
    raw_bal = S[(S.calib == 'raw') & (S.weighting == 'balanced')]
    h2 = (raw_bal.dev_half_mean.abs() <= tol).mean()
    # H3: after calibration, t* ~ pi regardless of weighting (both weightings, both calibrators)
    h3 = h1
    gap_raw = (S[(S.calib == 'raw')].pivot(index='model', columns='weighting', values='t_star_mean')
               .assign(g=lambda d: (d['none'] - d['balanced']).abs()).g)
    gap_cal = {c: (S[S.calib == c].pivot(index='model', columns='weighting', values='t_star_mean')
                   .assign(g=lambda d: (d['none'] - d['balanced']).abs()).g) for c in ('platt', 'isotonic')}
    rows = [
        dict(hypothesis='H1 calibrated => t*~pi', frac_within_tol=h1, tol=tol,
             mean_abs_dev=cal.dev_pi_mean.abs().mean(), verdict=verdict(h1)),
        dict(hypothesis='H2 balanced raw => t*~0.5', frac_within_tol=h2, tol=tol,
             mean_abs_dev=raw_bal.dev_half_mean.abs().mean(), verdict=verdict(h2)),
        dict(hypothesis='H3 weighting-invariance after calibration', frac_within_tol=h3, tol=tol,
             mean_abs_dev=cal.dev_pi_mean.abs().mean(), verdict=verdict(h3)),
    ]
    H = pd.DataFrame(rows)
    # compact table: weighting x calib
    T = (S.assign(abs_pi=S.dev_pi_mean.abs(), abs_half=S.dev_half_mean.abs())
          .groupby(['weighting', 'calib'])
          .agg(t_star=('t_star_mean', 'mean'), pi=('pi_mean', 'mean'),
               abs_t_minus_pi=('abs_pi', 'mean'), abs_t_minus_half=('abs_half', 'mean'),
               flat_width=('flat_width_mean', 'mean'), Brier=('Brier_mean', 'mean'),
               ECE=('ECE_mean', 'mean')).reset_index())
    T['dataset'] = ds
    T.to_csv(RESULTS_DIR / f"phase2_calibration_summary_{ds}.csv", index=False)
    H['dataset'] = ds
    H.to_csv(RESULTS_DIR / f"phase2_calibration_hypotheses_{ds}.csv", index=False)
    pd.set_option('display.width', 220)
    print(T.round(3).to_string(index=False)); print(H.round(3).to_string(index=False))
    print("worst deviations from pi after calibration:")
    print(cal.assign(a=cal.dev_pi_mean.abs()).nlargest(8, 'a')[['model', 'weighting', 'calib', 't_star_mean', 'pi_mean', 'a']].round(3).to_string(index=False))
    print("raw-score weighting gap |t*_none - t*_bal| per model:\n", gap_raw.round(3).to_dict())
    for c, g in gap_cal.items():
        print(f"  {c}: mean {g.mean():.3f}")

    # ---------------- figures (reference repeat 0)
    if ds != "D2":
        return
    FIGURES_DIR.mkdir(exist_ok=True)
    style()
    fig, axes = plt.subplots(2, 6, figsize=(14, 5.2), sharex=True, sharey=True)
    for r, w in enumerate(('none', 'balanced')):
        for c, m in enumerate(REPR):
            ax = axes[r, c]
            z = load_score(ds, 0, m, w)
            y_tv, y_te = y.values[z['idx_tv']], y.values[z['idx_te']]
            cals = make_calibrators(z['oof'].astype(float), y_tv)
            ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls='--')
            for k, (cn, col) in enumerate((('raw', SERIES[0]), ('isotonic', SERIES[1]))):
                px, py = reliability_curve(y_te, apply_cal(cals[cn], z['te'].astype(float)), 15)
                ax.plot(px, py, marker='o', ms=3.5, color=col, label='raw' if cn == 'raw' else 'isotonic')
            if r == 0:
                ax.set_title(m)
            if c == 0:
                ax.set_ylabel(f"{'unweighted' if w == 'none' else 'balanced'}\nobserved default rate")
            if r == 1:
                ax.set_xlabel('predicted score')
    axes[0, 0].legend(loc='upper left', fontsize=8)
    fig.suptitle('Reliability on the untouched test set (15 equal-frequency bins), D2, repeat 0', x=0.01, ha='left', fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "reliability_diagrams.png"); plt.close(fig)

    # t* vs pi and 0.5 : one panel per calibration, models on y, colour = weighting
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True, sharex=True)
    ys = np.arange(len(MODEL_NAMES))
    for ax, cn in zip(axes, ('raw', 'platt', 'isotonic')):
        for k, (w, col) in enumerate((('none', SERIES[0]), ('balanced', SERIES[1]))):
            d = S[(S.calib == cn) & (S.weighting == w)].set_index('model').loc[MODEL_NAMES]
            ax.errorbar(d.t_star_mean, ys + (k - 0.5) * 0.18, xerr=d.t_star_sd, fmt='o', ms=4.5,
                        color=col, ecolor=col, elinewidth=1, capsize=0,
                        label='unweighted' if w == 'none' else 'balanced')
        pi = S.pi_mean.mean()
        ax.axvline(pi, color=INK, lw=1, ls='--'); ax.axvline(0.5, color=MUTED, lw=1, ls=':')
        ax.set_title({'raw': 'raw scores', 'platt': 'Platt', 'isotonic': 'isotonic'}[cn])
        ax.set_xlabel("Youden threshold t*")
    axes[0].set_yticks(ys); axes[0].set_yticklabels(MODEL_NAMES); axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    axes[2].text(pi + 0.005, -0.45, 'π (prevalence)', fontsize=8, color=INK)
    axes[2].text(0.505, -0.45, '0.5', fontsize=8, color=MUTED)
    fig.suptitle('Youden t* by calibration (mean ± sd over repeats, D2); dashed = prevalence π, dotted = 0.5', x=0.01, ha='left', fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "tstar_vs_pi.png", bbox_inches='tight'); plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
