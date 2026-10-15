"""Figures for the out-of-time study (reads phase3a_oot*.csv)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from creditrisk.config import RESULTS_DIR, FIGURES_DIR
from creditrisk.plotting import style, SERIES, MUTED, INK
import matplotlib.pyplot as plt

F = pd.read_csv(RESULTS_DIR / "phase3a_oot_fold_metrics.csv"); R = pd.read_csv(RESULTS_DIR / "phase3a_oot.csv")
iso = F[F.calib == 'isotonic']
models = list(dict.fromkeys(iso.model))
style()
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
ax = axes[0]
for k, m in enumerate(models):
    d = iso[iso.model == m].sort_values('T')
    x = d['T'].values + (k - 2) * 0.12
    ax.vlines(x, d.AUC_oot, d.AUC_intime, color=SERIES[k], lw=1)
    ax.plot(x, d.AUC_intime, 'o', ms=5, color=SERIES[k], mfc='white', label=m)
    ax.errorbar(x, d.AUC_oot, yerr=[d.AUC_oot - d.AUC_oot_lo, d.AUC_oot_hi - d.AUC_oot], fmt='o', ms=5,
                color=SERIES[k], lw=0.8, capsize=0)
ax.set_xticks([2013, 2014, 2015]); ax.set_ylabel('ROC-AUC')
ax.set_title('In-time (open) vs out-of-time (filled)'); ax.legend(fontsize=7, loc='lower left')
ax = axes[1]
g = iso.groupby('T')[['pi_train', 'pi_cal', 'pi_test', 't_star_cal', 't_star_test']].mean()
for col, c, ls, lab in (('pi_train', SERIES[2], '--', 'π train'), ('pi_cal', SERIES[0], '--', 'π year T−1'),
                        ('pi_test', SERIES[1], '--', 'π test year'), ('t_star_cal', SERIES[0], '-', 't* (T−1)'),
                        ('t_star_test', SERIES[1], '-', 't* (oracle, test)')):
    ax.plot(g.index, g[col], marker='o', ms=4, color=c, ls=ls, label=lab)
ax.set_xticks([2013, 2014, 2015]); ax.set_ylabel('prevalence / threshold'); ax.legend(fontsize=7, ncol=2)
ax.set_title('Drift: π moves, t* does not follow')
ax = axes[2]
order = ['approve_all', 'youden', 'f1_max', 'elkan', 'profit_max']
t = (R[(R.calib == 'isotonic') & (R.scenario == 'P1') & R.rule.isin(order)]
     .groupby(['T', 'rule']).profit_per_1000.mean().unstack(0).loc[order] / 1e6)
w = 0.26
for k, Tyr in enumerate([2013, 2014, 2015]):
    ax.bar(np.arange(len(order)) + (k - 1) * w, t[Tyr], w * 0.9, color=SERIES[k], label=str(Tyr))
ax.set_xticks(range(len(order))); ax.set_xticklabels([o.replace('_', '\n') for o in order], fontsize=8)
ax.set_ylabel('P1 profit per applicant ($000)'); ax.legend(fontsize=8)
ax.set_title('P1 profit by rule (thr. from T−1)')
fig.tight_layout(); fig.savefig(FIGURES_DIR / "oot_study.png"); plt.close(fig)
