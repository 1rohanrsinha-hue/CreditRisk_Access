"""Phase 3A / task A: vintage x term audit on the full file (strict label)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from creditrisk.config import RESULTS_DIR, FIGURES_DIR
from creditrisk.data import (full_frame, issue_year, term_months, months_to_last_payment,
                             parse_month, profit_p1, profit_p2, STRICT_DEFAULT, STRICT_GOOD)
from creditrisk.plotting import style, SERIES, MUTED, INK
import matplotlib.pyplot as plt

d = full_frame()
d = d[d.loan_status.notna()].copy()
d['year'] = issue_year(d); d['term_m'] = term_months(d)
d['issue_dt'] = parse_month(d.issue_d)
d['strict'] = d.loan_status.isin(STRICT_GOOD + STRICT_DEFAULT)
strict = d[d.strict].copy()
n_strict = len(strict)
print(f"ALL labelled rows: {len(d):,}; strict-labelled rows: {n_strict:,} "
      f"(Fully Paid {int((strict.loan_status=='Fully Paid').sum()):,}, "
      f"Charged Off {int((strict.loan_status=='Charged Off').sum()):,}, "
      f"Default {int((strict.loan_status=='Default').sum()):,})")
strict['default'] = strict.loan_status.isin(STRICT_DEFAULT).astype(int)
strict['p1'] = profit_p1(strict)
strict['p2_5'] = profit_p2(strict, 0.05)
strict['m_last'] = months_to_last_payment(strict)
strict['early_payoff'] = ((strict.loan_status == 'Fully Paid') & (strict.m_last < 0.9 * strict.term_m)).astype(int)
strict['fp'] = (strict.loan_status == 'Fully Paid').astype(int)

rows = []
for (y, t), g in strict.groupby(['year', 'term_m']):
    a = d[(d.year == y) & (d.term_m == t)]
    last_issue = a.issue_dt.max()
    end_of_term = last_issue + pd.DateOffset(months=int(t))
    rows.append(dict(
        issue_year=int(y), term=int(t), n_all=len(a), n_strict=len(g),
        strict_coverage=len(g) / len(a),
        share_unresolved=1 - len(g) / len(a),
        default_rate=g.default.mean(), mean_p1=g.p1.mean(), mean_p2_5pct=g.p2_5.mean(),
        mean_months_to_last_pymnt=g.m_last.mean(),
        early_payoff_share_of_fp=g.early_payoff.sum() / max(g.fp.sum(), 1),
        last_issue_month=last_issue.strftime('%Y-%m'),
        matured_by_2018Q4=bool(end_of_term <= pd.Timestamp('2018-12-01'))))
A = pd.DataFrame(rows)
# censoring flag: unresolved loans >= 5% of originations OR not yet matured at the data cut
A['censored'] = (A.share_unresolved >= 0.05) | (~A.matured_by_2018Q4 & (A.share_unresolved >= 0.02))
A.round(4).to_csv(RESULTS_DIR / "phase3a_vintage_audit.csv", index=False)
pd.set_option('display.width', 250)
print(A.round(3).to_string(index=False))

style()
fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
for k, t in enumerate((36, 60)):
    a = A[A.term == t]
    axes[0].plot(a.issue_year, a.default_rate * 100, marker='o', ms=4, color=SERIES[k], label=f'{t} months')
    axes[1].plot(a.issue_year, a.share_unresolved * 100, marker='o', ms=4, color=SERIES[k], label=f'{t} months')
    axes[2].plot(a.issue_year, a.mean_p1 / 1000, marker='o', ms=4, color=SERIES[k], label=f'{t} months')
axes[0].set_title('Default rate (strict-labelled loans)'); axes[0].set_ylabel('%')
axes[1].set_title('Unresolved share of originations'); axes[1].set_ylabel('% Current / Late / Grace')
axes[2].set_title('Mean realized profit P1'); axes[2].set_ylabel('$000 per loan')
for ax in axes:
    ax.set_xlabel('issue year'); ax.legend(fontsize=8)
    ax.axvspan(2015.5, 2018.6, color=MUTED, alpha=0.08)
axes[0].text(2015.7, axes[0].get_ylim()[0] + 0.5, '≥2016: dropped\nfrom headline', fontsize=8, color=MUTED, va='bottom')
fig.tight_layout(); FIGURES_DIR.mkdir(exist_ok=True)
fig.savefig(FIGURES_DIR / "vintage_audit.png"); plt.close(fig)
