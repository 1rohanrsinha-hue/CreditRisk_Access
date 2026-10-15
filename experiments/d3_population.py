"""Phase 3B / task A: D3 (mature vintages) population report + D1/D2/D3 comparison."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from creditrisk.config import RESULTS_DIR
from creditrisk.data import (full_frame, issue_year, term_months, profit_p1, profit_p2,
                             load_first_n, label_legacy)
from creditrisk.repeats import d3_population, d3_sample, get_dataset

pop = d3_population()
pop = pop.assign(year=issue_year(pop).values, term_m=term_months(pop).values, p1=profit_p1(pop).values,
                 p2_3=profit_p2(pop, 0.03).values)
print(f"D3 population: {len(pop):,} strict-labelled loans; default rate {pop.default.mean():.4f}; "
      f"mean P1 ${pop.p1.mean():,.0f}; mean P2(3%) ${pop.p2_3.mean():,.0f}")
vt = (pop.groupby(['year', 'term_m']).agg(n=('default', 'size'), default_rate=('default', 'mean'),
                                          mean_p1=('p1', 'mean'), mean_p2_3pct=('p2_3', 'mean')).reset_index())
vt['share_of_population'] = vt.n / len(pop)
vt.round(4).to_csv(RESULTS_DIR / "phase3b_d3_population.csv", index=False)
print(vt.round(3).to_string(index=False))
smp = d3_sample()
print(f"\nD3 dev sample: {len(smp):,} rows, default rate {smp.default.mean():.4f}")

# D1 / D2 / D3 comparison
rows = []
d1 = label_legacy(load_first_n(100_000))
d2X, d2y, d2p = get_dataset('D2'); F = full_frame().loc[d2X.index]
d2yr, d2tm = issue_year(F).values, term_months(F).values
d3 = smp.assign(year=issue_year(smp).values, term_m=term_months(smp).values)
rows.append(dict(dataset='D1', rows=len(d1), sampling='first 100k CSV rows', label='legacy 5-status (Current = 0)',
                 features='legacy 16 incl. grade-derived credit_score', issue_years='Oct–Dec 2015 (one quarter)',
                 prevalence=d1.default.mean(), share_36m=np.nan, share_2016plus=0.0, purpose='replication of conference paper'))
rows.append(dict(dataset='D2', rows=len(d2y), sampling='seeded random 200k of labelled rows', label='strict (FP vs CO/Default)',
                 features='headline (fico mean, interest_rate, ordinal emp_length)', issue_years='2007–2018',
                 prevalence=d2y.mean(), share_36m=(d2tm == 36).mean(), share_2016plus=(d2yr >= 2016).mean(),
                 purpose='sensitivity (censored vintages included)'))
rows.append(dict(dataset='D3', rows=len(d3), sampling=f'seeded random 200k of {len(pop):,} mature loans', label='strict',
                 features='headline', issue_years='2010–2013 any term; 2014–2015 36m',
                 prevalence=d3.default.mean(), share_36m=(d3.term_m == 36).mean(), share_2016plus=0.0,
                 purpose='HEADLINE'))
C = pd.DataFrame(rows); C.round(4).to_csv(RESULTS_DIR / "phase3b_dataset_comparison.csv", index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60)
print(C.round(3).to_string(index=False))
