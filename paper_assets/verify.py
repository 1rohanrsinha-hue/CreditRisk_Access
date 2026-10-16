"""Recompute headline numbers from the source CSVs (independently of export_paper_assets.py) and compare them with
(1) the exported table CSVs and (2) numbers quoted in EXTENSION_TRACKER.md. Exit 1 on any table mismatch,
2 if only tracker statements disagree.   python paper_assets/verify.py"""
import re, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "experiments" / "results"; T = ROOT / "paper_assets" / "tables"
TR = (ROOT / "EXTENSION_TRACKER.md").read_text()
rd = lambda n: pd.read_csv(R / n); tb = lambda n: pd.read_csv(T / f"{n}.csv")
table_fail, tracker_fail, n_ok = [], [], 0


def chk(name, got, want, tol=1e-4, kind='table'):
    global n_ok
    ok = bool(abs(float(got) - float(want)) <= tol)
    if ok:
        n_ok += 1
    else:
        (table_fail if kind == 'table' else tracker_fail).append(f"{name}: asset/tracker {got} vs source {want}")
    print(f"[{'ok ' if ok else 'FAIL'}] {name}: {got} vs {want}")


def tracker_has(name, text, source_value, decimals, scale=1.0):
    """text must appear in the tracker and equal the source value rounded to `decimals` (half-unit tolerance)."""
    global n_ok
    present = text in TR
    m = re.search(r"(\d[\d,]*\.?\d*|\.\d+)", text.replace('−', '-'))
    v = float(m.group(0).replace(',', '')) if m else float('nan')
    ok = present and abs(v - abs(source_value * scale)) <= 0.5 * 10 ** -decimals + 1e-9
    if ok:
        n_ok += 1
    else:
        tracker_fail.append(f"{name}: tracker text {text!r} present={present}; source value {source_value} (scaled {source_value*scale:.6g})")
    print(f"[{'ok ' if ok else 'WARN'}] tracker '{name}': {text!r} vs {source_value:.6g}")


# ---------------- tables vs sources
d = rd("phase3b_dataset_comparison.csv").set_index('dataset'); chk("T1 D3 prevalence", tb("T1_datasets").set_index('dataset').loc['D3', 'prevalence'], d.loc['D3', 'prevalence'])
chk("T1b total n of D3 population", tb("T1b_d3_vintage_term").n.sum(), rd("phase3b_d3_population.csv").n.sum(), 0)
b = rd("baseline_table1.csv"); t2 = tb("T2a_table1_reproduction")
for r in t2.itertuples():
    chk(f"T2a {r.model} {r.metric} paper vs reproduction", r.reproduced_legacy, b[(b.variant == 'legacy_fill_bug') & (b.model == r.model)][r.metric].iloc[0], 1e-6)
    chk(f"T2a {r.model} {r.metric} |paper - reproduced| < .0015 (3 d.p.)", min(r.abs_diff_legacy, .0014), r.abs_diff_legacy, 1e-9)
w = rd("phase2_weighting_spread_D3.csv"); w = w[(w.weighting == 'balanced') & (w.thr_rule == '0.5')].recall_spread_mean.iloc[0]
chk("T3 D3 balanced@0.5 recall spread", tb("T3_weighting_x_threshold").query("dataset=='D3' and weighting=='balanced' and threshold=='0.5'")['recall spread (mean ± sd)'].iloc[0].split(' ')[0], w, 6e-4)
c = rd("phase2_calibration_summary_D3.csv"); c = c[(c.weighting == 'balanced') & (c.calib == 'isotonic')].abs_t_minus_pi.iloc[0]
chk("T4a D3 balanced isotonic |t*-pi|", tb("T4a_calibration").query("dataset=='D3' and weighting=='balanced' and calibration=='isotonic'")['|t*−π|'].iloc[0], c, 1e-6)
s = rd("phase3c_stats.csv"); s = s[(s.dataset == 'D3') & (s.metric == 'ROC_AUC') & (s.model == 'XGBoost')].mean_delta.iloc[0]
chk("T6 D3 XGBoost delta AUC", tb("T6_equivalence").query("dataset=='D3' and model=='XGBoost'")['Δ AUC vs best'].iloc[0], s, 1e-6)
p = rd("phase3b_profit_d3.csv"); p = p[(p.model == 'LightGBM') & (p.scenario == 'P2_3%')].set_index('rule').profit_per_1000_mean
t7 = tb("T7a_profit_grid").set_index('rule')
chk("T7a approve_all P2-3% (USD k)", t7.loc['approve_all', 'profit P2_3% (USD k)'], p['approve_all'] / 1000, 1e-4); chk("T7a youden P2-3% (USD k)", t7.loc['youden', 'profit P2_3% (USD k)'], p['youden'] / 1000, 1e-4)
cap = rd("phase3c_capacity.csv"); cap = cap[(cap.dataset == 'D3') & (cap.scenario == 'P2_3%') & (cap.k == .5)]
for rk, cp in (('default_risk', 'dollar'), ('expected_profit', 'dollar'), ('expected_profit', 'count')):
    chk(f"T8a {rk} {cp} gain", tb("T8a_capacity_d3").query("ranking==@rk and capacity==@cp")['profit gain over random (USD/1000)'].iloc[0], cap[(cap.ranking == rk) & (cap.capacity == cp)].gain_profit.mean(), 1e-3)
tv = rd("phase4_term_vintage.csv"); t8c = tb("T8c_term_vintage")
chk("T8c 2013 36m mixed-term gain", t8c[t8c.source.str.startswith('mixed') & (t8c['test year'] == 2013) & (t8c.term.astype(str) == '36')]['gain (USD/1000)'].iloc[0], tv[(tv['T'] == 2013) & (tv.term.astype(str) == '36') & (tv.ranking == 'default_risk') & (tv.k == .5)].gain_per_1000.iloc[0], 1e-3)
t36 = rd("phase5_term36.csv"); chk("T8c 2013 36m refit gain", t8c[t8c.source.str.startswith('36') & (t8c['test year'] == 2013)]['gain (USD/1000)'].iloc[0], t36[(t36['T'] == 2013) & (t36.ranking == 'default_risk') & (t36.k == .5)].gain_per_1000.iloc[0], 1e-3)
ac = rd("phase3c_ablation_cells.csv").set_index('cell'); chk("T9a AUC A", tb("T9a_ablation").set_index('cell').loc['A', 'ROC-AUC'], ac.loc['A', 'ROC_AUC'], 1e-6)
o = rd("phase3b_oot_fold_metrics.csv"); oi = o[o.calib == 'isotonic']
for Tn in (2013, 2014, 2015):
    chk(f"T10a ECE next year isotonic {Tn}", tb("T10a_oot").query("`test year`==@Tn and calibration=='isotonic'").ECE_next_year.iloc[0], oi[oi['T'] == Tn].ECE_test.mean(), 1e-6)
op = rd("phase3d_optuna_summary.csv").set_index('model'); chk("T12d RF dAUC", tb("T12d_tuning").set_index('model').loc['Random Forest', 'Δ AUC'], op.loc['Random Forest', 'dAUC'], 1e-6)
dm = rd("phase5_deploy_test_metrics.csv"); chk("T12e deployment AUC", tb("T12e_deployment_model").query("metric=='ROC-AUC' and scores=='calibrated PD'").value.iloc[0], dm.query("metric=='ROC-AUC' and scores=='calibrated PD'").value.iloc[0], 1e-6)
dep = rd("phase6_deploy_oot.csv").query("scope=='protocol'")
for r in dep.itertuples():
    chk(f"T12g ECE ratio {r.T}", tb("T12g_deploy_oot").query("`test year`==@r.T").ECE_ratio.iloc[0], r.ECE_ratio, 1e-4)
    seeds = rd("phase6_deploy_oot_seeds.csv").query("T==@r.T and scope=='protocol'").groupby('set').ECE.mean()
    chk(f"H25 ECE ratio {r.T} recomputed from seed-level ECE", r.ECE_ratio, seeds['oot'] / seeds['in_time'], 2e-2)
rc = rd("phase4_reasoncodes.csv").query("rep=='pooled' and ebm=='EBM' and k==3 and version=='rate_allowed'").share_overlap_ge2.iloc[0]
chk("T13c H22 share", tb("T13c_reason_codes").query("version=='rate_allowed' and k==3")['share with >=2 shared codes'].iloc[0], rc, 1e-6)
t14 = tb("T14_hypothesis_scorecard"); chk("T14 number of hypotheses", len(t14), 25, 0); chk("T14 H25 verdict present", int((t14.hypothesis == 'H25').sum()), 1, 0)
for ext in ('csv', 'tex'):
    files = sorted(T.glob(f"*.{ext}")); chk(f"{ext} files exported", len(files), len(list(T.glob('*.csv'))), 0)
bad = [f.name for f in T.glob('*.tex') if re.search(r'(^|& )nan( &|\s*\\\\|$)', f.read_text(), re.M)]
chk("no 'nan' in LaTeX tables", len(bad), 0, 0)
cl = pd.read_csv(ROOT / "paper_assets" / "claims_parsed_from_tracker.csv"); chk("T15 rows = parsed claims", len(tb("T15_claims")), len(cl.drop_duplicates(subset=['claim', 'tracker_section'])), 0)
for ext in ('png', 'pdf'):
    chk(f"figures exported ({ext})", len(list((ROOT / 'paper_assets' / 'figures').glob(f'*.{ext}'))), len(list((ROOT / 'paper_assets' / 'figures').glob('*.png'))), 0)

# ---------------- tracker statements vs sources
tracker_has("H11 delta AUC A-F", ".0483", ac.loc['A', 'ROC_AUC'] - ac.loc['F', 'ROC_AUC'], 4, 1)
tracker_has("H11 cap gain A", "−$39.1k", ac.loc['A', 'gain_dollar_k50'] / 1000, 1)
tracker_has("H11 cap gain F", "+$49.3k", ac.loc['F', 'gain_dollar_k50'] / 1000, 1)
capall = rd("phase3c_capacity.csv"); capall = capall[(capall.dataset == 'D3') & (capall.scenario == 'P2_3%')]
ep = lambda cp: capall[(capall.ranking == 'expected_profit') & (capall.capacity == cp)].gain_profit.mean() / 1000     # mean over k = 30..90%
tracker_has("H12 count gain (mean over k=30-90%)", "$191.6k", ep('count'), 1); tracker_has("H12 dollar gain (mean over k=30-90%)", "$102.5k", ep('dollar'), 1)
h20 = rd("phase4_h20.csv"); tracker_has("H20 diff 2014-2013", "+$36.1k", h20.point.iloc[0] / 1000, 1); tracker_has("H20 diff 2015-2013", "+$87.6k", h20.point.iloc[1] / 1000, 1)
h23 = rd("phase5_h23.csv"); tracker_has("H23 diff 2014-2013", "+$32.8k", h23.point.iloc[0] / 1000, 1); tracker_has("H23 diff 2015-2013", "+$75.5k", h23.point.iloc[1] / 1000, 1)
tracker_has("H16 dAUC CatBoost", "+.0023", op.loc['CatBoost', 'dAUC'], 4); tracker_has("H16 dAUC RF", "+.0224", op.loc['Random Forest', 'dAUC'], 4)
tracker_has("H22 rate allowed", ".989", rc, 3); tracker_has("OOT ECE next year 2014", ".036", oi[oi['T'] == 2014].ECE_test.mean(), 3)
tracker_has("deployment ROC-AUC", ".6898", dm.query("metric=='ROC-AUC' and scores=='calibrated PD'").value.iloc[0], 4)
gap = (o[o.calib == 'isotonic'].groupby('T').apply(lambda g: (g.AUC_intime - g.AUC_oot).mean(), include_groups=False))
print("OOT AUC drop by fold (iso, mean of 5 models x 3 seeds):", gap.round(4).to_dict())
st = rd("phase3c_stats.csv"); xg = st[(st.dataset == 'D3') & (st.metric == 'ROC_AUC') & (st.model == 'Gradient Boosting')].mean_delta.iloc[0]; tracker_has("H9 GB delta", "−.0005", xg, 4)
# the H4 statement 'J lost <= .007' is checked against the cell-level maximum
jmax = oi.J_loss.max(); m = re.search(r"J lost ≤ \.007", TR)
if m and jmax > .007 and "Phase 6 correction" not in TR:
    tracker_fail.append(f"H4 'J lost ≤ .007': the worst single (fold, model, seed) cell loses {jmax:.4f} ({(oi.J_loss/oi.J_test_max).max():.1%} of J_max); fold means are {oi.groupby('T').J_loss.mean().round(4).to_dict()}")

print(f"\n{n_ok} checks passed; {len(table_fail)} table mismatches; {len(tracker_fail)} tracker discrepancies")
for f in table_fail: print("TABLE MISMATCH:", f)
for f in tracker_fail: print("TRACKER DISCREPANCY:", f)
sys.exit(1 if table_fail else (2 if tracker_fail else 0))
