"""Phase 6 / task B: export every paper table (CSV + LaTeX booktabs) and figure (PNG 300 dpi + PDF) from the
cached result CSVs, plus paper_assets/MANIFEST.md. Reads only experiments/results/*.csv, EXTENSION_TRACKER.md
(T14 verdict list / T15 claims table) and, for the reliability figure, the cached score files in .cache/scores.
    python experiments/export_paper_assets.py          # then: python paper_assets/verify.py"""
import re, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
R = ROOT / "experiments" / "results"; FIGSRC = ROOT / "experiments" / "figures"
OUT = ROOT / "paper_assets"; TAB = OUT / "tables"; FIG = OUT / "figures"
CAP = TAB / "captioned"
TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True); CAP.mkdir(parents=True, exist_ok=True)
ASSETS = []
TRACKER = (ROOT / "EXTENSION_TRACKER.md").read_text()


def rd(name, **flt):
    d = pd.read_csv(R / name)
    for k, v in flt.items():
        d = d[d[k].isin(v)] if isinstance(v, (list, tuple)) else d[d[k] == v]
    return d


# ------------------------------------------------------------------------------------------ LaTeX writer
_U = {'−': '$-$', 'π': '$\\pi$', 'Δ': '$\\Delta$', '≥': '$\\geq$', '≤': '$\\leq$', '×': '$\\times$', '±': '$\\pm$',
      '≈': '$\\approx$', '→': '$\\to$', '–': '--', '—': '---', '<': '$<$', '>': '$>$', '≠': '$\\neq$', 'µ': '$\\mu$'}


def esc(s):
    s = str(s)
    for a, b in (('\\', '\\textbackslash{}'), ('&', '\\&'), ('%', '\\%'), ('_', '\\_'), ('#', '\\#'), ('$', '\\$'), ('{', '\\{'), ('}', '\\}')):
        s = s.replace(a, b)
    return ''.join(_U.get(ch, ch) for ch in s)


def cell(v, fmt):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ''
    if isinstance(v, (bool, np.bool_)):
        return 'yes' if v else 'no'
    if isinstance(v, (int, np.integer)):
        return f"{v:,}" if abs(v) >= 10000 else str(v)
    if isinstance(v, (float, np.floating)):
        return fmt.format(v)
    return esc(v)


def write_tex(df, path, fmt):
    cols = list(df.columns); spec = []
    for c in cols:
        s = df[c]
        if s.map(lambda v: isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)).all():
            spec.append('r')
        else:
            L = s.astype(str).str.len().max()
            spec.append('p{5.2cm}' if L > 48 else 'l')
    lines = ['\\begin{tabular}{' + ''.join(spec) + '}', '\\toprule', ' & '.join(esc(c) for c in cols) + ' \\\\', '\\midrule']
    for _, r in df.iterrows():
        lines.append(' & '.join(cell(r[c], fmt.get(c, '{:.4f}')) for c in cols) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    Path(path).write_text('\n'.join(lines) + '\n')


def emit(tid, title, df, sources, claim, caption, fmt=None, hyp=''):
    fmt = fmt or {}
    df = df.reset_index(drop=True)
    num = df.select_dtypes(include=[float]).columns
    df.assign(**{c: df[c].round(6) for c in num}).to_csv(TAB / f"{tid}.csv", index=False)
    write_tex(df, TAB / f"{tid}.tex", fmt)
    tab = (TAB / f"{tid}.tex").read_text()
    wide = len(df.columns) > 8
    (CAP / f"{tid}.tex").write_text("\\begin{table}[t]\n\\centering\n\\caption{" + esc(caption) + "}\n\\label{tab:" + tid + "}\n" +
                                    ("\\resizebox{\\textwidth}{!}{%\n" + tab.rstrip('\n') + "}\n" if wide else "\\small\n" + tab) + "\\end{table}\n")
    ASSETS.append(dict(kind='table', id=tid, title=title, sources=sources, claim=claim, caption=caption, hyp=hyp))
    print(f"[table] {tid}  ({len(df)} rows)")


def label_equiv(eq, lo, hi):
    return np.where(eq, 'equivalent', np.where((lo <= 0) & (hi >= 0), 'INCONCLUSIVE', 'not equivalent'))


# ------------------------------------------------------------------------------------------ tables
def t1():
    d = rd("phase3b_dataset_comparison.csv")
    d = d.rename(columns={'purpose': 'role', 'issue_years': 'issue years', 'share_36m': 'share 36m', 'share_2016plus': 'share 2016+'})
    emit("T1_datasets", "Datasets D1/D2/D3", d[['dataset', 'rows', 'sampling', 'label', 'features', 'issue years', 'prevalence', 'share 36m', 'share 2016+', 'role']],
         ["phase3b_dataset_comparison.csv"], "D3 (mature vintages) is the headline population; D1 replicates the conference setting; D2 is a censored sensitivity set",
         "Datasets used in the study.", {'prevalence': '{:.4f}', 'share 36m': '{:.3f}', 'share 2016+': '{:.3f}'})
    p = rd("phase3b_d3_population.csv").rename(columns={'term_m': 'term (months)', 'default_rate': 'default rate', 'mean_p1': 'mean P1 (USD)',
                                                          'mean_p2_3pct': 'mean P2-3% (USD)', 'share_of_population': 'share'})
    p['term (months)'] = p['term (months)'].astype(int)
    emit("T1b_d3_vintage_term", "D3 vintage x term breakdown", p, ["phase3b_d3_population.csv"],
         "D3 = 667,024 mature strict-labelled loans (2010-2013 any term; 2014-2015 36-month)", "Composition of the mature population D3 by issue year and term.",
         {'default rate': '{:.4f}', 'mean P1 (USD)': '{:,.0f}', 'mean P2-3% (USD)': '{:,.0f}', 'share': '{:.4f}'})


def t2():
    b = rd("baseline_table1.csv")
    paper = {('Logistic Regression', 'Accuracy'): .592, ('Logistic Regression', 'Precision'): .269, ('Logistic Regression', 'Recall'): .718,
             ('Logistic Regression', 'F1'): .391, ('Logistic Regression', 'AUC'): .696, ('Logistic Regression', 'Threshold'): .452,
             ('CatBoost', 'AUC'): .703, ('CatBoost', 'Threshold'): .519, ('Random Forest', 'Recall'): .668, ('Random Forest', 'Threshold'): .180}
    rows = []
    for (m, k), pv in paper.items():
        a = b[(b.variant == 'legacy_fill_bug') & (b.model == m)][k].iloc[0]; c = b[(b.variant == 'proper_impute') & (b.model == m)][k].iloc[0]
        rows.append(dict(model=m, metric=k, paper=pv, reproduced_legacy=a, abs_diff_legacy=abs(a - pv), reproduced_proper_impute=c))
    emit("T2a_table1_reproduction", "Conference Table I reproduction", pd.DataFrame(rows), ["baseline_table1.csv (values quoted from the paper as recorded in experiments/baseline_repro.py PAPER)"],
         "Table I reproduces to 3 d.p. once the app's unseeded mutual-information step is seeded", "Paper Table I values (those recorded) versus the reproduction with and without the pandas fill bug.",
         {'paper': '{:.3f}', 'reproduced_legacy': '{:.3f}', 'abs_diff_legacy': '{:.4f}', 'reproduced_proper_impute': '{:.3f}'})
    p = rd("protocol_comparison_all.csv"); rows = []
    for ds, g in p.groupby('dataset'):
        w = g.pivot(index='model', columns='protocol', values='Recall')
        from scipy.stats import spearmanr
        rows.append(dict(dataset=ds, mean_recall_A_test=w['A_test'].mean(), mean_recall_B_val=w['B_val'].mean(), mean_recall_C_oof=w['C_oof'].mean(),
                         mean_threshold_A=g[g.protocol == 'A_test'].Threshold.mean(), mean_threshold_B=g[g.protocol == 'B_val'].Threshold.mean(),
                         mean_threshold_C=g[g.protocol == 'C_oof'].Threshold.mean(),
                         spearman_recall_A_vs_B=spearmanr(w['A_test'], w['B_val'])[0], spearman_recall_A_vs_C=spearmanr(w['A_test'], w['C_oof'])[0]))
    emit("T2b_protocols", "Threshold protocols A/B/C", pd.DataFrame(rows), ["protocol_comparison_all.csv"],
         "Model ranking by tuned recall is unstable across threshold-selection protocols (test-tuned A, validation B, out-of-fold C)",
         "Mean recall and threshold by threshold-selection protocol (A: tuned on test, B: on validation, C: on out-of-fold scores) and rank correlation of model recall across protocols.",
         {c: '{:.3f}' for c in ['mean_recall_A_test', 'mean_recall_B_val', 'mean_recall_C_oof', 'mean_threshold_A', 'mean_threshold_B', 'mean_threshold_C',
                                'spearman_recall_A_vs_B', 'spearman_recall_A_vs_C']})


def t3():
    d = pd.concat([rd(f"phase2_weighting_spread_{x}.csv") for x in ('D1', 'D2', 'D3')])
    d['recall spread (mean ± sd)'] = d.apply(lambda r: f"{r.recall_spread_mean:.3f} ± {r.recall_spread_sd:.3f}", axis=1)
    d['AUC spread (mean ± sd)'] = d.apply(lambda r: f"{r.auc_spread_mean:.3f} ± {r.auc_spread_sd:.3f}", axis=1)
    d['mean recall'] = d.recall_mean_mean
    out = d.rename(columns={'thr_rule': 'threshold', 'n_rep': 'repeats'})[['dataset', 'weighting', 'threshold', 'recall spread (mean ± sd)', 'AUC spread (mean ± sd)', 'mean recall', 'repeats']]
    emit("T3_weighting_x_threshold", "Weighting x threshold 2x2", out, ["phase2_weighting_spread_D1.csv", "phase2_weighting_spread_D2.csv", "phase2_weighting_spread_D3.csv"],
         "Threshold tuning, not class weighting, determines recall; spread across models collapses once thresholds are tuned", "Recall and AUC spread across the 10 models by weighting and threshold rule.",
         {'mean recall': '{:.3f}'}, hyp="H3")


def t4():
    s = pd.concat([rd(f"phase2_calibration_summary_{x}.csv") for x in ('D1', 'D2', 'D3')])
    s = s.rename(columns={'calib': 'calibration', 't_star': 't*', 'abs_t_minus_pi': '|t*−π|', 'abs_t_minus_half': '|t*−0.5|'})[
        ['dataset', 'weighting', 'calibration', 't*', 'pi', '|t*−π|', '|t*−0.5|', 'Brier', 'ECE']]
    emit("T4a_calibration", "Calibration and threshold", s, [f"phase2_calibration_summary_{x}.csv" for x in ('D1', 'D2', 'D3')],
         "After calibration the Youden-optimal threshold equals the prevalence, whatever the weighting; raw balanced scores put t* near 0.5",
         "Youden threshold t*, prevalence pi, Brier and ECE by weighting and calibration (mean over models and repeats).", {'pi': '{:.4f}', 'ECE': '{:.4f}'})
    h = pd.concat([rd(f"phase2_calibration_hypotheses_{x}.csv") for x in ('D1', 'D2', 'D3')])
    emit("T4b_h1_h3", "H1-H3 verdicts", h[['dataset', 'hypothesis', 'frac_within_tol', 'tol', 'mean_abs_dev', 'verdict']], [f"phase2_calibration_hypotheses_{x}.csv" for x in ('D1', 'D2', 'D3')],
         "H1-H3 hold on D1, D2 and D3", "Verdicts for H1 (t*~pi after calibration), H2 (t*~0.5 for balanced raw scores) and H3 (weighting invariance).", hyp="H1-H3")


def t5():
    d = pd.concat([rd("phase2_leaf_sweep.csv").assign(dataset='D2'), rd("phase2_leaf_sweep_D3.csv").assign(dataset='D3')])
    d = d.rename(columns={'Recall_fixed_0.5_mean': 'recall@0.5', 'Recall_youden_mean': 'recall (Youden)', 'threshold_youden_mean': 't* (Youden, raw)',
                          'ROC_AUC_mean': 'ROC-AUC', 'Brier_mean': 'Brier', 'F1_youden_mean': 'F1 (Youden)'})
    emit("T5_leaf_purity", "RF/ET min_samples_leaf sweep", d[['dataset', 'model', 'weighting', 'leaf', 'recall@0.5', 'recall (Youden)', 't* (Youden, raw)', 'F1 (Youden)', 'ROC-AUC', 'Brier']],
         ["phase2_leaf_sweep.csv", "phase2_leaf_sweep_D3.csv"], "RF/ET recall collapse at 0.5 is leaf purity, not class weighting", "Effect of min_samples_leaf on raw probabilities of balanced RF/ET.",
         {c: '{:.3f}' for c in ['recall@0.5', 'recall (Youden)', 't* (Youden, raw)', 'F1 (Youden)', 'ROC-AUC', 'Brier']}, hyp="H2, H8")


def t6():
    s = rd("phase3c_stats.csv", metric='ROC_AUC')
    s = s.assign(label=label_equiv(s.equivalent.values, s.ci_lo.values, s.ci_hi.values))
    out = s.rename(columns={'mean_value': 'AUC', 'mean_delta': 'Δ AUC vs best', 'ci_lo': 'CI low', 'ci_hi': 'CI high', 'tost_p_holm': 'TOST p (Holm)'})[
        ['dataset', 'model', 'best', 'AUC', 'Δ AUC vs best', 'CI low', 'CI high', 'TOST p (Holm)', 'label']]
    emit("T6_equivalence", "Equivalence tests (margin ±0.005 AUC)", out, ["phase3c_stats.csv"],
         "Four to five boosted models are statistically equivalent; 'not equivalent' does not mean 'worse'",
         "Nadeau-Bengio corrected equivalence (TOST, margin 0.005 AUC, Holm) against the best model per dataset.", {'AUC': '{:.4f}', 'TOST p (Holm)': '{:.4f}'}, hyp="H9")


def t7():
    d = rd("phase3b_profit_d3.csv", model='LightGBM')
    order = ['approve_all', 'fixed_0.5', 'profit_max', 'expected_profit', 'elkan', 'f1_max', 'youden', 'recall_0.70', 'recall_0.80']
    sc = ['P1', 'P2_0%', 'P2_2%', 'P2_3%', 'P2_5%', 'P2_8%']
    P = d.pivot(index='rule', columns='scenario', values='profit_per_1000_mean').loc[order, sc] / 1000
    P.columns = [f"profit {c} (USD k)" for c in P.columns]
    A = d.pivot(index='rule', columns='scenario', values='accept_rate_mean').loc[order, sc]
    A.columns = [f"accept {c}" for c in A.columns]
    emit("T7a_profit_grid", "Profit by rule and discount rate (D3, LightGBM)", pd.concat([P, A], axis=1).reset_index(), ["phase3b_profit_d3.csv"],
         "On mature vintages no threshold rule earns more than approving everyone (<=5% discount); Youden forgoes about half of the profit",
         "Mean profit per 1000 applicants (USD thousands) and acceptance rate by decision rule and discount rate, LightGBM, D3 (10 repeats).",
         {**{c: '{:,.0f}' for c in P.columns}, **{c: '{:.3f}' for c in A.columns}}, hyp="H5")
    b = rd("phase3b_profit_d3_bootstrap.csv"); b = b[b['vs'] == 'approve_all']
    g = b.groupby('scenario').apply(lambda x: pd.Series(dict(pairs=len(x), rule_beats_approve_all_CI_lo_gt_0=int((x.ci_lo > 0).sum()), rule_below_CI_hi_lt_0=int((x.ci_hi < 0).sum()))), include_groups=False).reset_index()
    emit("T7b_h5", "H5 verdict: rules vs approve-all (paired bootstrap, repeat 0)", g, ["phase3b_profit_d3_bootstrap.csv"],
         "H5: no (model, rule) pair beats approve-all with CI above 0 at P1/0%/2%/3%/5%; exceptions at 8%", "Number of (model, rule) pairs whose profit exceeds / falls below approve-all with a 95% paired-bootstrap CI excluding 0.", hyp="H5")


def t8():
    c0 = rd("phase3c_capacity.csv", dataset='D3', scenario='P2_3%')
    c = c0[c0.k == 0.5].groupby(['ranking', 'capacity']).agg(gain_profit=('gain_profit', 'mean'), gain_roi=('gain_roi', 'mean'), bad_dollar=('bad_dollar', 'mean')).reset_index()
    c = c.merge(c0.groupby(['ranking', 'capacity']).gain_profit.mean().rename('gain_profit_mean_k').reset_index(), on=['ranking', 'capacity'])
    c['ranking'] = pd.Categorical(c.ranking, ['default_risk', 'expected_profit', 'expected_roi', 'ir_high_first', 'ir_low_first']); c = c.sort_values(['ranking', 'capacity'])
    emit("T8a_capacity_d3", "Capacity-constrained approval, D3, k=50%", c.rename(columns={'gain_profit_mean_k': 'profit gain, mean over k=30-90% (USD/1000)', 'gain_profit': 'profit gain over random (USD/1000)', 'gain_roi': 'ROI gain over random', 'bad_dollar': 'dollar-weighted bad rate'}),
         ["phase3c_capacity.csv"], "Default-risk ranking earns less than random on D3 in-time; expected-profit ranking beats random; dollar capacity halves the count-capacity gain",
         "Profit and ROI gain over random selection at 50% capacity, P2-3%, mean over 5 models and 10 repeats.", {'profit gain over random (USD/1000)': '{:,.0f}', 'ROI gain over random': '{:.4f}', 'dollar-weighted bad rate': '{:.3f}', 'profit gain, mean over k=30-90% (USD/1000)': '{:,.0f}'}, hyp="H7, H12, H13")
    q = rd("phase3d_quick_capacity.csv").groupby(['T', 'ranking', 'k']).agg(gain=('gain_per_1000', 'mean'), ci_lo=('ci_lo', 'mean'), ci_hi=('ci_hi', 'mean'),
                                                                           cells_ci_excludes_0=('excludes_zero', 'sum'), cells=('excludes_zero', 'size')).reset_index()
    emit("T8b_oot_capacity", "OOT capacity gain over random", q.rename(columns={'T': 'test year', 'gain': 'gain (USD/1000)', 'ci_lo': 'CI low (mean)', 'ci_hi': 'CI high (mean)'}),
         ["phase3d_quick_capacity.csv"], "Default-risk ranking: below random in 2013, above random in 2014 and 2015", "Out-of-time capacity gain over random, P2-3%, dollar capacity, mean over 5 models x 3 seeds.",
         {'gain (USD/1000)': '{:,.0f}', 'CI low (mean)': '{:,.0f}', 'CI high (mean)': '{:,.0f}'}, hyp="H7")
    tv = rd("phase4_term_vintage.csv", ranking='default_risk', k=0.5).assign(source='mixed-term fit (phase 4)')
    t36 = rd("phase5_term36.csv", ranking='default_risk', k=0.5).assign(source='36-month refit (phase 5)')
    tvc = pd.concat([tv, t36])[['source', 'T', 'term', 'n_test', 'gain_per_1000', 'ci_lo', 'ci_hi', 'mean_rate', 'default_rate', 'corr_p_rate']]
    tvc = tvc.rename(columns={'T': 'test year', 'gain_per_1000': 'gain (USD/1000)', 'ci_lo': 'CI low', 'ci_hi': 'CI high', 'mean_rate': 'mean rate %', 'default_rate': 'default rate', 'corr_p_rate': 'corr(PD, rate)'})
    emit("T8c_term_vintage", "Default-risk capacity gain by test year and term (H20, H23)", tvc, ["phase4_term_vintage.csv", "phase5_term36.csv"],
         "The sign change of the default-risk capacity gain across vintages is not explained by term mix (H20) and survives a 36-month-only refit (H23); no cause is claimed",
         "Default-risk-ranking dollar-capacity gain over random (P2-3%, k=50%) by test year and term, with mixed-term and 36-month-only fits.",
         {'gain (USD/1000)': '{:,.0f}', 'CI low': '{:,.0f}', 'CI high': '{:,.0f}', 'mean rate %': '{:.2f}', 'default rate': '{:.3f}', 'corr(PD, rate)': '{:.2f}'}, hyp="H20, H23")


def t9():
    c = rd("phase3c_ablation_cells.csv").rename(columns={'mean|t*-pi| calibrated': 'mean |t*−π| (calibrated)', 'max|t*-pi|': 'max |t*−π|', 'gain_dollar_k50': 'default-risk gain k=50% (USD/1000)',
                                                          'ROC_AUC': 'ROC-AUC', 'PR_AUC': 'PR-AUC', 'bad_dollar': 'dollar bad rate', 'label': 'features'})
    emit("T9a_ablation", "Endogenous-feature ablation (6 cells)", c[['cell', 'features', 'ROC-AUC', 'PR-AUC', 'Brier', 'ECE', 'mean |t*−π| (calibrated)', 'max |t*−π|', 'default-risk gain k=50% (USD/1000)', 'roi']],
         ["phase3c_ablation_cells.csv"], "Removing the interest rate and credit score costs about 0.048 AUC and flips the sign of the default-risk capacity gain; t*~pi holds in every cell",
         "Six feature-set cells (interest rate in/out x credit score fico/grade-derived/none), D3, 5 models x 5 repeats.", {'ROC-AUC': '{:.4f}', 'PR-AUC': '{:.4f}', 'Brier': '{:.4f}', 'ECE': '{:.4f}', 'mean |t*−π| (calibrated)': '{:.4f}',
                                                                                                                  'max |t*−π|': '{:.4f}', 'default-risk gain k=50% (USD/1000)': '{:,.0f}', 'roi': '{:.4f}'}, hyp="H10, H11")
    f = rd("phase3c_ablation_features.csv"); f = f.sort_values(['cell', 'share'], ascending=[True, False]).groupby('cell').head(5)
    emit("T9b_ablation_shap", "Top-5 SHAP features per ablation cell (LightGBM)", f.rename(columns={'mean_abs_shap': 'mean |SHAP|', 'share': 'share of mean |SHAP|'}), ["phase3c_ablation_features.csv"],
         "interest_rate carries about a third of attribution; without it income, inquiries and loan amount lead", "Top-5 features by share of mean |SHAP| in each ablation cell (rep 0, 2,000 rows).", {'mean |SHAP|': '{:.4f}', 'share of mean |SHAP|': '{:.3f}'}, hyp="H11")
    s = rd("phase4_shap_with_without_rate.csv")
    emit("T9c_shap_rate", "SHAP with vs without the interest rate", s, ["phase4_shap_with_without_rate.csv"], "Excluding the rate leaves attribution ranks of the other features broadly unchanged (Spearman ~0.83-0.87)",
         "LightGBM and EBM attribution with and without the interest rate.", {'spearman_shared_features': '{:.3f}', 'rate_share_with': '{:.3f}'}, hyp="H21")


def t10():
    o = rd("phase3b_oot_fold_metrics.csv")
    g = o.groupby(['T', 'calib']).agg(AUC_in_time=('AUC_intime', 'mean'), AUC_oot=('AUC_oot', 'mean'), ECE_next_year=('ECE_test', 'mean'), ECE_next_year_min=('ECE_test', 'min'), ECE_next_year_max=('ECE_test', 'max'), ECE_calibration_year=('ECE_cal_insample', 'mean'),
                                      J_lost=('J_loss', 'mean'), pi_cal=('pi_cal', 'mean'), pi_test=('pi_test', 'mean')).reset_index()
    emit("T10a_oot", "Out-of-time performance (rolling origin)", g.rename(columns={'T': 'test year', 'calib': 'calibration'}), ["phase3b_oot_fold_metrics.csv"],
         "Out-of-time AUC falls 0.015-0.021; drift is stated as the absolute ECE on the next year (the ratio to in-sample ECE depends on the fold and on the size of the in-time set; H25 not supported); stale thresholds lose little Youden J (H4 not supported)",
         "Mean over 5 models and 3 seeds: AUC in time and out of time, absolute ECE on the next year (with min-max over the 15 model-seed cells; bootstrap CIs of the deployed configuration are in T12g) vs the calibration-year ECE, and Youden J lost to a stale threshold.",
         {c: '{:.4f}' for c in ['AUC_in_time', 'AUC_oot', 'ECE_next_year', 'ECE_next_year_min', 'ECE_next_year_max', 'ECE_calibration_year', 'J_lost', 'pi_cal', 'pi_test']}, hyp="H4")
    d = rd("phase3c_drift.csv"); e = d.groupby(['fix', 'T']).ECE.mean().unstack('T')
    base = e.loc['iso_T-1 (baseline)']; imp = (1 - e.div(base, axis=1))
    out = e.copy(); out.columns = [f"ECE {c}" for c in e.columns]
    for c in e.columns:
        out[f"ECE reduction vs baseline {c}"] = imp[c]
    out = out.reset_index().rename(columns={'fix': 'recalibration'})
    emit("T10b_drift_fixes", "Drift fixes (ECE on the new year)", out, ["phase3c_drift.csv"],
         "Only trailing-6-month isotonic recalibration helps, and in 1 of 3 folds by >=30% (H14 rejected); prior-shift EM diverges across years",
         "ECE on year T for recalibration strategies (mean over 5 models x 3 seeds) and relative reduction versus isotonic fitted on year T-1.",
         {c: '{:.4f}' for c in out.columns if c.startswith('ECE ') and 'reduction' not in c} | {c: '{:.2%}' for c in out.columns if 'reduction' in c}, hyp="H14")


def t11():
    d = rd("phase3d_resampling.csv", dataset='D3')
    d = d[d.model.isin(['Logistic Regression', 'Random Forest', 'LightGBM', 'XGBoost'])]
    r = d[d.setting == 'Youden (raw)'][['model', 'weighting', 'threshold_mean', 'Recall_mean', 'F1_mean', 'ROC_AUC_mean']]
    r5 = d[d.setting == 'thr0.5 (raw)'][['model', 'weighting', 'F1_mean']].rename(columns={'F1_mean': 'F1 at 0.5'})
    r = r.merge(r5, on=['model', 'weighting']).rename(columns={'weighting': 'method', 'threshold_mean': 't* (raw Youden)', 'Recall_mean': 'recall (Youden)', 'F1_mean': 'F1 (Youden)', 'ROC_AUC_mean': 'ROC-AUC'})
    emit("T11a_resampling", "Resampling vs threshold moving (D3)", r, ["phase3d_resampling.csv"],
         "Undersampling matches unweighted + Youden; SMOTE moves boosted-tree probabilities little and costs AUC; threshold moving is sufficient",
         "Raw-score Youden threshold, recall, F1 and AUC by resampling method (none = unweighted) on D3, 5 repeats.", {c: '{:.3f}' for c in ['t* (raw Youden)', 'recall (Youden)', 'F1 (Youden)', 'F1 at 0.5', 'ROC-AUC']}, hyp="H15")
    t = rd("phase3d_resampling_auc_tests.csv"); t = t.assign(label=label_equiv(t.equivalent.values, t.ci_lo.values, t.ci_hi.values))
    emit("T11b_resampling_tests", "Resampling AUC equivalence vs unweighted", t[['model', 'method', 'mean_delta', 'ci_lo', 'ci_hi', 'tost_p_holm', 'label']].rename(columns={'mean_delta': 'Δ AUC', 'ci_lo': 'CI low', 'ci_hi': 'CI high', 'tost_p_holm': 'TOST p (Holm)'}),
         ["phase3d_resampling_auc_tests.csv"], "RF-undersample, RF-SMOTE, LightGBM-SMOTE and XGBoost-SMOTE are not equivalent to the unweighted baseline (H15 not supported)",
         "TOST equivalence (margin 0.005 AUC) of each resampled model against its unweighted counterpart.", {'Δ AUC': '{:+.4f}', 'CI low': '{:+.4f}', 'CI high': '{:+.4f}', 'TOST p (Holm)': '{:.4f}'}, hyp="H15")


def t12():
    e = rd("phase3d_ensembles.csv").groupby(['method', 'kind'])[['ROC_AUC', 'Brier', 'ECE', 'abs_t_pi']].mean().reset_index().sort_values('ROC_AUC', ascending=False)
    et = rd("phase3d_ensembles_tests.csv"); et = et.assign(label=np.where(et.equivalent, 'equivalent', 'not equivalent'))
    emit("T12a_ensembles", "Ensembles vs single models (D3, 10 repeats)", e.rename(columns={'ROC_AUC': 'ROC-AUC', 'abs_t_pi': 'mean |t*−π|'}), ["phase3d_ensembles.csv"],
         "Voting and stacking add at most 0.0007 AUC over the best single model; 'Ensemble' is not justified as a combination method", "Mean test metrics of single models and ensembles.",
         {'ROC-AUC': '{:.4f}', 'Brier': '{:.4f}', 'ECE': '{:.4f}', 'mean |t*−π|': '{:.4f}'}, hyp="H17")
    emit("T12b_ensemble_tests", "Ensemble equivalence tests", et[['ensemble', 'vs', 'family', 'mean_delta', 'ci_lo', 'ci_hi', 'tost_p_holm', 'label']].rename(columns={'mean_delta': 'Δ AUC', 'ci_lo': 'CI low', 'ci_hi': 'CI high', 'tost_p_holm': 'TOST p (Holm)'}),
         ["phase3d_ensembles_tests.csv"], "Ensembles are equivalent to the best single model and not superior by more than 0.005", "TOST equivalence of soft voting and stacking against the best single model.",
         {'Δ AUC': '{:+.4f}', 'CI low': '{:+.4f}', 'CI high': '{:+.4f}', 'TOST p (Holm)': '{:.4f}'}, hyp="H17")
    it = rd("phase3d_interp_tests.csv")
    emit("T12c_interpretable", "EBM and WoE scorecard vs best booster", it[['cell', 'model', 'best_boosting', 'model_auc', 'best_auc', 'mean_delta', 'ci_lo', 'ci_hi', 'tost_p_holm', 'label']].rename(columns={'mean_delta': 'Δ AUC', 'ci_lo': 'CI low', 'ci_hi': 'CI high', 'tost_p_holm': 'TOST p (Holm)'}),
         ["phase3d_interp_tests.csv"], "EBM is equivalent to the best booster; the WoE scorecard is at least 0.005 AUC below", "EBM and WoE scorecard against the best boosting model, with and without the interest rate.",
         {'model_auc': '{:.4f}', 'best_auc': '{:.4f}', 'Δ AUC': '{:+.4f}', 'CI low': '{:+.4f}', 'CI high': '{:+.4f}', 'TOST p (Holm)': '{:.4f}'}, hyp="H18")
    o = rd("phase3d_optuna_summary.csv")
    emit("T12d_tuning", "Tuned vs default hyperparameters (nested CV)", o[['model', 'AUC_default', 'AUC_tuned', 'dAUC', 'dAUC_lo', 'dAUC_hi', 'dBrier', 'abs_t_pi_tuned']].rename(columns={'dAUC': 'Δ AUC', 'dAUC_lo': 'CI low', 'dAUC_hi': 'CI high'}),
         ["phase3d_optuna_summary.csv"], "Tuning moves boosters by at most 0.0023 AUC; the random forest gains 0.022", "Nested-CV Optuna tuning (5 outer folds, 25 trials) against defaults.",
         {c: '{:.4f}' for c in ['AUC_default', 'AUC_tuned', 'Δ AUC', 'CI low', 'CI high', 'dBrier', 'abs_t_pi_tuned']}, hyp="H16")
    m = rd("phase5_deploy_test_metrics.csv")
    emit("T12e_deployment_model", "Deployment model held-out metrics", m.rename(columns={'value': 'value', 'ci_lo': 'CI low', 'ci_hi': 'CI high'}), ["phase5_deploy_test_metrics.csv (copied from app-v2 models/deploy/test_metrics.csv)"],
         "The deployed tuned, isotonic-calibrated LightGBM discriminates and calibrates in time; the EBM companion is equivalent in AUC", "Held-out (in-time) metrics of the deployment model on 133,405 loans with 95% bootstrap CIs.",
         {'value': '{:.4f}', 'CI low': '{:.4f}', 'CI high': '{:.4f}'}, hyp="H24")
    tr = rd("phase5_deploy_tier_report.csv")
    emit("T12f_deployment_tiers", "Review tiers on the held-out split", tr[['tier', 'definition', 'n', 'share', 'observed_default_rate', 'obs_ci_lo', 'obs_ci_hi', 'mean_pd', 'gap_obs_minus_pd', 'gap_ci_lo', 'gap_ci_hi']],
         ["phase5_deploy_tier_report.csv"], "Review tiers anchored on prevalence are calibrated in time", "Share, observed default rate and calibration gap per review tier.",
         {c: '{:.4f}' for c in ['share', 'observed_default_rate', 'obs_ci_lo', 'obs_ci_hi', 'mean_pd', 'gap_obs_minus_pd', 'gap_ci_lo', 'gap_ci_hi']})
    if (R / "phase6_deploy_oot.csv").exists():
        d = rd("phase6_deploy_oot.csv", scope='protocol')
        emit("T12g_deploy_oot", "Deployment configuration out of time (H25)", d[['T', 'n_in_time', 'n_oot', 'AUC_in_time', 'AUC_oot', 'AUC_gap', 'AUC_gap_ci_lo', 'AUC_gap_ci_hi', 'ECE_in_time', 'ECE_oot', 'ECE_oot_ci_lo', 'ECE_oot_ci_hi', 'ECE_ratio', 'ratio_ci_lo', 'ratio_ci_hi']].rename(columns={'T': 'test year'}),
             ["phase6_deploy_oot.csv"], "Out-of-time ECE of the deployed configuration is .018-.039 (absolute, with bootstrap CIs); the ratio to in-time ECE is fold-dependent (2.4/2.4/5.8) and H25 is not supported", "Deployed configuration on rolling-origin folds: in-time vs out-of-time AUC and ECE; absolute out-of-time ECE with 95% bootstrap CIs; the ratio is fold-dependent.",
             {c: '{:.4f}' for c in ['AUC_in_time', 'AUC_oot', 'AUC_gap', 'AUC_gap_ci_lo', 'AUC_gap_ci_hi', 'ECE_in_time', 'ECE_oot', 'ECE_oot_ci_lo', 'ECE_oot_ci_hi']} | {c: '{:.1f}' for c in ['ECE_ratio', 'ratio_ci_lo', 'ratio_ci_hi']}, hyp="H25")


def t13():
    r = rd("phase4_shap_rankcorr.csv"); names = sorted(set(r.a) | set(r.b))
    M = pd.DataFrame(1.0, index=names, columns=names)
    for x in r.itertuples():
        M.loc[x.a, x.b] = M.loc[x.b, x.a] = x.spearman
    emit("T13a_shap_rank_corr", "SHAP rank agreement (Spearman)", M.reset_index().rename(columns={'index': 'model'}), ["phase4_shap_rankcorr.csv", "phase4_h21.csv"],
         "Attribution ranks agree across models (H21): boosters >=0.96, booster vs LR/WoE >=0.85", "Spearman correlation of mean |attribution| feature ranks across eight models (3 repeats x 2,000 rows).", {c: '{:.3f}' for c in names}, hyp="H21")
    s = rd("phase4_shap.csv", cell='with_rate'); s = s.groupby(['model', 'feature']).share.mean().unstack(0)
    s = s.loc[s.mean(axis=1).sort_values(ascending=False).index].head(8).reset_index()
    emit("T13b_shap_shares", "Share of mean |attribution| by feature", s, ["phase4_shap.csv"], "interest_rate is first in every model (29-38%); credit score and DTI are not top-3 in most models",
         "Share of mean |attribution| (3 repeats) for the eight most important features.", {c: '{:.3f}' for c in s.columns if c != 'feature'}, hyp="H21")
    rc = rd("phase4_reasoncodes.csv", rep='pooled', ebm='EBM')
    emit("T13c_reason_codes", "EBM vs tuned LightGBM reason-code overlap", rc[['version', 'k', 'share_overlap_ge2', 'share_overlap_ge2_strict', 'mean_jaccard', 'share_lt_k_codes']].rename(columns={'share_overlap_ge2': 'share with >=2 shared codes'}),
         ["phase4_reasoncodes.csv"], "Top-3 reason codes agree for 98.9% of loans (H22); not a regulatory-compliance claim", "Overlap of EBM reason codes with TreeSHAP codes of tuned LightGBM (3 repeats x 2,000 loans).",
         {c: '{:.3f}' for c in ['share with >=2 shared codes', 'share_overlap_ge2_strict', 'mean_jaccard', 'share_lt_k_codes']}, hyp="H22")


# ------------------------------------------------------------------------------------------ external datasets (Phase 7)
EXT = {'GERMAN': 'German Credit', 'TAIWAN': 'Taiwan default'}
BOOST4 = ['LightGBM', 'XGBoost', 'CatBoost', 'Gradient Boosting']


def ext_have():
    return [d for d in EXT if (R / f"phase3e_external_{d}.csv").exists() and (R / f"phase3e_external_verdicts_{d}.csv").exists()]


def crit_lending(ds):
    """C1-C5 for D1/D2/D3 from the Phase 2 CSVs (10 models)."""
    w = rd(f"phase2_weighting_{ds}.csv"); c = rd(f"phase2_calibration_{ds}.csv")
    r5 = w[w.thr_rule.astype(str) == '0.5'].pivot(index='model', columns='weighting', values='Recall_mean')
    rt = w[w.thr_rule == 'tuned'].pivot(index='model', columns='weighting', values='Recall_mean')
    cc = c[c.calib != 'raw']; rb = c[(c.calib == 'raw') & (c.weighting == 'balanced')]
    au = w[(w.weighting == 'none') & (w.thr_rule == 'tuned') & w.model.isin(BOOST4)].set_index('model').ROC_AUC_mean
    n = {'D1': 100000, 'D2': 200000, 'D3': 200000}[ds]
    return dict(n=n, pi=float(c.pi_mean.mean()), C1=float((r5['balanced'] - r5['none']).mean()), C2=float((rt['balanced'] - rt['none']).abs().mean()),
                C3=float(cc.dev_pi_mean.abs().mean()), C4=float((rb.dev_half_mean.abs() <= .05).mean()), C5=float(au.max() - au.min()))


def crit_external(ds):
    C = pd.read_csv(R / f"phase3e_external_{ds}.csv"); raw = C[C.calib == 'raw']
    r5 = raw.groupby(['model', 'weighting'])['Recall_0.5'].mean().unstack(); rt = raw.groupby(['model', 'weighting']).Recall_tuned.mean().unstack()
    cc = C[C.calib != 'raw']; rb = raw[raw.weighting == 'balanced'].groupby('model').abs_dev_half.mean()
    au = raw[(raw.weighting == 'none') & raw.model.isin(BOOST4)].groupby('model').ROC_AUC_raw.mean()
    return dict(n=int(C.n_train.iloc[0] + C.n_test.iloc[0]), pi=float(C.pi.mean()), C1=float((r5['balanced'] - r5['none']).mean()), C2=float((rt['balanced'] - rt['none']).abs().mean()),
                C3=float(cc.abs_dev_pi.mean()), C4=float((rb <= .05).mean()), C5=float(au.max() - au.min()))


def t16():
    ds_list = ext_have()
    if not ds_list:
        print("[table] T16 skipped (no external results)"); return
    V = pd.concat([pd.read_csv(R / f"phase3e_external_verdicts_{d}.csv") for d in ds_list])
    C = pd.concat([pd.read_csv(R / f"phase3e_external_{d}.csv") for d in ds_list])
    rows = []
    for d in ds_list:
        v = V[V.dataset == d].set_index('hyp'); c = C[(C.dataset == d) & (C.calib == 'isotonic') & (C.weighting == 'none')]
        b = v[v.index.str.startswith('H26 (bootstrap')]
        rows.append({'dataset': EXT[d], 'n': int(v.n.iloc[0]), 'repeats': int(c.rep.nunique()), 'π': c.pi.mean(), 'mean t* (isotonic)': c.t_star.mean(), 'mean |t*−π| (isotonic)': c.abs_dev_pi.mean(),
                     'models with |t*−π| ≤ .02': f"{int(v.loc['H26', 'models_ok'])}/6", 'bootstrap CI contains π': (f"{int(b.models_ok.iloc[0])}/6" if len(b) else 'n/a'),
                     'H26': (b.result.iloc[0] if len(b) else v.loc['H26', 'result']), 'H27': v.loc['H27', 'result'], 'H28': v.loc['H28', 'result']})
    emit("T16a_external_summary", "External datasets: threshold theory, weighting and model-family verdicts", pd.DataFrame(rows), [f"phase3e_external_{d}.csv" for d in ds_list] + [f"phase3e_external_verdicts_{d}.csv" for d in ds_list],
         "t* tracks the base rate on external datasets (H26), balanced weighting moves raw t* toward 0.5 without changing tuned recall (H27); booster equivalence (H28) is dataset-dependent",
         "Pre-registered H26-H28 verdicts per external dataset (isotonic on OOF scores, unweighted; 10 repeats Taiwan, 30 repeats German Credit).",
         {'π': '{:.3f}', 'mean t* (isotonic)': '{:.3f}', 'mean |t*−π| (isotonic)': '{:.3f}'}, hyp="H26, H27, H28")
    vv = V[['dataset', 'hyp', 'result', 'detail']].copy(); vv['dataset'] = vv.dataset.map(EXT)
    emit("T16b_external_verdict_detail", "External datasets: numbers behind each verdict", vv, [f"phase3e_external_verdicts_{d}.csv" for d in ds_list],
         "Numbers behind every H26-H28 verdict, including the clause-level results", "Clause-level results of H26-H28 per external dataset.", hyp="H26, H27, H28")
    m = C.groupby(['dataset', 'model', 'weighting', 'calib']).agg(ROC_AUC=('ROC_AUC', 'mean'), ROC_AUC_sd=('ROC_AUC', 'std'), PR_AUC=('PR_AUC', 'mean'), Brier=('Brier', 'mean'), ECE=('ECE', 'mean'), pi=('pi', 'mean'),
                                                              t_star=('t_star', 'mean'), t_star_sd=('t_star', 'std'), abs_dev_pi=('abs_dev_pi', 'mean'), abs_dev_half=('abs_dev_half', 'mean'),
                                                              recall_0_5=('Recall_0.5', 'mean'), recall_tuned=('Recall_tuned', 'mean')).reset_index()
    m['dataset'] = m.dataset.map(EXT)
    emit("T16c_external_models", "External datasets: per-model results", m.rename(columns={'abs_dev_pi': '|t*−π|', 'abs_dev_half': '|t*−0.5|', 'recall_0_5': 'recall at 0.5', 'recall_tuned': 'recall at t*', 'ROC_AUC': 'ROC-AUC', 'PR_AUC': 'PR-AUC', 't_star': 't*', 'pi': 'π', 'ROC_AUC_sd': 'ROC-AUC sd', 't_star_sd': 't* sd'}),
         [f"phase3e_external_{d}.csv" for d in ds_list], "Per-model ROC-AUC, PR-AUC, Brier, ECE, t*, |t*−π|, |t*−0.5| and recall for raw, Platt and isotonic scores, unweighted and balanced",
         "Mean over repeats (10 Taiwan, 30 German Credit) per model, weighting and calibration; ROC-AUC of calibrated scores (isotonic has ties).", {c: '{:.4f}' for c in ['ROC-AUC', 'ROC-AUC sd', 'PR-AUC', 'Brier', 'ECE', 'π', 't*', 't* sd', '|t*−π|', '|t*−0.5|', 'recall at 0.5', 'recall at t*']}, hyp="H26, H27")
    B = pd.concat([pd.read_csv(R / f"phase3e_external_tstar_ci_{d}.csv") for d in ds_list]); B['dataset'] = B.dataset.map(EXT)
    emit("T16d_external_tstar_ci", "External datasets: bootstrap CI of t* (isotonic, unweighted)", B[['dataset', 'model', 'pi', 't_star_boot_mean', 'ci_lo', 'ci_hi', 'pi_in_ci', 'n_boot']].rename(columns={'pi': 'π', 't_star_boot_mean': 'bootstrap mean t*', 'ci_lo': 'CI low', 'ci_hi': 'CI high', 'pi_in_ci': 'π in CI'}),
         [f"phase3e_external_tstar_ci_{d}.csv" for d in ds_list], "On German Credit the bootstrap CI of t* contains π for every model although the mean t* sits above π; the CI is wide",
         "Percentile CI of t* from 1,000 resamples of the out-of-fold (score, label) pairs of repeat 0, refitting isotonic calibration and the Youden threshold in each resample.",
         {'π': '{:.3f}', 'bootstrap mean t*': '{:.3f}', 'CI low': '{:.3f}', 'CI high': '{:.3f}'}, hyp="H26")
    T = pd.concat([pd.read_csv(R / f"phase3e_external_tests_{d}.csv") for d in ds_list]); T['dataset'] = T.dataset.map(EXT)
    emit("T16e_external_equivalence", "External datasets: AUC equivalence to the best model", T[['dataset', 'group', 'model', 'best', 'mean_auc', 'best_auc', 'mean_delta', 'ci_lo', 'ci_hi', 'p_holm', 'tost_p_holm', 'result']].rename(columns={'mean_auc': 'AUC', 'best_auc': 'best AUC', 'mean_delta': 'Δ AUC', 'ci_lo': 'CI low', 'ci_hi': 'CI high', 'p_holm': 'p (Holm)', 'tost_p_holm': 'TOST p (Holm)'}),
         [f"phase3e_external_tests_{d}.csv" for d in ds_list], "Nadeau-Bengio corrected TOST (margin 0.005 AUC); failed TOST with a CI including 0 is INCONCLUSIVE, not 'worse'",
         "Corrected resampled TOST against the best model (raw AUC, unweighted); boosters enter H28, other models are reported for context.", {'AUC': '{:.4f}', 'best AUC': '{:.4f}', 'Δ AUC': '{:+.4f}', 'CI low': '{:+.4f}', 'CI high': '{:+.4f}', 'p (Holm)': '{:.4f}', 'TOST p (Holm)': '{:.4f}'}, hyp="H28")
    rr = []
    for d in ds_list:
        r = C[(C.dataset == d) & (C.calib == 'raw')]
        for (w, lab, col) in [(w, l, c) for w in ('none', 'balanced') for l, c in (('0.5', 'Recall_0.5'), ('tuned', 'Recall_tuned'))]:
            g = r[r.weighting == w].groupby('rep').agg(sp=(col, lambda v: v.max() - v.min()), mu=(col, 'mean'))
            rr.append({'dataset': EXT[d], 'weighting': w, 'threshold': lab, 'recall spread across models (mean ± sd)': f"{g.sp.mean():.3f} ± {g.sp.std():.3f}", 'mean recall': g.mu.mean(), 'repeats': len(g)})
    emit("T16f_external_recall_spread", "External datasets: recall spread across models", pd.DataFrame(rr), [f"phase3e_external_{d}.csv" for d in ds_list],
         "Balanced-weighting spread across models halves when thresholds are tuned (as in T3), but the unweighted spread does not shrink and the tuned spread (.13-.19) exceeds D3's (.08)", "Recall spread (max - min across 6 models) by weighting and threshold, raw scores, mean ± sd over repeats.", {'mean recall': '{:.3f}'}, hyp="H27")


def t17():
    ds_list = ext_have()
    if not ds_list:
        print("[table] T17 skipped (no external results)"); return
    cols = {'D1': crit_lending('D1'), 'D2': crit_lending('D2'), 'D3': crit_lending('D3')}
    for d in ds_list:
        cols[EXT[d]] = crit_external(d)
    def surv(k, v, n):
        if k == 'C1': return v >= .20
        if k == 'C2': return v <= (.04 if n < 5000 else .02)
        if k == 'C3': return v <= (.05 if n < 5000 else .02)
        if k == 'C4': return v >= .70
        if k == 'C5': return v <= .010
    names = {'C1': 'recall@0.5 rises with balanced weighting (balanced − none)', 'C2': 'tuned-recall gap |balanced − none|', 'C3': 'calibrated |t*−π|', 'C4': 'share of raw balanced models with |t*−0.5| ≤ .05', 'C5': 'AUC spread across boosters'}
    crit = {'C1': '≥ .20', 'C2': '≤ .02 (≤ .04 if n < 5,000)', 'C3': '≤ .02 (≤ .05 if n < 5,000)', 'C4': '≥ .70', 'C5': '≤ .010'}
    rows = []
    for k in names:
        r = {'claim': k, 'quantity': names[k], 'survives if': crit[k]}
        for ds, v in cols.items():
            r[ds] = f"{v[k]:.3f}"; r[f"{ds} verdict"] = 'SURVIVES' if surv(k, v[k], v['n']) else 'CHANGES'
        rows.append(r)
    out = pd.DataFrame(rows)
    emit("T17_cross_dataset_replication", "Cross-dataset replication of the main claims", out, ["phase2_weighting_D*.csv", "phase2_calibration_D*.csv"] + [f"phase3e_external_{d}.csv" for d in ds_list],
         "Which Lending Club claims survive on the external datasets (criteria fixed before the external analysis)", "Replication table: value of each claim's statistic per dataset and SURVIVES/CHANGES against criteria fixed before the external analysis (tracker, Phase 7).",
         hyp="H26, H27, H28")
    return cols


# ------------------------------------------------------------------------------------------ T14 scorecard
PREREG = {'H1-H3': 'none frozen (stated in the Phase 2 task; first committed with results at b7a4fee)', 'H4': 'none frozen (Phase 3A task; results at 7ac7a5d)',
          'H5-H8': '4f86f74', 'H9-H14': '158e212', 'H15-H19': '6fa9d00', 'H20-H22': 'b3240f4', 'H23-H24': 'f60b319', 'H25': 'e647452', 'H26-H28': 'a23c443 (clarifications b6ae301, before scoring)'}


def pre(h):
    n = int(h[1:])
    return PREREG['H1-H3'] if n <= 3 else PREREG['H4'] if n == 4 else PREREG['H5-H8'] if n <= 8 else PREREG['H9-H14'] if n <= 14 else PREREG['H15-H19'] if n <= 19 else \
        PREREG['H20-H22'] if n <= 22 else PREREG['H23-H24'] if n <= 24 else PREREG['H25'] if n == 25 else PREREG['H26-H28']


def t14():
    hyp = rd("phase2_calibration_hypotheses_D3.csv"); h1 = {d: rd(f"phase2_calibration_hypotheses_{d}.csv").set_index('hypothesis') for d in ('D1', 'D2', 'D3')}
    oot = rd("phase3b_oot_fold_metrics.csv", calib='isotonic')
    agree = np.mean([np.sign(r.pi_test - r.pi_cal) == np.sign(r.t_star_test - r.t_star_cal) for r in oot.itertuples()])
    st = rd("phase3c_stats.csv", dataset='D3', metric='ROC_AUC'); boost = ['Gradient Boosting', 'XGBoost', 'CatBoost']
    n_eq = int(st[st.model.isin(boost)].equivalent.sum())
    ab = rd("phase3c_ablation.csv"); ac = rd("phase3c_ablation_cells.csv").set_index('cell')
    dr = rd("phase3c_drift.csv"); e = dr.groupby(['fix', 'T']).ECE.mean().unstack('T'); imp = 1 - e.loc['iso_trailing6m'] / e.loc['iso_T-1 (baseline)']
    rs = rd("phase3d_resampling_auc_tests.csv"); n_ne = int((~rs.equivalent).sum())
    op = rd("phase3d_optuna_summary.csv").set_index('model'); et = rd("phase3d_ensembles_tests.csv", family='H17').set_index('ensemble'); it = rd("phase3d_interp_tests.csv", cell='with_rate').set_index('model')
    sc = rd("phase3d_scale.csv"); h20 = rd("phase4_h20.csv"); h21 = rd("phase4_h21.csv"); rc = rd("phase4_reasoncodes.csv", rep='pooled', ebm='EBM', k=3).set_index('version')
    h23 = rd("phase5_h23.csv"); d25 = rd("phase6_deploy_oot.csv", scope='protocol') if (R / "phase6_deploy_oot.csv").exists() else None
    pb = rd("phase3b_profit_d3_bootstrap.csv"); pb = pb[pb['vs'] == 'approve_all']
    h5 = int(((pb.scenario.isin(['P1', 'P2_0%', 'P2_3%'])) & (pb.ci_lo > 0)).sum())
    rows = [
        ('H1', 'calibrated => t*~pi', 'SUPPORTED', f"mean |t*-pi| {h1['D1'].loc['H1 calibrated => t*~pi','mean_abs_dev']:.4f}/{h1['D2'].loc['H1 calibrated => t*~pi','mean_abs_dev']:.4f}/{h1['D3'].loc['H1 calibrated => t*~pi','mean_abs_dev']:.4f} (D1/D2/D3)", 'phase2_calibration_hypotheses_D*.csv'),
        ('H2', 'balanced raw => t*~0.5', 'SUPPORTED', f"share within .05: {h1['D1'].loc['H2 balanced raw => t*~0.5','frac_within_tol']:.2f}/{h1['D2'].loc['H2 balanced raw => t*~0.5','frac_within_tol']:.2f}/{h1['D3'].loc['H2 balanced raw => t*~0.5','frac_within_tol']:.2f} (RF/ET fail: leaf purity)", 'phase2_calibration_hypotheses_D*.csv'),
        ('H3', 'weighting invariance after calibration', 'SUPPORTED', f"mean |t*_none - t*_bal| {h1['D1'].loc['H3 weighting-invariance after calibration','mean_abs_dev']:.4f}/{h1['D2'].loc['H3 weighting-invariance after calibration','mean_abs_dev']:.4f}/{h1['D3'].loc['H3 weighting-invariance after calibration','mean_abs_dev']:.4f}", 'phase2_calibration_hypotheses_D*.csv'),
        ('H4', 'stale thresholds under drift', 'NOT SUPPORTED', f"sign agreement of delta-pi and delta-t* {agree:.0%} over {len(oot)} (fold, model, seed) cells; Youden J lost to a stale threshold: fold means {', '.join(f'{k}: {v:.4f}' for k, v in oot.groupby('T').J_loss.mean().items())}, worst single cell {oot.J_loss.max():.4f} ({(oot.J_loss/oot.J_test_max).max():.1%} of J_max)", 'phase3b_oot_fold_metrics.csv'),
        ('H5', 'approve-all optimal in-time on D3', 'SUPPORTED', f"{h5} (model, rule) pairs beat approve-all with CI lower bound >0 at P1/0%/3%", 'phase3b_profit_d3_bootstrap.csv'),
        ('H6', 'expected-profit rule not better than approve-all out of time', 'SUPPORTED', 'CI lower bound <= 0 in every fold/scenario/seed (tracker, Phase 3B)', 'phase3b_expprofit_oot.csv; tracker'),
        ('H7', 'capacity: model family irrelevant and all beat random (default-risk ranking)', 'NOT SUPPORTED', '(a) 3/7 acceptance levels, (b) 0/7 (default-risk ranking below random) (tracker, Phase 3B)', 'phase3b_capacity_summary.csv; tracker'),
        ('H8', 'Phase 2 calibration/weighting results replicate on D3', 'SUPPORTED', 'all six replication criteria pass (REPLICATES)', 'phase3b_replication.csv'),
        ('H9', 'majority of boosting models equivalent to best', 'SUPPORTED', f"{n_eq}/3 of GB, XGBoost, CatBoost equivalent to LightGBM (+/-.005); plus LightGBM itself = 4 of 5 boosting models", 'phase3c_stats.csv'),
        ('H10', 't*~pi in all ablation cells', 'SUPPORTED', f"max mean |t*-pi| over cells/models/calibrators (Platt, isotonic) {ab[ab.calib.isin(['platt', 'isotonic'])].abs_t_pi_mean.max():.4f}", 'phase3c_ablation.csv'),
        ('H11', 'removing rate+score: dAUC>.005, larger capacity gain', 'SUPPORTED', f"dAUC(A-F) {ac.loc['A','ROC_AUC']-ac.loc['F','ROC_AUC']:.4f}; default-risk capacity gain {ac.loc['A','gain_dollar_k50']:,.0f} -> {ac.loc['F','gain_dollar_k50']:,.0f}", 'phase3c_ablation_cells.csv'),
        ('H12', 'dollar capacity gain < count capacity gain', 'SUPPORTED', '$191.6k vs $102.5k per 1000 (-46%) (tracker, Phase 3C)', 'phase3c_capacity.csv; tracker'),
        ('H13', 'default-risk < random under ROI; rate-only ~ best', 'REJECTED', '(a) holds (10/10 models negative ROI gain); (b) fails: rate-only ROI -0.0092 vs best, CI excludes 0 (tracker)', 'phase3c_capacity.csv; tracker'),
        ('H14', 'cheap recalibration fix cuts ECE >=30% in >=2 of 3 folds', 'REJECTED', f"trailing-6m isotonic reduction {imp.loc[2013]:.1%}/{imp.loc[2014]:.1%}/{imp.loc[2015]:.1%}", 'phase3c_drift.csv'),
        ('H15', 'resampling equivalent to unweighted', 'NOT SUPPORTED', f"{n_ne} of {len(rs)} (method, model) cells not equivalent", 'phase3d_resampling_auc_tests.csv'),
        ('H16', 'tuning: boosters <.005, RF gains most, spread shrinks', 'SUPPORTED', f"dAUC LightGBM {op.loc['LightGBM','dAUC']:+.4f}, XGBoost {op.loc['XGBoost','dAUC']:+.4f}, CatBoost {op.loc['CatBoost','dAUC']:+.4f}, RF {op.loc['Random Forest','dAUC']:+.4f}", 'phase3d_optuna_summary.csv'),
        ('H17', 'ensembles equivalent to best single model', 'SUPPORTED', f"voting {et.loc['Soft voting','mean_delta']:+.4f}, stacking {et.loc['Stacking (LR meta)','mean_delta']:+.4f} vs LightGBM", 'phase3d_ensembles_tests.csv'),
        ('H18', 'EBM equivalent; WoE >=.005 below', 'SUPPORTED', f"EBM {it.loc['EBM','mean_delta']:+.4f}; WoE {it.loc['WoE Scorecard','mean_delta']:+.4f}", 'phase3d_interp_tests.csv'),
        ('H19', '667k scale: calibration, weighting, AUC conclusions unchanged', 'SUPPORTED', f"max calibrated |t*-pi| on the 667k population {sc[(sc.dataset == 'D3FULL') & sc.calib.isin(['platt', 'isotonic'])].abs_t_pi_mean.max():.4f}", 'phase3d_scale.csv'),
        ('H20', '36m-only 2013 gain below 2014 and 2015', 'SUPPORTED', f"gain(2014)-gain(2013) {h20.point.iloc[0]:+,.0f} [{h20.ci_lo.iloc[0]:,.0f}, {h20.ci_hi.iloc[0]:,.0f}]; gain(2015)-gain(2013) {h20.point.iloc[1]:+,.0f} [{h20.ci_lo.iloc[1]:,.0f}, {h20.ci_hi.iloc[1]:,.0f}]", 'phase4_h20.csv'),
        ('H21', 'attribution rank agreement', 'SUPPORTED', f"booster pairs min Spearman {h21.value.iloc[0]:.3f}; booster vs LR/WoE min {h21.value.iloc[1]:.3f}; interest_rate first everywhere", 'phase4_h21.csv'),
        ('H22', 'EBM vs LightGBM top-3 reason codes overlap >=2/3 for >=70%', 'SUPPORTED', f"{rc.loc['rate_allowed','share_overlap_ge2']:.3f} (rate allowed), {rc.loc['rate_excluded','share_overlap_ge2']:.3f} (excluded)", 'phase4_reasoncodes.csv'),
        ('H23', '36m refit keeps 2013 < 2014, 2015', 'SUPPORTED', f"gain(2014)-gain(2013) {h23.point.iloc[0]:+,.0f} [{h23.ci_lo.iloc[0]:,.0f}, {h23.ci_hi.iloc[0]:,.0f}]; gain(2015)-gain(2013) {h23.point.iloc[1]:+,.0f} [{h23.ci_lo.iloc[1]:,.0f}, {h23.ci_hi.iloc[1]:,.0f}]", 'phase5_h23.csv'),
        ('H24', 'serving parity (reload and API vs library)', 'SUPPORTED', 'max |diff| 1.1e-16 (reload), 0.0 (API) on 1,000 held-out loans (tracker, Phase 5; app-v2 tests/test_parity.py)', 'app-v2 models/deploy; tracker'),
    ]
    if d25 is not None:
        P = d25.set_index('T'); c1 = int((P.ECE_ratio >= 5).sum()); c2 = bool(((P.AUC_gap >= .01) & (P.AUC_gap <= .03)).all())
        rows.append(('H25', 'deployed configuration out of time: ECE >=5x in >=2/3 folds and AUC gap .010-.030', 'SUPPORTED' if (c1 >= 2 and c2) else 'NOT SUPPORTED',
                     f"ECE ratios {', '.join(f'{k}: {v:.1f}' for k, v in P.ECE_ratio.items())}; AUC gaps {', '.join(f'{k}: {v:.4f}' for k, v in P.AUC_gap.items())}", 'phase6_deploy_oot.csv'))
    if ext_have():
        V = pd.concat([pd.read_csv(R / f"phase3e_external_verdicts_{d}.csv") for d in ext_have()])
        for h, st in (('H26', 'external: isotonic-calibrated Youden t* within .02 of the training prevalence (n >= 20,000) / bootstrap CI contains pi (German Credit)'),
                      ('H27', 'external: balanced weighting moves raw t* toward 0.5 and changes tuned recall by < .02 (< .04 if n < 5,000)'),
                      ('H28', 'external: LightGBM, XGBoost, CatBoost and GB equivalent to the best model on ROC-AUC (TOST +/-.005, N-B, Holm)')):
            g = V[(V.hyp == h) | ((h == 'H26') & V.hyp.str.startswith('H26 (bootstrap'))]
            if h == 'H26':
                g = V[(V.hyp == 'H26') | V.hyp.str.startswith('H26 (bootstrap')]
                g = g[~((g.hyp == 'H26') & (g.n < 20000))]
            verdict = '; '.join(f"{EXT[r.dataset]}: {r.result}" for r in g.itertuples())
            num = ' | '.join(f"{EXT[r.dataset]}: {r.detail}" for r in g.itertuples())
            rows.append((h, st, verdict, num, 'phase3e_external_verdicts_*.csv'))
    df = pd.DataFrame(rows, columns=['hypothesis', 'statement', 'verdict', 'number behind the verdict', 'source']); df['pre-registration commit'] = [pre(h) for h in df.hypothesis]
    emit("T14_hypothesis_scorecard", "Hypothesis scorecard H1-H28", df, ["multiple CSVs (see source column)", "EXTENSION_TRACKER.md for H6, H7, H12, H13, H24"],
         "Every pre-registered hypothesis with its verdict, the number behind it and its frozen pre-registration commit (H1-H4 were not frozen before running)",
         "Scorecard of all hypotheses (H26-H28 verdicts are per external dataset); verdicts are reported against the frozen pre-registered wording.")


# ------------------------------------------------------------------------------------------ T15 claims
def cat_of(status):
    s = status.upper()
    if 'WRONG' in s or 'REJECTED' in s or 'WITHDRAWN' in s or 'NOT SUPPORTED' in s:
        return 'WRONG'
    if 'WEAKENED' in s or 'PARTIAL' in s or 'SPLIT' in s:
        return 'WEAKENED'
    if 'CHANGED' in s or 'REPLACED' in s or 'SUPERSEDED' in s:
        return 'CHANGED'
    if 'CONFIRMED' in s or 'SUPPORTED' in s or 'SURVIVES' in s or 'HOLDS' in s:
        return 'CONFIRMED'
    return 'OTHER'


def parse_claims():
    sec = TRACKER[TRACKER.index('## Paper claims affected'):TRACKER.index('### Limitations added in Phase 4') if '### Limitations added in Phase 4' in TRACKER else None]
    rows, loc = [], ''
    heading = None
    for ln in sec.splitlines():
        if ln.startswith('### '):
            heading = ln[4:].strip()
        if not ln.startswith('|') or set(ln.replace('|', '').strip()) <= set('-: '):
            continue
        cells = [c.strip() for c in ln.strip().strip('|').split(' | ')]
        if cells[0] in ('Manuscript location', 'Claim'):
            continue
        if len(cells) == 4:
            rows.append(dict(manuscript_location=cells[0], claim=cells[1].strip('"'), status_text=cells[2], category=cat_of(cells[2]), evidence=cells[3], tracker_section=heading or 'Paper claims affected (author mapping)'))
        elif len(cells) == 3:
            rows.append(dict(manuscript_location='(not mapped by the author)', claim=cells[0].strip('"'), status_text=cells[1], category=cat_of(cells[1]), evidence=cells[2], tracker_section=heading))
    return pd.DataFrame(rows)


def t15():
    c = parse_claims().drop_duplicates(subset=['claim', 'tracker_section'])
    c['status_text'] = c.status_text.str.replace('**', '', regex=False)
    c.to_csv(OUT / "claims_parsed_from_tracker.csv", index=False)
    emit("T15_claims", "Conference-paper claims versus the journal evidence", c[['manuscript_location', 'claim', 'category', 'status_text', 'evidence', 'tracker_section']].rename(columns={'status_text': 'tracker status'}),
         ["EXTENSION_TRACKER.md (claim tables)"], "Status of each conference claim (category mapped mechanically from the tracker's status wording; see claims_audit.md)", "Conference claims, status and the numbers behind them.")


def claims_audit():
    c = parse_claims(); c['status_text'] = c.status_text.str.replace('**', '', regex=False)
    c['key'] = c.claim.str.lower().str.replace(r'[^a-z0-9]', '', regex=True).str[:45]
    c['latest'] = ~c.duplicated('key', keep='first')              # tracker lists the newest phase first
    mapped = c[c.manuscript_location != '(not mapped by the author)']; other = c[c.manuscript_location == '(not mapped by the author)']
    cnt = lambda d: d.category.value_counts().reindex(['CONFIRMED', 'WEAKENED', 'CHANGED', 'WRONG', 'OTHER']).fillna(0).astype(int)
    L = ["# Claims audit: conference paper versus the journal-extension evidence", "",
         "Generated by `experiments/export_paper_assets.py` from the claim tables in `EXTENSION_TRACKER.md`. **The manuscript text is not in this repository**, so this lists the *claims* the tracker has tested and the corrected figures; "
         "the author must locate the matching numerical statements in the manuscript. Section references are only those the author mapped (2026-10-15); all other rows are the tracker's own wording and carry no manuscript location. "
         "Category is mapped mechanically from the tracker's status wording: WRONG = WRONG / REJECTED / WITHDRAWN / NOT SUPPORTED; WEAKENED = WEAKENED / PARTIAL / SPLIT; CHANGED = CHANGED / REPLACED / SUPERSEDED; CONFIRMED = CONFIRMED / SUPPORTED / SURVIVES / HOLDS. "
         "\"Not equivalent\" is not \"worse\".", "", "## Counts", "", "| scope | CONFIRMED | WEAKENED | CHANGED | WRONG | OTHER | total |", "|---|---|---|---|---|---|---|"]
    for nm, d in (("author-mapped claims (manuscript sections known)", mapped), ("additional tracker claims (no manuscript location)", other), ("all rows, latest wording per claim only", c[c.latest]), ("all rows including superseded earlier wording", c)):
        k = cnt(d); L.append(f"| {nm} | {k['CONFIRMED']} | {k['WEAKENED']} | {k['CHANGED']} | {k['WRONG']} | {k['OTHER']} | {len(d)} |")
    for title, d in (("Part 1: claims mapped to manuscript sections by the author", mapped), ("Part 2: other claims tested by the tracker (manuscript location to be identified)", other)):
        L += ["", f"## {title}", ""]
        for cat in ('WRONG', 'WEAKENED', 'CHANGED'):
            g = d[d.category == cat]
            if g.empty:
                continue
            L += [f"### {cat} ({len(g)})", "", "| manuscript section | claim | tracker status | corrected figure / evidence | tracker section | current? |", "|---|---|---|---|---|---|"]
            for r in g.itertuples():
                L.append(f"| {r.manuscript_location} | {r.claim} | {r.status_text} | {r.evidence} | {r.tracker_section} | {'yes' if r.latest else 'superseded by a later row'} |")
            L.append("")
    (OUT / "claims_audit.md").write_text("\n".join(L) + "\n"); print("[doc] claims_audit.md")


# ------------------------------------------------------------------------------------------ figures
def save(fig, name, caption, sources, claim):
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches='tight')
    import matplotlib.pyplot as plt; plt.close(fig)
    ASSETS.append(dict(kind='figure', id=name, title=name, sources=sources, claim=claim, caption=caption, hyp=''))
    print(f"[figure] {name}")


def figures():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'figure.dpi': 100, 'pdf.fonttype': 42})
    C = ['#4c78a8', '#f58518', '#54a24b', '#e45756', '#72b7b2', '#b279a2']
    # F1 reliability (from cached scores of D3 rep 0)
    try:
        from creditrisk.repeats import get_dataset, load_score
        from creditrisk.calibration import make_calibrators, apply_cal, reliability_curve
        X, y, _ = get_dataset('D3'); yv = y.values
        fig, axes = plt.subplots(1, 4, figsize=(11, 2.9), sharex=True, sharey=True)
        for ax, m in zip(axes, ['Logistic Regression', 'Random Forest', 'LightGBM', 'CatBoost']):
            z = load_score('D3', 0, m, 'balanced'); cals = make_calibrators(z['oof'].astype(float), yv[z['idx_tv']]); y_te = yv[z['idx_te']]
            ax.plot([0, 1], [0, 1], 'k:', lw=.8)
            for (cn, lab), col in zip([('raw', 'raw (balanced)'), ('platt', 'Platt'), ('isotonic', 'isotonic')], C):
                px, py = reliability_curve(y_te, apply_cal(cals[cn], z['te'].astype(float)), 15); ax.plot(px, py, 'o-', ms=2.5, lw=1, color=col, label=lab)
            ax.set_title(m); ax.set_xlabel('mean predicted probability')
        axes[0].set_ylabel('observed default rate'); axes[0].legend(frameon=False); fig.tight_layout()
        save(fig, "F1_reliability", "Reliability diagrams of class-balanced models before and after Platt/isotonic calibration (D3, held-out test, 15 equal-frequency bins).",
             ["cached scores .cache/scores/D3 (rep 0)"], "Calibration repairs the probability scale distorted by class weighting")
    except Exception as ex:
        print("[figure] F1 skipped:", repr(ex))
    # F2 t* vs pi
    s = pd.concat([rd(f"phase2_calibration_summary_{x}.csv") for x in ('D1', 'D2', 'D3')])
    fig, ax = plt.subplots(figsize=(4.2, 3.6)); mk = {'raw': 'x', 'platt': 'o', 'isotonic': 's'}
    for (w, cal), g in s.groupby(['weighting', 'calib']):
        ax.scatter(g.pi, g.t_star, marker=mk[cal], s=34, color=C[0] if w == 'none' else C[1], label=f"{w}, {cal}", alpha=.85)
    ax.plot([.1, .25], [.1, .25], 'k:', lw=.8, label='t* = π'); ax.axhline(.5, color='grey', lw=.6, ls='--'); ax.set_xlabel('prevalence π'); ax.set_ylabel('Youden threshold t*')
    ax.legend(frameon=False, fontsize=6, ncol=2); fig.tight_layout()
    save(fig, "F2_tstar_vs_pi", "Youden-optimal threshold against dataset prevalence for D1, D2, D3: calibrated scores lie on t* = π; raw balanced scores lie near 0.5.", ["phase2_calibration_summary_D*.csv"], "t* ≈ π after calibration, whatever the weighting (H1, H3)")
    # F3 vintage audit
    v = rd("phase3a_vintage_audit.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True)
    for term, col in ((36, C[0]), (60, C[1])):
        g = v[v.term == term]; axes[0].plot(g.issue_year, g.strict_coverage, 'o-', color=col, label=f"{term} months"); axes[1].plot(g.issue_year, g.default_rate, 'o-', color=col, label=f"{term} months")
    axes[0].set_ylabel('share of loans with a resolved strict label'); axes[1].set_ylabel('default rate among resolved loans')
    for a in axes: a.set_xlabel('issue year')
    axes[0].legend(frameon=False); fig.tight_layout()
    save(fig, "F3_vintage_audit", "Resolved-label coverage and default rate by issue year and term: late vintages are censored, motivating the mature-vintage population D3.", ["phase3a_vintage_audit.csv"], "Censoring and survivorship bias in late vintages")
    # F4 profit vs threshold rule
    p = rd("phase3b_profit_d3.csv", model='LightGBM', scenario='P2_3%').set_index('rule'); order = ['approve_all', 'fixed_0.5', 'expected_profit', 'profit_max', 'elkan', 'f1_max', 'youden', 'recall_0.70', 'recall_0.80']
    fig, ax = plt.subplots(figsize=(5.2, 3.2)); vals = p.loc[order, 'profit_per_1000_mean'] / 1000; sd = p.loc[order, 'profit_per_1000_sd'] / 1000
    ax.bar(range(len(order)), vals, yerr=sd, color=[C[2] if o == 'approve_all' else C[0] for o in order], capsize=2)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([o.replace('_', ' ') for o in order], rotation=40, ha='right'); ax.set_ylabel('profit per 1000 applicants (USD thousands)')
    ax.set_title('D3, LightGBM, P2 at 3% discount'); fig.tight_layout()
    save(fig, "F4_profit_by_rule", "Profit per 1000 applicants under each decision rule on mature vintages; approving everyone earns at least as much as any threshold rule (error bars: SD over 10 repeats).", ["phase3b_profit_d3.csv"], "No cutoff beats approve-all (H5)")
    # F5 capacity / ROI
    c = rd("phase3c_capacity.csv", dataset='D3', scenario='P2_3%', capacity='dollar'); c = c.groupby(['ranking', 'k']).agg(g=('gain_profit', 'mean'), r=('gain_roi', 'mean')).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for i, rk in enumerate(['default_risk', 'expected_profit', 'expected_roi', 'ir_high_first']):
        g = c[c.ranking == rk]; axes[0].plot(g.k, g.g / 1000, 'o-', color=C[i], label=rk.replace('_', ' ')); axes[1].plot(g.k, g.r, 'o-', color=C[i])
    for a in axes: a.axhline(0, color='k', lw=.6); a.set_xlabel('share of dollars approved (k)')
    axes[0].set_ylabel('profit gain over random (USD thousands per 1000)'); axes[1].set_ylabel('ROI gain over random'); axes[0].legend(frameon=False, fontsize=6); fig.tight_layout()
    save(fig, "F5_capacity_roi", "Gain over random selection under dollar capacity, by ranking (D3, P2-3%): default-risk ranking is below random while expected-profit and expected-ROI rankings beat it.", ["phase3c_capacity.csv"], "Capacity-constrained approval (H7, H12, H13)")
    # F6 ablation
    a = rd("phase3c_ablation_cells.csv"); fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].bar(a.cell, a.ROC_AUC, color=C[0]); axes[0].set_ylim(.6, .69); axes[0].set_ylabel('ROC-AUC'); axes[1].bar(a.cell, a.gain_dollar_k50 / 1000, color=[C[3] if v < 0 else C[2] for v in a.gain_dollar_k50]); axes[1].axhline(0, color='k', lw=.6)
    axes[1].set_ylabel('default-risk capacity gain, k=50% (USD thousands per 1000)')
    for ax in axes: ax.set_xlabel('feature-set cell (A: rate+FICO ... F: no rate, no score)')
    fig.tight_layout(); save(fig, "F6_ablation", "Endogenous-feature ablation: AUC and default-risk capacity gain for six feature sets (D3).", ["phase3c_ablation_cells.csv"], "Interest rate drives performance and the capacity paradox (H11)")
    # F7 OOT / drift
    o = rd("phase3b_oot_fold_metrics.csv", calib='isotonic').groupby('T').agg(a=('AUC_intime', 'mean'), b=('AUC_oot', 'mean'), e1=('ECE_cal_insample', 'mean'), e2=('ECE_test', 'mean')).reset_index()
    d = rd("phase3c_drift.csv").groupby(['fix', 'T']).ECE.mean().unstack('T'); fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    x = np.arange(3); axes[0].bar(x - .18, o.a, .36, color=C[0], label='in time'); axes[0].bar(x + .18, o.b, .36, color=C[1], label='out of time'); axes[0].set_ylim(.62, .72); axes[0].set_ylabel('ROC-AUC')
    axes[1].bar(x - .18, o.e1, .36, color=C[0], label='calibration year (in-sample)'); axes[1].bar(x + .18, o.e2, .36, color=C[1], label='next year'); axes[1].set_ylabel('ECE of isotonic calibrator'); axes[1].set_ylim(0, .05)
    for a_ in axes[:2]: a_.set_xticks(x); a_.set_xticklabels(o['T']); a_.legend(frameon=False, fontsize=6)
    fx = ['raw', 'iso_T-1 (baseline)', 'platt_T-1', 'iso_trailing6m', 'iso+prior_EM']
    for i, f in enumerate(fx): axes[2].plot(d.columns, d.loc[f], 'o-', color=C[i], label=f)
    axes[2].set_yscale('log'); axes[2].set_yticks([.01, .02, .05, .1, .2, .3]); axes[2].set_yticklabels(['0.01', '0.02', '0.05', '0.1', '0.2', '0.3']); axes[2].minorticks_off(); axes[2].set_ylabel('ECE on the new year'); axes[2].legend(frameon=False, fontsize=5.5); axes[2].set_xticks(list(d.columns))
    fig.tight_layout(); save(fig, "F7_oot_drift", "Out-of-time AUC and ECE (left, centre) and ECE of recalibration strategies (right; log scale) by test year, mean of 5 models x 3 seeds.", ["phase3b_oot_fold_metrics.csv", "phase3c_drift.csv"], "Drift degrades calibration; cheap fixes mostly fail (H4, H14)")
    # F8 SHAP rank heatmap
    r = rd("phase4_shap_rankcorr.csv"); names = sorted(set(r.a) | set(r.b)); M = pd.DataFrame(1.0, index=names, columns=names)
    for q in r.itertuples():
        M.loc[q.a, q.b] = M.loc[q.b, q.a] = q.spearman
    fig, ax = plt.subplots(figsize=(5.6, 4.8)); im = ax.imshow(M.values, vmin=.6, vmax=1, cmap='viridis'); ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=6.5); ax.set_yticklabels(names, fontsize=6.5)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{M.values[i, j]:.2f}", ha='center', va='center', fontsize=6, color='w' if M.values[i, j] < .8 else 'k')
    fig.colorbar(im, label='Spearman of feature ranks'); fig.tight_layout()
    save(fig, "F8_shap_rank_heatmap", "Spearman correlation of mean |attribution| feature ranks across eight models (D3, 3 repeats x 2,000 rows). Scales differ; ranks only.", ["phase4_shap_rankcorr.csv"], "Attribution ranks agree across models (H21)")
    # F9 term / vintage
    t = rd("phase4_term_vintage.csv", ranking='default_risk', k=0.5); t36 = rd("phase5_term36.csv", ranking='default_risk', k=0.5)
    fig, ax = plt.subplots(figsize=(6.4, 3.2)); cells = [(2013, '36'), (2014, '36'), (2015, '36'), (2013, '60'), (2014, '60')]; xs = np.arange(len(cells))
    t['term'] = t.term.astype(str)
    v1 = [t[(t['T'] == a) & (t.term == b)] for a, b in cells]
    ax.bar(xs - .2, [x.gain_per_1000.iloc[0] / 1000 for x in v1], .4, yerr=[[(x.gain_per_1000.iloc[0] - x.ci_lo.iloc[0]) / 1000 for x in v1], [(x.ci_hi.iloc[0] - x.gain_per_1000.iloc[0]) / 1000 for x in v1]], color=C[0], label='mixed-term fit', capsize=2)
    v2 = [t36[t36['T'] == a] if b == '36' else None for a, b in cells]
    xs2 = [i for i, v in enumerate(v2) if v is not None]
    ax.bar([i + .2 for i in xs2], [v2[i].gain_per_1000.iloc[0] / 1000 for i in xs2], .4, yerr=[[(v2[i].gain_per_1000.iloc[0] - v2[i].ci_lo.iloc[0]) / 1000 for i in xs2], [(v2[i].ci_hi.iloc[0] - v2[i].gain_per_1000.iloc[0]) / 1000 for i in xs2]], color=C[1], label='36-month-only refit', capsize=2)
    ax.axhline(0, color='k', lw=.6); ax.set_xticks(xs); ax.set_xticklabels([f"{a}\n{b}m" for a, b in cells]); ax.set_ylabel('default-risk gain over random (USD thousands per 1000)'); ax.legend(frameon=False); fig.tight_layout()
    save(fig, "F9_term_vintage", "Default-risk-ranking capacity gain over random by test year and term (P2-3%, k=50%): the sign change across vintages persists within 36-month loans and under a 36-month-only refit.", ["phase4_term_vintage.csv", "phase5_term36.csv"], "Vintage dependence without a term-mix explanation (H20, H23)")


def figures_ext():
    ds_list = ext_have()
    if not ds_list:
        print("[figure] F10/F11 skipped (no external results)"); return
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'figure.dpi': 100, 'pdf.fonttype': 42})
    C = ['#4c78a8', '#f58518', '#54a24b', '#e45756', '#72b7b2', '#b279a2']
    s = pd.concat([rd(f"phase2_calibration_summary_{x}.csv") for x in ('D1', 'D2', 'D3')])[['dataset', 'weighting', 'calib', 't_star', 'pi']]
    e = pd.concat([pd.read_csv(R / f"phase3e_external_{d}.csv") for d in ds_list]).groupby(['dataset', 'weighting', 'calib']).agg(t_star=('t_star', 'mean'), pi=('pi', 'mean')).reset_index()
    e['dataset'] = e.dataset.map(EXT); a = pd.concat([s, e], ignore_index=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.0)); mk = {'raw': 'x', 'platt': 'o', 'isotonic': 's'}
    for (w, cal), g in a.groupby(['weighting', 'calib']):
        ax.scatter(g.pi, g.t_star, marker=mk[cal], s=34, color=C[0] if w == 'none' else C[1], label=f"{w}, {cal}", alpha=.85, linewidths=1.2 if cal == 'raw' else .6, edgecolors='none' if cal == 'raw' else 'k')
    xs = np.linspace(.1, .32, 2); ax.plot(xs, xs, 'k:', lw=.8, label='t* = π'); ax.axhline(.5, color='grey', lw=.6, ls='--')
    for ds, g in a.groupby('dataset'):
        ax.annotate(ds, (g.pi.mean(), g.t_star.max()), xytext=(0, 6), textcoords='offset points', ha='center', fontsize=7)
    ax.set_xlabel('training prevalence π'); ax.set_ylabel('Youden threshold t* (mean over models and repeats)'); ax.set_ylim(0, .66)
    ax.legend(frameon=False, fontsize=6, ncol=3, loc='upper left'); fig.tight_layout()
    save(fig, "F10_tstar_vs_pi_all", "Youden-optimal threshold against prevalence for Lending Club (D1-D3) and the external datasets: calibrated scores lie near t* = π (German Credit sits above it, with a wide CI), raw balanced scores lie toward 0.5 (0.38-0.47 on the external datasets); raw unweighted scores lie near π on Lending Club and Taiwan and below it on German Credit.",
         ["phase2_calibration_summary_D*.csv"] + [f"phase3e_external_{d}.csv" for d in ds_list], "t* tracks the base rate across datasets after calibration (H1, H26)")
    # F11 reliability on Taiwan (cached scores, repeat 0)
    d = 'TAIWAN' if 'TAIWAN' in ds_list else ds_list[0]
    try:
        from creditrisk.external import LOADERS
        from creditrisk.calibration import make_calibrators, apply_cal, reliability_curve
        import experiments.external_study as ES
    except Exception:
        sys.path.insert(0, str(ROOT / "experiments")); import external_study as ES
        from creditrisk.external import LOADERS
        from creditrisk.calibration import make_calibrators, apply_cal, reliability_curve
    _, y, _ = LOADERS[d](); yv = y.values
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9), sharex=True, sharey=True)
    for ax, m in zip(axes, ['Logistic Regression', 'Random Forest', 'LightGBM', 'CatBoost']):
        z = ES.load(d, 0, m, 'balanced'); cals = make_calibrators(z['oof'], yv[z['idx_tv']]); y_te = yv[z['idx_te']]
        ax.plot([0, 1], [0, 1], 'k:', lw=.8)
        for (cn, lab), col in zip([('raw', 'raw (balanced)'), ('platt', 'Platt'), ('isotonic', 'isotonic')], C):
            px, py = reliability_curve(y_te, apply_cal(cals[cn], z['te']), 15); ax.plot(px, py, 'o-', ms=2.5, lw=1, color=col, label=lab)
        ax.set_title(m); ax.set_xlabel('mean predicted probability')
    axes[0].set_ylabel('observed default rate'); axes[0].legend(frameon=False); fig.tight_layout()
    save(fig, "F11_reliability_external", f"Reliability diagrams of class-balanced models before and after Platt/isotonic calibration on {EXT[d]} (held-out test set of repeat 0, 15 equal-frequency bins).",
         [f"cached scores .cache/external/{d} (rep 0)"], "Calibration repairs the probability scale distorted by class weighting on an external dataset as well")


# ------------------------------------------------------------------------------------------ manifest
def git(*a):
    for g in ('git', '/Library/Developer/CommandLineTools/usr/bin/git'):
        try:
            r = subprocess.run([g, *a], cwd=ROOT, capture_output=True, text=True, timeout=20)
            if r.returncode == 0:
                return r.stdout.strip()
        except Exception:
            pass
    return 'unknown'


def manifest():
    head = git('log', '-1', '--format=%h', '--', 'experiments/export_paper_assets.py'); dirty = 'script not yet committed' if not head else 'script as committed'
    L = ["# Paper assets manifest", "", f"Produced by `experiments/export_paper_assets.py` (last commit touching the script: `{head or 'uncommitted'}`; inputs are the result CSVs and EXTENSION_TRACKER.md of the commit that adds this manifest). Verify with `python paper_assets/verify.py`.",
         "Tables are in `tables/` (CSV with full precision; bare LaTeX booktabs `tabular` with display rounding; `tables/captioned/` holds the same tables wrapped in `table` with caption and label, needing booktabs and graphicx); figures in `figures/` (PNG 300 dpi and PDF). Wording rule: \"not equivalent\" is not \"worse\"; a failed TOST with a CI including 0 is INCONCLUSIVE.", "",
         "| id | kind | files | source(s) | hypotheses | paper claim it supports | caption suggestion |", "|---|---|---|---|---|---|---|"]
    for a in ASSETS:
        fl = f"tables/{a['id']}.csv, tables/{a['id']}.tex, tables/captioned/{a['id']}.tex" if a['kind'] == 'table' else f"figures/{a['id']}.png, figures/{a['id']}.pdf"
        L.append(f"| {a['id']} | {a['kind']} | {fl} | {'; '.join(a['sources'])} | {a['hyp']} | {a['claim']} | {a['caption']} |")
    (OUT / "MANIFEST.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    for f in (t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t16, t17, t14, t15):
        f()
    figures(); figures_ext(); claims_audit(); manifest(); print("done:", len(ASSETS), "assets")
