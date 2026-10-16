"""Phase 4 / task C: EBM reason codes vs tuned-LightGBM TreeSHAP reason codes (H22, frozen at b3240f4).
Uses the contributions cached by shap_study.py (3 reps x 2,000 rows).  Primary EBM = the 3D
configuration (interactions=10; interaction terms split equally between their two features);
EBM without interactions is a sensitivity."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from creditrisk.config import RESULTS_DIR
from creditrisk.reason_codes import top_codes, overlap_stats, label
from creditrisk.repeats import get_dataset
from shap_study import SHAP_DIR, REPS

VERSIONS = {'rate_allowed': (), 'rate_excluded': ('interest_rate',)}


def main():
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    Z = [np.load(SHAP_DIR / f"rep{r}.npz", allow_pickle=True) for r in range(REPS)]
    cols = list(Z[0]['cols'])
    rows, dist = [], []
    for ebm_name in ('EBM', 'EBM_(no_interactions)'):
        for ver, exc in VERSIONS.items():
            for k in (3, 4):
                ov_all, jac_all, av_all, rate_in = [], [], [], []
                for r, z in enumerate(Z):
                    A = top_codes(z[ebm_name], cols, k, exc); B = top_codes(z['LightGBM_(tuned)'], cols, k, exc)
                    ov, jac, av = overlap_stats(A, B)
                    ov_all.append(ov); jac_all.append(jac); av_all.append(av)
                    rows.append(dict(ebm=ebm_name, version=ver, k=k, rep=r, share_overlap_ge2=float(np.mean(ov >= min(2, k))),
                                     share_overlap_ge2_strict=float(np.mean(ov >= 2)), mean_jaccard=float(jac.mean()),
                                     share_top1_same=float(np.mean([a[:1] == b[:1] for a, b in zip(A, B)])),
                                     share_lt_k_codes=float(np.mean(av < k))))
                ov, jac, av = map(np.concatenate, (ov_all, jac_all, av_all))
                hit = np.mean(ov >= np.minimum(2, np.maximum(av, 0))) if k == 3 else np.nan
                strict = np.mean(ov >= 2)
                rows.append(dict(ebm=ebm_name, version=ver, k=k, rep='pooled', share_overlap_ge2=hit if k == 3 else np.mean(ov >= 2),
                                 share_overlap_ge2_strict=strict, mean_jaccard=jac.mean(),
                                 share_top1_same=np.nan, share_lt_k_codes=float(np.mean(av < k))))
                for q in (0, .05, .25, .5, .75, 1):
                    dist.append(dict(ebm=ebm_name, version=ver, k=k, quantile=q, jaccard=float(np.quantile(jac, q))))
    R = pd.DataFrame(rows); R.round(4).to_csv(RESULTS_DIR / "phase4_reasoncodes.csv", index=False)
    pd.DataFrame(dist).round(4).to_csv(RESULTS_DIR / "phase4_reasoncodes_jaccard.csv", index=False)
    print(R[R.rep == 'pooled'].round(4).to_string(index=False))
    print("\nper-rep, primary EBM, k=3:\n", R[(R.ebm == 'EBM') & (R.k == 3) & (R.rep != 'pooled')].round(4).to_string(index=False))
    print("\nJaccard quantiles (primary EBM, k=3):\n", pd.DataFrame(dist).query("ebm=='EBM' and k==3").round(3).to_string(index=False))
    for ver in VERSIONS:
        p = R[(R.ebm == 'EBM') & (R.version == ver) & (R.k == 3) & (R.rep == 'pooled')].share_overlap_ge2.iloc[0]
        print(f"H22 [{ver}]: pooled share with >=2/3 overlap = {p:.3f} -> {'SUPPORTED' if p >= .70 else 'NOT SUPPORTED'}")
    # how often interest_rate is a top-3 code
    z = Z[0]
    for nm in ('EBM', 'LightGBM_(tuned)'):
        t3 = top_codes(z[nm], cols, 3)
        print(f"{nm}: interest_rate in top-3 for {np.mean(['interest_rate' in t for t in t3]):.3f} of loans (rep 0); "
              f"top code distribution:", pd.Series([t[0] if t else 'none' for t in t3]).value_counts(normalize=True).head(5).round(3).to_dict())
    # worked examples (rep 0), anonymised: rounded values, no identifiers
    X, y, _ = get_dataset('D3')
    z = Z[0]; sel = z['sel']; Xs = X.iloc[sel].reset_index(drop=True)
    A = top_codes(z['EBM'], cols, 3); B = top_codes(z['LightGBM_(tuned)'], cols, 3)
    ov, _, _ = overlap_stats(A, B)
    rng = np.random.RandomState(0); picks = []
    for target in (3, 2, 2, 1, 0):
        cand = [i for i in np.where(ov == target)[0] if i not in picks]
        if cand:
            picks.append(int(rng.choice(cand)))
    def anon(f, v):
        if f in ('loan_amount',): return int(round(v / 500) * 500)
        if f == 'annual_income': return int(round(v / 5000) * 5000)
        if isinstance(v, (float, np.floating)): return round(float(v), 1)
        return v
    ex = []; md = ["# Phase 4 worked reason-code examples (rep 0; anonymised: amounts rounded, no identifiers)\n",
                   "_Codes = largest risk-increasing contributions (positive = raises log-odds of default). Not a statement of regulatory compliance._\n"]
    for n, i in enumerate(picks, 1):
        vals = {f: anon(f, Xs.loc[i, f]) for f in cols}
        ce = {f: float(z['EBM'][i, cols.index(f)]) for f in A[i]}; cl = {f: float(z['LightGBM_(tuned)'][i, cols.index(f)]) for f in B[i]}
        ex.append(dict(example=n, overlap=int(ov[i]), ebm_codes='|'.join(A[i]), lgbm_shap_codes='|'.join(B[i]), **{f"x_{f}": v for f, v in vals.items()}))
        md.append(f"## Example {n}: overlap {ov[i]}/3\n")
        md.append("Feature values: " + ", ".join(f"{f}={vals[f]}" for f in cols) + "\n")
        md.append("| rank | EBM code | contribution | tuned LightGBM code | contribution |\n|---|---|---|---|---|")
        for j in range(3):
            a = A[i][j] if j < len(A[i]) else ''; b = B[i][j] if j < len(B[i]) else ''
            md.append(f"| {j+1} | {label(a) if a else ''} ({a}) | {ce.get(a, 0):+.3f} | {label(b) if b else ''} ({b}) | {cl.get(b, 0):+.3f} |")
        md.append("")
    pd.DataFrame(ex).to_csv(RESULTS_DIR / "phase4_reasoncode_examples.csv", index=False)
    (RESULTS_DIR / "phase4_reasoncode_examples.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
