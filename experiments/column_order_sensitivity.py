"""Why Random Forest in Table I differs run-to-run: the legacy pipeline's feature
selection step reorders columns using an unseeded mutual_info_classif, and
RF/CatBoost results (and hence the Youden threshold + recall) depend on column order.
Same data, same seed 42, N random column permutations."""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from creditrisk.config import RESULTS_DIR
from creditrisk.data import load_first_n, legacy_clean
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, recall_score, roc_auc_score

ap = argparse.ArgumentParser(); ap.add_argument("--n-perm", type=int, default=20)
ap.add_argument("--sample-size", type=int, default=100_000); a = ap.parse_args()
df = legacy_clean(load_first_n(a.sample_size), True)
y = df.default; X = df.drop(columns='default').copy()
for c in X.select_dtypes(include=['object', 'string']).columns:
    X[c] = LabelEncoder().fit_transform(X[c].astype(str))
X = pd.DataFrame(StandardScaler().fit_transform(X.apply(pd.to_numeric)), columns=X.columns)
rng, rows = np.random.RandomState(0), []
for i in range(a.n_perm):
    cols = list(X.columns) if i == 0 else list(rng.permutation(X.columns))
    Xtr, Xte, ytr, yte = train_test_split(X[cols], y, test_size=.2, random_state=42, stratify=y)
    p = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced').fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    f, t, th = roc_curve(yte, p); thr = float(th[np.argmax(t - f)])
    rows.append(dict(perm=i, Recall=recall_score(yte, (p >= thr).astype(int)), Threshold=thr,
                     AUC=roc_auc_score(yte, p), first_cols="|".join(cols[:3])))
r = pd.DataFrame(rows); r.to_csv(RESULTS_DIR / "rf_column_order_sensitivity.csv", index=False)
print(r[['Recall', 'Threshold', 'AUC']].describe().loc[['mean', 'std', 'min', 'max']].round(4))
