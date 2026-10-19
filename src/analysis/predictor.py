#!/usr/bin/env python3
"""N10/N11: predict, when L1 finishes, whether recovery will decode a failed block; spend the GPU on the likely ones.

Features (all known at the conventional receiver's result, before any recovery run):
  cbfail      code blocks of this TB that failed the CB CRC (of 10 at MCS 14)
  cbfail_p    the same for the other user of the slot
  partner_ok  the other user's TB decoded
  rho         similarity of the two users' channels (channel estimate)
  gain_db     this user's channel gain, gain_p_db the other user's
  ebno        the slot's Eb/No point
Label: the large neural receiver decodes the TB from the first reception (nrx_c0).
Models: the code-block rule (cbfail <= K), logistic regression, gradient-boosted trees.  Folds are whole chunks
(each chunk has its own seed), so no slot is in both training and test.
Outputs per condition and pooled: AUC, and the budget curve: share of recoverable failures captured when recovery is
attempted on the top x% of failures ranked by the model.
usage: predictor.py --sets n1_umi_v3_cb n3_uma_v3_more ... --output results/n10_predictor.json
"""
import argparse, glob, json, os
os.environ.setdefault("OMP_NUM_THREADS", "4")   # the login node is shared: many threads spin against each other
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

FEATS = ["cbfail", "cbfail_p", "partner_ok", "rho", "gain_db", "gain_p_db", "ebno"]


def load(name):
    X, y, grp = [], [], []
    for k, f in enumerate(sorted(glob.glob(f"results/{name}/c*.npz"))):
        d = np.load(f)
        if "cbfail_t0" not in d.files or d["conv_t0"].shape[1] != 2:
            continue
        cb = d["cbfail_t0"].astype(float)
        if (cb < 0).any():
            continue
        conv, rec = d["conv_t0"], d["nrx_c0"]
        g = 10 * np.log10(d["gain"][:, 0, :] + 1e-30)
        for u in (0, 1):
            m = ~conv[:, u]
            p = 1 - u
            X.append(np.stack([cb[m, u], cb[m, p], conv[m, p].astype(float), d["corr"][m, 0], g[m, u], g[m, p], d["point"][m]], 1))
            y.append(rec[m, u]); grp.append(np.full(m.sum(), hash((name, k)) % 10**9))
    if not X:
        return None
    return np.concatenate(X), np.concatenate(y).astype(int), np.concatenate(grp)


def budget(score, y, fracs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)):
    order = np.argsort(-score)
    tot = max(1, y.sum())
    return {f"{f:.1f}": float(y[order[:int(round(f * len(y)))]].sum() / tot) for f in fracs}


def evaluate(X, y, grp):
    res = {"failures": int(len(y)), "recoverable_pct": 100.0 * float(y.mean())}
    cb_score = -X[:, 0] + 0.01 * X[:, 3]                       # the code-block rule (ties broken by similarity)
    res["auc_cbfail"] = float(roc_auc_score(y, cb_score))
    res["budget_cbfail"] = budget(cb_score, y)
    for name, mk in (("logistic", lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))),
                     ("trees", lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05))):
        p = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, grp):
            m = mk().fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
        res[f"auc_{name}"] = float(roc_auc_score(y, p))
        res[f"budget_{name}"] = budget(p, y)
        for th in (0.5, 0.7, 0.9):
            sel = p >= th
            res[f"{name}_p{th}"] = {"attempt_pct": 100.0 * float(sel.mean()), "precision": float(y[sel].mean()) if sel.any() else None,
                                    "recall": float(y[sel].sum() / max(1, y.sum()))}
    res["budget_random"] = {k: float(k) for k in res["budget_cbfail"]}
    return res


ap = argparse.ArgumentParser()
ap.add_argument("--sets", nargs="+", required=True)
ap.add_argument("--output", required=True)
a = ap.parse_args()
out, pool = {}, []
for s in a.sets:
    r = load(s)
    if r is None:
        print(s, "no code-block counts"); continue
    out[s] = evaluate(*r); pool.append(r)
    o = out[s]
    print(f"{s:22s} fail {o['failures']:5d} recov {o['recoverable_pct']:5.1f}% | AUC cb {o['auc_cbfail']:.3f} logit {o['auc_logistic']:.3f} trees {o['auc_trees']:.3f} | "
          f"30% budget: cb {o['budget_cbfail']['0.3']:.2f} trees {o['budget_trees']['0.3']:.2f} | 50%: cb {o['budget_cbfail']['0.5']:.2f} trees {o['budget_trees']['0.5']:.2f}")
X = np.concatenate([p[0] for p in pool]); y = np.concatenate([p[1] for p in pool]); g = np.concatenate([p[2] for p in pool])
out["pooled"] = evaluate(X, y, g); o = out["pooled"]
print(f"{'pooled':22s} fail {o['failures']:5d} recov {o['recoverable_pct']:5.1f}% | AUC cb {o['auc_cbfail']:.3f} logit {o['auc_logistic']:.3f} trees {o['auc_trees']:.3f} | budget trees {o['budget_trees']}")
print("trees thresholds:", {k: v for k, v in o.items() if k.startswith("trees_p")})
m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05).fit(X, y)
from sklearn.inspection import permutation_importance
imp = permutation_importance(m, X[:8000], y[:8000], n_repeats=3, random_state=0, scoring="roc_auc", n_jobs=1)
out["pooled"]["importance_auc_drop"] = {f: float(v) for f, v in zip(FEATS, imp.importances_mean)}
print("importance (AUC drop):", {f: round(v, 3) for f, v in out["pooled"]["importance_auc_drop"].items()})
json.dump(out, open(a.output, "w"), indent=1)
