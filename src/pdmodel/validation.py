"""Model performance metrics: discrimination and calibration.

Discrimination = does the model RANK risky firms above safe ones?  (AUC, Gini, KS)
Calibration    = are the PD LEVELS right?                         (binomial, Hosmer-Lemeshow, Brier)
A model can rank perfectly and still be badly calibrated, so banks test both.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from . import config


def auc(y, p):
    return roc_auc_score(y, p)


def gini(y, p):
    """Gini = 2*AUC - 1 (accuracy ratio). 0 = random, 1 = perfect ranking."""
    return 2 * roc_auc_score(y, p) - 1


def ks_statistic(y, p):
    """Max distance between the cumulative distributions of defaulters and non-defaulters."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def bootstrap_gini_ci(y, p, n_boot=1000, alpha=0.05, seed=config.SEED):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        vals.append(gini(y[idx], p[idx]))
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def hosmer_lemeshow(y, p, groups=10):
    """HL test: compares observed vs expected defaults across PD deciles. High p-value = good fit."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["g"] = pd.qcut(df["p"].rank(method="first"), groups, labels=False)
    agg = df.groupby("g").agg(n=("y", "size"), obs=("y", "sum"), exp=("p", "sum"))
    hl = (((agg["obs"] - agg["exp"]) ** 2) / (agg["exp"] * (1 - agg["exp"] / agg["n"]))).sum()
    return float(hl), float(1 - stats.chi2.cdf(hl, groups - 2))


def binomial_test(defaults, n, pd_estimate):
    """One-sided test of H0: true default rate <= predicted PD.

    A small p-value means the model UNDER-estimates risk - the dangerous direction
    for a bank, because capital and pricing would be too low.
    """
    return float(stats.binomtest(int(defaults), int(n), float(pd_estimate), alternative="greater").pvalue)


def jeffreys_interval(defaults, n, alpha=0.05):
    a, b = defaults + 0.5, n - defaults + 0.5
    return float(stats.beta.ppf(alpha / 2, a, b)), float(stats.beta.ppf(1 - alpha / 2, a, b))


def performance_summary(y, p):
    lo, hi = bootstrap_gini_ci(y, p)
    hl, hl_p = hosmer_lemeshow(y, p)
    y = np.asarray(y)
    return {
        "n": int(len(y)), "defaults": int(y.sum()), "default_rate": float(y.mean()),
        "mean_pd": float(np.mean(p)), "auc": float(auc(y, p)), "gini": float(gini(y, p)),
        "gini_ci_95": [lo, hi], "ks": ks_statistic(y, p), "brier": float(brier_score_loss(y, p)),
        "hosmer_lemeshow": hl, "hosmer_lemeshow_p": hl_p,
        "binomial_p_portfolio": binomial_test(y.sum(), len(y), np.mean(p)),
    }


def decile_table(y, p):
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["decile"] = pd.qcut(df["p"].rank(method="first", ascending=False), 10, labels=range(1, 11))
    t = df.groupby("decile", observed=True).agg(n=("y", "size"), defaults=("y", "sum"), mean_pd=("p", "mean"))
    t["default_rate"] = t["defaults"] / t["n"]
    t["cum_capture"] = t["defaults"].cumsum() / t["defaults"].sum()
    return t.reset_index()
