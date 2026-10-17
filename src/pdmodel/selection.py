"""Variable selection funnel: IV -> correlation -> VIF -> stepwise logistic regression.

Each stage is recorded in a log so the model documentation can show exactly why
every variable was kept or dropped (validators always ask for this).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from . import config


def iv_filter(iv: pd.DataFrame):
    keep = iv.loc[iv["iv"] >= config.IV_MIN, "variable"].tolist()
    dropped = iv.loc[iv["iv"] < config.IV_MIN, "variable"].tolist()
    return keep, {v: f"IV < {config.IV_MIN}" for v in dropped}


def correlation_filter(woe_df: pd.DataFrame, ranked_vars):
    """Walk down the IV ranking; drop a variable if it is highly correlated with one already kept."""
    corr = woe_df[ranked_vars].corr().abs()
    kept, log = [], {}
    for v in ranked_vars:
        clash = [k for k in kept if corr.loc[v, k] > config.CORR_MAX]
        if clash:
            log[v] = f"|corr| {corr.loc[v, clash[0]]:.2f} > {config.CORR_MAX} with {clash[0]} (higher IV)"
        else:
            kept.append(v)
    return kept, log


def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    Xc = sm.add_constant(X, has_constant="add")
    return pd.DataFrame({
        "variable": X.columns,
        "vif": [variance_inflation_factor(Xc.values, i + 1) for i in range(X.shape[1])],
    })


def vif_filter(woe_df: pd.DataFrame, variables):
    """Iteratively drop the variable with the highest VIF until all are below the limit."""
    kept, log = list(variables), {}
    while len(kept) > 1:
        vt = vif_table(woe_df[kept])
        worst = vt.loc[vt["vif"].idxmax()]
        if worst["vif"] <= config.VIF_MAX:
            break
        log[worst["variable"]] = f"VIF {worst['vif']:.1f} > {config.VIF_MAX}"
        kept.remove(worst["variable"])
    return kept, log


def _fit(X, y):
    return sm.Logit(y, sm.add_constant(X, has_constant="add")).fit(disp=0)


def forward_stepwise(woe_df: pd.DataFrame, y: pd.Series, candidates):
    """Forward selection on log-likelihood.

    A variable may enter only if (a) its coefficient is negative (higher WoE = safer =
    lower PD, i.e. it behaves the way a credit analyst would expect) and (b) every
    variable in the model stays significant at P_VALUE_MAX.
    """
    selected, log = [], {}
    remaining = list(candidates)
    while remaining and len(selected) < config.MAX_MODEL_VARS:
        best, best_llf = None, -np.inf
        for v in remaining:
            res = _fit(woe_df[selected + [v]], y)
            coefs, pvals = res.params.drop("const"), res.pvalues.drop("const")
            if (coefs < 0).all() and (pvals < config.P_VALUE_MAX).all() and res.llf > best_llf:
                best, best_llf = v, res.llf
        if best is None:
            break
        selected.append(best)
        remaining.remove(best)
    for v in remaining:
        log[v] = ("not selected: wrong sign / insignificant / no improvement"
                  if len(selected) < config.MAX_MODEL_VARS else
                  f"not selected: model capped at {config.MAX_MODEL_VARS} variables")
    return selected, log


def run_funnel(iv: pd.DataFrame, woe_train: pd.DataFrame, y_train: pd.Series):
    """Run the full selection funnel. Returns (final variables, funnel log DataFrame, stage counts)."""
    log = {}
    stage_counts = {"candidates": len(iv)}

    v1, l1 = iv_filter(iv)
    log.update({k: ("1. IV filter", r) for k, r in l1.items()})
    stage_counts["after IV filter"] = len(v1)

    v2, l2 = correlation_filter(woe_train, v1)
    log.update({k: ("2. Correlation filter", r) for k, r in l2.items()})
    stage_counts["after correlation filter"] = len(v2)

    v3, l3 = vif_filter(woe_train, v2)
    log.update({k: ("3. VIF filter", r) for k, r in l3.items()})
    stage_counts["after VIF filter"] = len(v3)

    v4, l4 = forward_stepwise(woe_train, y_train, v3)
    log.update({k: ("4. Stepwise regression", r) for k, r in l4.items()})
    stage_counts["final model"] = len(v4)

    rows = [dict(variable=v, stage="Selected", reason="kept through all stages") for v in v4]
    rows += [dict(variable=v, stage=s, reason=r) for v, (s, r) in log.items()]
    funnel = pd.DataFrame(rows).merge(iv[["variable", "iv"]], on="variable", how="left")
    return v4, funnel, stage_counts
