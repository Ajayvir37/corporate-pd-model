"""Ongoing model monitoring: population stability, discrimination decay and calibration.

After a model goes live, banks re-check it every quarter:
  * PSI  (Population Stability Index) - has the SCORE distribution shifted?
  * CSI  (Characteristic Stability Index) - which INPUT variable caused the shift?
  * Gini - does the model still rank risk as well as at development?
  * Binomial test - are observed defaults still consistent with predicted PDs?
Each metric gets a Red / Amber / Green (RAG) status against agreed thresholds.

No real post-development data exists for this dataset, so quarters are SIMULATED
from the hold-out sample with two kinds of controlled drift:
  * risk tilt  - the portfolio mix shifts towards riskier firms (e.g. a downturn)
  * feature noise - financial ratios become noisier / less informative
                    (e.g. accounting changes, data-quality problems)
  * default uplift - more firms default than the model predicts
                    (e.g. a downturn the financial ratios have not yet reflected)
This lets us show that the framework detects drift and which metric reacts to what.
"""
import numpy as np
import pandas as pd

from . import config, master_scale, validation


def psi(expected, actual, eps=1e-4):
    """PSI = sum((A% - E%) * ln(A% / E%)) over bins.  <0.10 stable, 0.10-0.25 watch, >0.25 shifted."""
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    e, a = e / e.sum(), a / a.sum()
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def score_psi_table(dev_scores, cur_scores, n_bins=10):
    """PSI on the model score, using deciles of the development score as bins."""
    edges = np.unique(np.quantile(dev_scores, np.linspace(0, 1, n_bins + 1)[1:-1]))
    dev_idx = np.searchsorted(edges, dev_scores, side="right")
    cur_idx = np.searchsorted(edges, cur_scores, side="right")
    k = len(edges) + 1
    dev_n = np.bincount(dev_idx, minlength=k)
    cur_n = np.bincount(cur_idx, minlength=k)
    lows = np.r_[-np.inf, edges]
    highs = np.r_[edges, np.inf]
    t = pd.DataFrame({
        "score_band": [f"[{lo:.0f}, {hi:.0f})" for lo, hi in zip(lows, highs)],
        "dev_share": dev_n / dev_n.sum(), "current_share": cur_n / cur_n.sum(),
    })
    e = np.clip(t["dev_share"], 1e-4, None)
    a = np.clip(t["current_share"], 1e-4, None)
    t["psi_contribution"] = (a - e) * np.log(a / e)
    return t, float(t["psi_contribution"].sum())


def csi(binner, dev_values, cur_values):
    """CSI = PSI computed on one variable's WoE bins."""
    dev = pd.Series(binner.bin_labels(dev_values)).value_counts()
    cur = pd.Series(binner.bin_labels(cur_values)).value_counts()
    bins = dev.index.union(cur.index)
    return psi(dev.reindex(bins, fill_value=0), cur.reindex(bins, fill_value=0))


def rag(value, amber, red, higher_is_worse=True):
    if higher_is_worse:
        return "Red" if value > red else "Amber" if value > amber else "Green"
    return "Red" if value < red else "Amber" if value < amber else "Green"


def overall_rag(statuses):
    return "Red" if "Red" in statuses else "Amber" if "Amber" in statuses else "Green"


def simulate_quarter(pool, model, profile, train, tilt, noise, dr_uplift, rng):
    """Draw one simulated monitoring quarter from the hold-out pool.

    tilt      : sampling weight exp(tilt * standardised log-odds) -> riskier portfolio mix
    noise     : Gaussian noise added to model inputs, scaled by each ratio's IQR
    dr_uplift : extra defaults among surviving firms, so the observed default rate is
                ~dr_uplift x the model PD (a downturn the ratios have not yet picked up)
    """
    lo = model.log_odds(pool)
    z = (lo - lo.mean()) / lo.std()
    w = np.exp(tilt * z)
    idx = rng.choice(len(pool), size=len(pool), replace=True, p=w / w.sum())
    q = pool.iloc[idx].copy().reset_index(drop=True)
    if noise > 0:
        caps = profile.set_index("variable")
        for v in model.variables:
            sd = train[v].quantile(0.75) - train[v].quantile(0.25)
            vals = q[v].to_numpy(dtype=float)
            noisy = vals + rng.normal(0, noise * sd, len(vals))
            lo_cap, hi_cap = caps.loc[v, "cap_low"], caps.loc[v, "cap_high"]
            q[v] = np.where(np.isnan(vals), np.nan, np.clip(noisy, lo_cap, hi_cap))
    if dr_uplift > 1:
        pd_true = model.predict_pd(pool.iloc[idx])
        flip = (q["default_flag"].to_numpy() == 0) & (rng.random(len(q)) < np.clip(pd_true * (dr_uplift - 1), 0, 1))
        q.loc[flip, "default_flag"] = 1
    return q


def run_monitoring(model, train, test, profile, dev_gini):
    """Run all monitoring quarters. Returns a dict of DataFrames for reporting."""
    rng = np.random.default_rng(config.SEED)
    dev_scores = model.score(train)
    summary, psi_details, csi_rows, grade_rows = [], [], [], []

    for label, tilt, noise, dr_uplift in config.DRIFT_SCENARIOS:
        q = simulate_quarter(test, model, profile, train, tilt, noise, dr_uplift, rng)
        y = q["default_flag"].to_numpy()
        pdv = model.predict_pd(q)
        scores = model.score(q)

        psi_tab, psi_val = score_psi_table(dev_scores, scores)
        psi_tab.insert(0, "quarter", label)
        psi_details.append(psi_tab)

        csis = {v: csi(model.binners[v], train[v], q[v]) for v in model.variables}
        csi_rows.append({"quarter": label, **csis})

        g = validation.gini(y, pdv)
        gini_drop = (dev_gini - g) / dev_gini
        n_def, mean_pd = int(y.sum()), float(pdv.mean())
        binom_p = validation.binomial_test(n_def, len(y), mean_pd)

        statuses = {
            "psi_rag": rag(psi_val, config.PSI_AMBER, config.PSI_RED),
            "gini_rag": rag(gini_drop, config.GINI_DROP_AMBER, config.GINI_DROP_RED),
            "calibration_rag": rag(binom_p, config.CALIB_P_AMBER, config.CALIB_P_RED, higher_is_worse=False),
        }
        summary.append({
            "quarter": label, "risk_tilt": tilt, "feature_noise": noise, "dr_uplift": dr_uplift, "n_firms": len(y),
            "defaults": n_def, "observed_dr": float(y.mean()), "mean_pd": mean_pd,
            "psi": psi_val, "max_csi": max(csis.values()), "max_csi_variable": max(csis, key=csis.get),
            "gini": g, "gini_relative_drop": gini_drop, "binomial_p": binom_p,
            **statuses, "overall_rag": overall_rag(list(statuses.values())),
        })

        gt = master_scale.grade_table(y, pdv)
        gt.insert(0, "quarter", label)
        grade_rows.append(gt)

    return {
        "summary": pd.DataFrame(summary),
        "psi_detail": pd.concat(psi_details, ignore_index=True),
        "csi": pd.DataFrame(csi_rows),
        "grades": pd.concat(grade_rows, ignore_index=True),
    }
