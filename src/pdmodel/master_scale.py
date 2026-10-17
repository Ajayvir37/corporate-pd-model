"""Rating master scale: maps a continuous PD to a small number of rating grades.

Banks use grades (not raw PDs) for credit approval authority, limits, pricing and
reporting. Grades must be (1) monotonic - worse grade, higher default rate - and
(2) well calibrated - observed default rate consistent with the grade PD.
"""
import numpy as np
import pandas as pd

from . import config
from .validation import binomial_test, jeffreys_interval


def assign_grade(pd_values):
    idx = np.searchsorted(config.MASTER_SCALE_BOUNDS, np.asarray(pd_values), side="right")
    return np.array(config.GRADE_LABELS)[idx]


def grade_definitions():
    lows = [0.0] + config.MASTER_SCALE_BOUNDS
    highs = config.MASTER_SCALE_BOUNDS + [1.0]
    return pd.DataFrame({"grade": config.GRADE_LABELS, "pd_lower": lows, "pd_upper": highs})


def grade_table(y, pd_values):
    df = pd.DataFrame({"y": np.asarray(y), "pd": np.asarray(pd_values)})
    df["grade"] = assign_grade(df["pd"])
    t = (df.groupby("grade").agg(n=("y", "size"), defaults=("y", "sum"), mean_pd=("pd", "mean"))
         .reindex(config.GRADE_LABELS).fillna(0).reset_index())
    t["share"] = t["n"] / t["n"].sum()
    t["observed_dr"] = np.where(t["n"] > 0, t["defaults"] / t["n"].replace(0, np.nan), np.nan)
    ci = [jeffreys_interval(d, n) if n > 0 else (np.nan, np.nan) for d, n in zip(t["defaults"], t["n"])]
    t["dr_ci_low"], t["dr_ci_high"] = zip(*ci)
    t["binomial_p"] = [binomial_test(d, n, p) if n > 0 else np.nan
                       for d, n, p in zip(t["defaults"], t["n"], t["mean_pd"])]
    t = grade_definitions().merge(t, on="grade")
    return t


def is_monotonic(table):
    dr = table.loc[table["n"] > 0, "observed_dr"].to_numpy()
    return bool(np.all(np.diff(dr) >= 0))
