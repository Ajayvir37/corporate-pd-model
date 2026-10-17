"""Final logistic-regression PD model, points scorecard and XGBoost benchmark."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from xgboost import XGBClassifier

from . import config, woe


class PDScorecard:
    """Logistic regression on WoE variables, with an industry-style points scorecard.

    PD      = 1 / (1 + exp(-(b0 + sum b_i * WoE_i)))
    Score   = Offset + Factor * ln(odds of non-default)
    Factor  = PDO / ln(2);  Offset = BASE_SCORE - Factor * ln(BASE_ODDS)
    """

    def __init__(self, binners: dict, variables):
        self.binners = {v: binners[v] for v in variables}
        self.variables = list(variables)
        self.factor = config.PDO / np.log(2)
        self.offset = config.BASE_SCORE - self.factor * np.log(config.BASE_ODDS)

    def fit(self, df: pd.DataFrame, target="default_flag"):
        X = sm.add_constant(woe.transform_all(df, self.binners)[self.variables], has_constant="add")
        self.result_ = sm.Logit(df[target], X).fit(disp=0)
        self.coef_ = self.result_.params
        return self

    def woe_frame(self, df):
        return woe.transform_all(df, self.binners)[self.variables]

    def log_odds(self, df):
        W = self.woe_frame(df)
        return self.coef_["const"] + W.values @ self.coef_[self.variables].values

    def predict_pd(self, df):
        return 1 / (1 + np.exp(-self.log_odds(df)))

    def score(self, df):
        return self.offset - self.factor * self.log_odds(df)

    def coefficient_table(self):
        r = self.result_
        ci = r.conf_int()
        return pd.DataFrame({
            "variable": r.params.index, "coefficient": r.params.values, "std_error": r.bse.values,
            "z": r.tvalues.values, "p_value": r.pvalues.values,
            "ci_lower": ci[0].values, "ci_upper": ci[1].values,
        })

    def points_table(self):
        """Points per bin: the scorecard a credit analyst can read without any code."""
        n = len(self.variables)
        b0 = self.coef_["const"]
        rows = []
        for v in self.variables:
            t = self.binners[v].table_
            for _, r in t.iterrows():
                pts = self.offset / n - self.factor * (self.coef_[v] * r["woe"] + b0 / n)
                rows.append(dict(variable=v, bin=r["bin"], n=r["n"], default_rate=r["default_rate"],
                                 woe=r["woe"], points=round(pts)))
        return pd.DataFrame(rows)


def fit_xgboost_benchmark(train: pd.DataFrame, features, target="default_flag"):
    """Challenger model on the capped raw ratios (XGBoost handles missing values natively).

    Used only as a benchmark: if a black-box model barely beats the scorecard, the
    interpretable scorecard is the right choice for a regulated bank.
    """
    model = XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, eval_metric="auc",
        random_state=config.SEED, n_jobs=1,
    )
    model.fit(train[features], train[target])
    return model
