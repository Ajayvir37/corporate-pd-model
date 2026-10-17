"""Generates notebooks/01_pd_model_walkthrough.ipynb (run once; the notebook is then executed)."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Corporate PD Model - step-by-step walkthrough

This notebook walks through the full model build in the order a model developer would present it
to a validator. Every step calls the same functions as `run_pipeline.py`, so the numbers match the
model documentation.

**Objective:** estimate the 1-year probability of default (PD) of a corporate borrower from its
financial ratios, map it to a rating grade, and monitor the model after deployment.

**Steps**
1. Data and SQL feature engineering
2. Weight-of-Evidence binning and Information Value
3. Variable selection funnel (IV -> correlation -> VIF -> stepwise)
4. Logistic regression and points scorecard
5. Validation: discrimination and calibration
6. Benchmark against XGBoost
7. Rating master scale
8. Monitoring framework (PSI, CSI, Gini decay, calibration)
""")

code("""
import sys, warnings
sys.path.insert(0, "../src")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from IPython.display import Image, display
from pdmodel import config, data, woe, selection, scorecard, master_scale, monitoring
from pdmodel import validation as V
pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 70)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
""")

md("""
## 1. Data and SQL feature engineering

**Data:** UCI *Polish Companies Bankruptcy* dataset (Tomczak, 2016), file `5year.arff`: financial
statements from the 5th year of the observation window, with bankruptcy status **one year later**.
A 1-year horizon is used because Basel IRB PDs are defined over one year.

**Why SQL?** In a bank, model data comes from warehouses, so data preparation is done in SQL
(`sql/feature_engineering.sql`, run in DuckDB):
- unpivot to long format and profile every ratio (missing rate, percentiles) **on the training sample only**
- drop ratios with more than 20% missing values
- cap (winsorise) each ratio at its training 1st and 99th percentiles - ratios like
  `current assets / short-term liabilities` explode when the denominator is near zero
- add a data-quality feature: number of missing ratios per firm
""")

code("""
raw = data.add_split(data.load_raw())
features, profile, con = data.build_features(raw)
train = features[features["sample"] == "train"].reset_index(drop=True)
test = features[features["sample"] == "test"].reset_index(drop=True)
candidates = [c for c in features.columns if c.startswith("Attr")] + ["n_missing_ratios"]
defs = data.attribute_definitions()
print(features.groupby("sample")["default_flag"].agg(firms="size", defaults="sum", default_rate="mean"))
""")

code("""
# Data-quality profile computed in SQL - note the extreme raw min/max before capping
con.execute(\"\"\"
    SELECT variable, round(missing_rate, 3) AS missing_rate, cap_low, cap_high, raw_min, raw_max, keep_flag
    FROM variable_profile ORDER BY missing_rate DESC LIMIT 6
\"\"\").df()
""")

md("""
## 2. Weight of Evidence (WoE) and Information Value (IV)

Each ratio is cut into bins (quantile pre-bins, merged until every bin holds at least 5% of firms
and the default rate moves **monotonically** across bins). Each bin is replaced by its WoE:

$$WoE = \\ln\\left(\\frac{\\%\\,\\text{non-defaulters in bin}}{\\%\\,\\text{defaulters in bin}}\\right)
\\qquad IV = \\sum (\\%\\text{non-def} - \\%\\text{def}) \\times WoE$$

- Higher WoE = safer bin. WoE makes the relationship with log-odds linear, which logistic
  regression assumes, and it handles outliers and missing values (own bin) naturally.
- IV measures a variable's stand-alone predictive power: <0.02 useless, 0.02-0.1 weak,
  0.1-0.3 medium, 0.3-0.5 strong, >0.5 very strong (review for leakage).
""")

code("""
binners, iv = woe.fit_all(train, candidates)
iv["strength"] = iv["iv"].apply(woe.iv_strength)
iv["definition"] = iv["variable"].map(defs)
iv.head(12)
""")

code("""
# Example: the WoE table for the quick ratio - default rate falls steadily as liquidity improves
binners["Attr46"].table_
""")

md("""
**Observation:** many IVs are above 0.5. That would be a red flag in an application scorecard,
but here the label is bankruptcy within one year, so distressed firms already show collapsing sales
and negative profits in their last statements. The strength is plausible, but it is flagged in the
documentation as a limitation (performance will be lower at longer horizons).
""")

md("""
## 3. Variable selection funnel

Financial ratios are heavily correlated (many share numerators like *gross profit* or denominators
like *total assets*). The funnel keeps the model small, stable and explainable:

1. **IV >= 0.02**
2. **Correlation filter:** walking down the IV ranking, drop a variable if |corr| > 0.7 with one already kept
3. **VIF <= 5:** remove remaining multicollinearity
4. **Forward stepwise logistic regression:** add the variable that most improves log-likelihood, but
   only if every coefficient is **negative** (intuitive sign) and **significant** (p < 0.05)
""")

code("""
woe_train = woe.transform_all(train, binners)
selected, funnel, stage_counts = selection.run_funnel(iv, woe_train, train["default_flag"])
print(stage_counts)
print("Selected:", selected)
funnel.assign(definition=funnel["variable"].map(defs)).query("stage != '1. IV filter'").head(12)
""")

md("""
The stepwise stage stopped at **4 variables**: no other candidate added significant information
with the correct sign once these four were in. The four cover four distinct credit-risk dimensions:

| Variable | Ratio | Risk dimension |
|---|---|---|
| Attr21 | sales (n) / sales (n-1) | Growth / business momentum |
| Attr27 | operating profit / financial expenses | Debt-service capacity (interest cover) |
| Attr46 | (current assets - inventory) / short-term liabilities | Liquidity (quick ratio) |
| Attr38 | constant capital / total assets | Capital structure / long-term funding |
""")

code("""
selection.vif_table(woe_train[selected])
""")

md("""
## 4. Logistic regression and points scorecard

$$PD = \\frac{1}{1 + e^{-(\\beta_0 + \\sum_i \\beta_i \\, WoE_i)}}$$

All coefficients are negative (higher WoE = safer = lower PD) and highly significant.
The model is also expressed as a **points scorecard** (600 points at 50:1 odds, 20 points to double
the odds), so a credit analyst can score a company by hand.
""")

code("""
model = scorecard.PDScorecard(binners, selected).fit(train)
model.coefficient_table()
""")

code("""
model.points_table().query("variable == 'Attr27'")
""")

md("""
## 5. Validation

**Discrimination** - does the model rank risky firms above safe ones?
- **AUC**: probability a random defaulter gets a higher PD than a random non-defaulter
- **Gini** = 2 x AUC - 1 (0 = random, 1 = perfect)
- **KS**: maximum gap between the cumulative distributions of defaulters and non-defaulters

**Calibration** - are the PD *levels* right?
- **Hosmer-Lemeshow**: observed vs expected defaults across PD deciles (high p-value = good)
- **Binomial test** (one-sided): is the observed default rate significantly *above* the predicted PD?
""")

code("""
pd_train, pd_test = model.predict_pd(train), model.predict_pd(test)
perf = pd.DataFrame({"train": V.performance_summary(train["default_flag"], pd_train),
                     "test": V.performance_summary(test["default_flag"], pd_test)})
perf.loc[["n", "defaults", "default_rate", "mean_pd", "auc", "gini", "gini_ci_95", "ks", "brier",
          "hosmer_lemeshow_p", "binomial_p_portfolio"]]
""")

md("""
Test Gini is slightly *higher* than train Gini. This is sampling noise (the 95% bootstrap intervals
overlap heavily), not a problem - the key point is there is **no sign of over-fitting**.
""")

code("""
V.decile_table(test["default_flag"], pd_test)
""")

code("""
display(Image("../reports/figures/roc.png", width=420))
display(Image("../reports/figures/calibration.png", width=420))
""")

md("""
**Sensitivity check.** Firms with a missing sales-growth ratio (no prior-year sales) default
at a very high rate in this sample. To make sure the model is not leaning on that data artefact,
performance is re-measured without those firms:
""")

code("""
m = test["Attr21"].notna()
print(f"Test Gini, all firms: {V.gini(test['default_flag'], pd_test):.3f}")
print(f"Test Gini, excluding missing Attr21 ({m.sum()} firms): {V.gini(test.loc[m, 'default_flag'], pd_test[m.to_numpy()]):.3f}")
""")

md("""
## 6. Benchmark: XGBoost challenger

A gradient-boosted tree model on all 64 capped ratios is used as a **challenger**, not a replacement.
""")

code("""
xgb = scorecard.fit_xgboost_benchmark(train, candidates)
print(f"XGBoost  Gini train {V.gini(train['default_flag'], xgb.predict_proba(train[candidates])[:,1]):.3f} | "
      f"test {V.gini(test['default_flag'], xgb.predict_proba(test[candidates])[:,1]):.3f}")
print(f"Scorecard Gini train {V.gini(train['default_flag'], pd_train):.3f} | test {V.gini(test['default_flag'], pd_test):.3f}")
""")

md("""
XGBoost ranks better on the test set (~7 Gini points) but uses 64 inputs, shows a large train/test
gap (over-fitting risk) and cannot be explained bin by bin. In a regulated bank, model risk
guidance (e.g. SR 11-7, ECB TRIM) favours a transparent, stable model whose drivers credit officers
understand - so the 4-variable scorecard is the champion and XGBoost is kept as a benchmark.
""")

md("""
## 7. Rating master scale

PDs are mapped to **8 grades** with fixed PD bands (chosen on the training sample). A good master
scale is **monotonic** (worse grade -> higher default rate) and **calibrated** (observed default
rate inside the confidence interval of the grade PD).
""")

code("""
gt = master_scale.grade_table(test["default_flag"], pd_test)
print("Monotonic on test:", master_scale.is_monotonic(gt))
gt
""")

code("""
display(Image("../reports/figures/master_scale.png", width=560))
""")

md("""
## 8. Monitoring framework

After deployment the model is reviewed quarterly. There is no post-development data for this
dataset, so four quarters are **simulated** from the hold-out sample with controlled drift:

| Quarter | Scenario | Risk tilt | Ratio noise | Default uplift |
|---|---|---|---|---|
| Q1 | business as usual | 0 | 0 | 1.0x |
| Q2 | mild deterioration | 0.12 | 0.10 | 1.15x |
| Q3 | moderate downturn | 0.25 | 0.35 | 1.4x |
| Q4 | severe stress | 0.40 | 0.70 | 2.0x |

- **PSI** on the score (vs development deciles): <0.10 Green, 0.10-0.25 Amber, >0.25 Red
- **CSI** per model variable - points to *which* input is shifting
- **Relative Gini drop** vs development: <5% Green, 5-10% Amber, >10% Red
- **Binomial test** of observed defaults vs mean PD: p >= 0.05 Green, 0.01-0.05 Amber, < 0.01 Red
""")

code("""
mon = monitoring.run_monitoring(model, train, test, profile, V.gini(test["default_flag"], pd_test))
mon["summary"][["quarter", "observed_dr", "mean_pd", "psi", "psi_rag", "gini", "gini_relative_drop",
                "gini_rag", "binomial_p", "calibration_rag", "overall_rag"]]
""")

code("""
mon["csi"]
""")

code("""
display(Image("../reports/figures/monitoring.png", width=900))
""")

md("""
**Reading the results:** Q1 and Q2 stay Green. In Q3 all three metrics turn Amber - the portfolio
is shifting (PSI 0.2), ranking power is slipping and defaults are running above predicted PDs.
In Q4 PSI and calibration are Red: the model would be escalated for redevelopment or a PD
recalibration (e.g. adjusting the intercept to the new central tendency). CSI shows the shift is
driven mainly by `Attr27` (interest cover), which is where a credit analyst would look first.

The full quarterly report with RAG colouring is in `reports/monitoring_report.xlsx`.
""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parent / "01_pd_model_walkthrough.ipynb"
nbf.write(nb, out)
print(f"wrote {out}")
