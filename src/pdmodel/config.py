"""Central configuration for the corporate PD model.

Every modelling choice that a reviewer might challenge lives here, so it is
documented in one place and can be changed without touching the pipeline code.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = ROOT / "data" / "raw" / "5year.arff"          # 1-year default horizon file
ATTRIBUTE_DEFS = ROOT / "data" / "raw" / "attribute_definitions.csv"
SQL_DIR = ROOT / "sql"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

SEED = 42                 # reproducibility for split, bootstrap and drift simulation
TEST_SIZE = 0.30          # stratified hold-out share

# --- Data preparation (applied in SQL, parameters estimated on the training sample only)
MISSING_THRESHOLD = 0.20  # drop ratios with >20% missing values in the training sample
WINSOR_LOW = 0.01         # cap each ratio at its 1st percentile ...
WINSOR_HIGH = 0.99        # ... and 99th percentile

# --- WoE binning
N_PREBINS = 10            # initial quantile bins before merging
MIN_BIN_SHARE = 0.05      # every bin must hold >= 5% of the training sample
MIN_MISSING_BIN = 30      # missing values get their own bin only if there are >= 30 of them

# --- Variable selection
IV_MIN = 0.02             # below this a variable has no useful predictive power
IV_SUSPICIOUS = 0.50      # above this, flag for review (possible leakage / over-fit)
CORR_MAX = 0.70           # pairwise |correlation| limit between WoE variables
VIF_MAX = 5.0             # multicollinearity limit
P_VALUE_MAX = 0.05        # significance level for backward elimination
MAX_MODEL_VARS = 10       # keep the scorecard parsimonious and explainable

# --- Score scaling (industry-style points scorecard)
BASE_SCORE = 600          # score at the base odds
BASE_ODDS = 50            # good:bad odds of 50:1 at the base score
PDO = 20                  # points to double the odds

# --- Rating master scale: upper PD bound of each grade (last grade is open-ended)
# (chosen on the training sample so all 8 grades are populated and default rates rise monotonically)
MASTER_SCALE_BOUNDS = [0.0065, 0.01, 0.015, 0.03, 0.06, 0.15, 0.35]
GRADE_LABELS = [f"G{i}" for i in range(1, len(MASTER_SCALE_BOUNDS) + 2)]

# --- Monitoring thresholds (RAG = Red / Amber / Green)
PSI_AMBER, PSI_RED = 0.10, 0.25
GINI_DROP_AMBER, GINI_DROP_RED = 0.05, 0.10     # relative fall in Gini vs development
CALIB_P_AMBER, CALIB_P_RED = 0.05, 0.01         # binomial test p-value

# Simulated monitoring quarters (see monitoring.py):
# (label, risk tilt, feature noise as share of IQR, default-rate uplift vs model PD)
DRIFT_SCENARIOS = [
    ("Q1", 0.00, 0.00, 1.00),   # business as usual
    ("Q2", 0.12, 0.10, 1.15),   # mild deterioration
    ("Q3", 0.25, 0.35, 1.40),   # moderate downturn + noisier financials
    ("Q4", 0.40, 0.70, 2.00),   # severe stress
]
