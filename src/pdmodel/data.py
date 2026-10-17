"""Data loading and SQL-based feature engineering."""
import duckdb
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split

from . import config


def load_raw(path=config.RAW_DATA) -> pd.DataFrame:
    """Read the ARFF file into a DataFrame with a firm id and a 0/1 default flag."""
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    df["default_flag"] = df.pop("class").astype(str).str.strip("b'").astype(int)
    df.insert(0, "firm_id", np.arange(len(df)))
    return df


def add_split(df: pd.DataFrame) -> pd.DataFrame:
    """Stratified train/test split so both samples keep the same default rate."""
    train_idx, _ = train_test_split(
        df.index, test_size=config.TEST_SIZE, stratify=df["default_flag"], random_state=config.SEED
    )
    df = df.copy()
    df.insert(2, "sample", np.where(df.index.isin(train_idx), "train", "test"))
    return df


def attribute_definitions() -> dict:
    defs = pd.read_csv(config.ATTRIBUTE_DEFS)
    out = dict(zip(defs["variable"], defs["definition"]))
    out["n_missing_ratios"] = "number of the 64 ratios missing for the firm (data-quality signal)"
    return out


def build_features(df: pd.DataFrame):
    """Run sql/feature_engineering.sql in DuckDB.

    Returns (features, variable_profile, connection).
    """
    sql = (config.SQL_DIR / "feature_engineering.sql").read_text().format(
        winsor_low=config.WINSOR_LOW,
        winsor_high=config.WINSOR_HIGH,
        missing_threshold=config.MISSING_THRESHOLD,
    )
    con = duckdb.connect()
    con.register("raw_df", df)
    con.execute("CREATE OR REPLACE TABLE raw AS SELECT * FROM raw_df")
    con.execute(sql)
    features = con.execute("SELECT * FROM features").df()
    profile = con.execute(
        "SELECT * FROM variable_profile ORDER BY CAST(substr(variable, 5) AS INTEGER)"
    ).df()
    # restore natural column order Attr1, Attr2, ... (PIVOT sorts alphabetically)
    attr_cols = sorted([c for c in features.columns if c.startswith("Attr")], key=lambda c: int(c[4:]))
    features = features[["firm_id", "default_flag", "sample"] + attr_cols + ["n_missing_ratios"]]
    return features, profile, con
