-- Feature engineering for the corporate PD model (DuckDB).
--
-- Input : table `raw` (firm_id, default_flag, sample, Attr1..Attr64)
-- Output: table `features`        - cleaned, capped modelling dataset (wide)
--         table `variable_profile` - per-variable data-quality profile
--
-- All thresholds are estimated on the TRAINING sample only and then applied to
-- every sample, so no information leaks from the test set into the model.
-- Placeholders such as {{winsor_low}} are filled in from config.py.

-- 1. Long format: one row per firm x ratio (keep NULLs - missingness is information)
CREATE OR REPLACE TABLE raw_long AS
SELECT firm_id, default_flag, sample, variable, value
FROM raw
UNPIVOT INCLUDE NULLS (value FOR variable IN (COLUMNS('^Attr[0-9]+$')));

-- 2. Data-quality profile per ratio, measured on the training sample
CREATE OR REPLACE TABLE variable_profile AS
SELECT
    variable,
    COUNT(*)                                            AS n_obs,
    AVG(CASE WHEN value IS NULL THEN 1.0 ELSE 0.0 END)  AS missing_rate,
    quantile_cont(value, {winsor_low})                  AS cap_low,
    quantile_cont(value, {winsor_high})                 AS cap_high,
    MIN(value)                                          AS raw_min,
    MAX(value)                                          AS raw_max,
    AVG(CASE WHEN value IS NULL THEN 1.0 ELSE 0.0 END) <= {missing_threshold} AS keep_flag
FROM raw_long
WHERE sample = 'train'
GROUP BY variable;

-- 3. Cap (winsorise) every kept ratio at its training percentiles
CREATE OR REPLACE TABLE capped_long AS
SELECT
    r.firm_id,
    r.default_flag,
    r.sample,
    r.variable,
    CASE
        WHEN r.value IS NULL THEN NULL
        ELSE LEAST(GREATEST(r.value, p.cap_low), p.cap_high)
    END AS value
FROM raw_long r
JOIN variable_profile p USING (variable)
WHERE p.keep_flag;

-- 4. Data-quality feature: how many ratios are missing for each firm
--    (incomplete financial statements are themselves a warning sign)
CREATE OR REPLACE TABLE firm_quality AS
SELECT firm_id, SUM(CASE WHEN value IS NULL THEN 1 ELSE 0 END) AS n_missing_ratios
FROM raw_long
GROUP BY firm_id;

-- 5. Back to wide format: one row per firm
CREATE OR REPLACE TABLE features AS
WITH wide AS (
    PIVOT capped_long
    ON variable
    USING first(value)
    GROUP BY firm_id, default_flag, sample
)
SELECT w.*, q.n_missing_ratios
FROM wide w
JOIN firm_quality q USING (firm_id)
ORDER BY firm_id;
