"""End-to-end pipeline: data -> features -> WoE -> selection -> model -> validation -> monitoring.

Usage:  python run_pipeline.py
Outputs go to reports/ (tables, figures, metrics.json, monitoring_report.xlsx).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd

from pdmodel import config, data, master_scale, monitoring, plots, report_excel, scorecard, selection
from pdmodel import validation as V
from pdmodel import woe


def main():
    config.REPORTS.mkdir(exist_ok=True)
    config.FIGURES.mkdir(exist_ok=True)

    # 1. Data + SQL feature engineering ------------------------------------------------
    raw = data.add_split(data.load_raw())
    features, profile, _ = data.build_features(raw)
    train = features[features["sample"] == "train"].reset_index(drop=True)
    test = features[features["sample"] == "test"].reset_index(drop=True)
    candidates = [c for c in features.columns if c.startswith("Attr")] + ["n_missing_ratios"]
    defs = data.attribute_definitions()
    print(f"[data] {len(features)} firms | train {len(train)} | test {len(test)} | "
          f"default rate {features['default_flag'].mean():.2%} | candidates {len(candidates)}")

    # 2. WoE binning + IV ----------------------------------------------------------------
    binners, iv = woe.fit_all(train, candidates)
    iv["strength"] = iv["iv"].apply(woe.iv_strength)
    iv["definition"] = iv["variable"].map(defs)
    woe_train = woe.transform_all(train, binners)

    # 3. Variable selection funnel -------------------------------------------------------
    selected, funnel, stage_counts = selection.run_funnel(iv, woe_train, train["default_flag"])
    funnel["definition"] = funnel["variable"].map(defs)
    print(f"[selection] {stage_counts}")
    print(f"[selection] final variables: {selected}")

    # 4. Final model + benchmark ---------------------------------------------------------
    model = scorecard.PDScorecard(binners, selected).fit(train)
    pd_train, pd_test = model.predict_pd(train), model.predict_pd(test)
    xgb = scorecard.fit_xgboost_benchmark(train, candidates)
    xgb_test = xgb.predict_proba(test[candidates])[:, 1]
    xgb_train = xgb.predict_proba(train[candidates])[:, 1]

    # 5. Validation ----------------------------------------------------------------------
    perf = {"train": V.performance_summary(train["default_flag"], pd_train),
            "test": V.performance_summary(test["default_flag"], pd_test)}
    perf["xgboost_test"] = {"auc": V.auc(test["default_flag"], xgb_test),
                            "gini": V.gini(test["default_flag"], xgb_test),
                            "ks": V.ks_statistic(test["default_flag"], xgb_test),
                            "gini_train": V.gini(train["default_flag"], xgb_train)}
    # sensitivity: performance without the firms whose sales-growth ratio is missing
    m = test["Attr21"].notna()
    perf["sensitivity_test_excl_missing_attr21"] = {
        "n": int(m.sum()), "gini": V.gini(test.loc[m, "default_flag"], pd_test[m.to_numpy()])}
    vif = selection.vif_table(woe_train[selected])

    # 6. Master scale --------------------------------------------------------------------
    grades_train = master_scale.grade_table(train["default_flag"], pd_train)
    grades_test = master_scale.grade_table(test["default_flag"], pd_test)
    deciles = V.decile_table(test["default_flag"], pd_test)

    # 7. Monitoring ----------------------------------------------------------------------
    mon = monitoring.run_monitoring(model, train, test, profile, perf["test"]["gini"])

    # 8. Outputs -------------------------------------------------------------------------
    R, F = config.REPORTS, config.FIGURES
    profile.to_csv(R / "variable_profile.csv", index=False)
    iv.to_csv(R / "iv_summary.csv", index=False)
    funnel.to_csv(R / "selection_funnel.csv", index=False)
    pd.concat([binners[v].table_ for v in selected]).to_csv(R / "woe_bins_selected.csv", index=False)
    model.coefficient_table().to_csv(R / "coefficients.csv", index=False)
    model.points_table().to_csv(R / "scorecard_points.csv", index=False)
    vif.to_csv(R / "vif.csv", index=False)
    grades_train.to_csv(R / "master_scale_train.csv", index=False)
    grades_test.to_csv(R / "master_scale_test.csv", index=False)
    deciles.to_csv(R / "deciles_test.csv", index=False)
    for k, v in mon.items():
        v.to_csv(R / f"monitoring_{k}.csv", index=False)
    (R / "model_summary.txt").write_text(str(model.result_.summary()))

    metrics = {
        "data": {"firms": len(features), "train": len(train), "test": len(test),
                 "defaults": int(features["default_flag"].sum()),
                 "default_rate": float(features["default_flag"].mean()),
                 "candidates": len(candidates),
                 "dropped_missing": profile.loc[~profile["keep_flag"], "variable"].tolist()},
        "selection": {"stage_counts": stage_counts, "selected": selected},
        "performance": perf,
        "master_scale": {"monotonic_train": master_scale.is_monotonic(grades_train),
                         "monotonic_test": master_scale.is_monotonic(grades_test),
                         "bounds": config.MASTER_SCALE_BOUNDS},
        "monitoring": mon["summary"].to_dict(orient="records"),
    }
    (R / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    plots.roc_plot([("Scorecard - train", train["default_flag"], pd_train),
                    ("Scorecard - test", test["default_flag"], pd_test),
                    ("XGBoost - test", test["default_flag"], xgb_test)], F / "roc.png")
    plots.ks_plot(test["default_flag"], pd_test, F / "ks.png")
    plots.calibration_plot(deciles, F / "calibration.png")
    plots.master_scale_plot(grades_test, F / "master_scale.png")
    plots.iv_plot(iv, selected, F / "iv_top20.png")
    plots.monitoring_plot(mon["summary"], F / "monitoring.png", config.PSI_AMBER, config.PSI_RED,
                          config.GINI_DROP_AMBER, config.GINI_DROP_RED)

    thresholds = [
        f"PSI: Green < {config.PSI_AMBER}, Amber {config.PSI_AMBER}-{config.PSI_RED}, Red > {config.PSI_RED}",
        f"Relative Gini drop vs development: Green < {config.GINI_DROP_AMBER:.0%}, "
        f"Amber {config.GINI_DROP_AMBER:.0%}-{config.GINI_DROP_RED:.0%}, Red > {config.GINI_DROP_RED:.0%}",
        f"Calibration (one-sided binomial test, H0: DR <= mean PD): Green p >= {config.CALIB_P_AMBER}, "
        f"Amber {config.CALIB_P_RED}-{config.CALIB_P_AMBER}, Red p < {config.CALIB_P_RED}",
        "Quarters are simulated from the hold-out sample (see README / model documentation).",
    ]
    report_excel.build_report(R / "monitoring_report.xlsx", mon, model.coefficient_table(),
                              model.points_table(), thresholds, perf["test"])

    t = perf["test"]
    print(f"[test] AUC {t['auc']:.3f} | Gini {t['gini']:.3f} (95% CI {t['gini_ci_95'][0]:.3f}-"
          f"{t['gini_ci_95'][1]:.3f}) | KS {t['ks']:.3f} | HL p {t['hosmer_lemeshow_p']:.3f}")
    print(f"[benchmark] XGBoost test Gini {perf['xgboost_test']['gini']:.3f}")
    print(f"[master scale] monotonic train={metrics['master_scale']['monotonic_train']} "
          f"test={metrics['master_scale']['monotonic_test']}")
    print(mon["summary"][["quarter", "psi", "gini", "binomial_p", "overall_rag"]].round(3).to_string(index=False))
    print(f"[done] outputs written to {R}")


if __name__ == "__main__":
    main()
