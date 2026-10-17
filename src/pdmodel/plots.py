"""Static charts for the README and model documentation (matplotlib)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

# Validated categorical palette (slots 1-3) and status colours
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GOOD, WARNING, CRITICAL = "#0ca30c", "#fab219", "#d03b3b"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False, "lines.linewidth": 2,
})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def roc_plot(curves, path):
    """curves: list of (label, y, p)."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for (label, y, p), c in zip(curves, [BLUE, ORANGE, AQUA]):
        fpr, tpr, _ = roc_curve(y, p)
        g = 2 * np.trapezoid(tpr, fpr) - 1
        ax.plot(fpr, tpr, color=c, label=f"{label} (Gini {g:.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="Random")
    ax.set(xlabel="False positive rate (non-defaulters flagged)", ylabel="True positive rate (defaulters caught)",
           title="ROC curve", xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(loc="lower right")
    _save(fig, path)


def ks_plot(y, p, path):
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(-p)
    ys = y[order]
    cum_bad = np.cumsum(ys) / ys.sum()
    cum_good = np.cumsum(1 - ys) / (1 - ys).sum()
    x = np.arange(1, len(ys) + 1) / len(ys)
    i = np.argmax(cum_bad - cum_good)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(x, cum_bad, color=ORANGE, label="Defaulters (cumulative %)")
    ax.plot(x, cum_good, color=BLUE, label="Non-defaulters (cumulative %)")
    ax.vlines(x[i], cum_good[i], cum_bad[i], color=INK, lw=1.2, ls=":")
    ax.annotate(f"KS = {cum_bad[i] - cum_good[i]:.3f}", (x[i], (cum_bad[i] + cum_good[i]) / 2),
                xytext=(8, 0), textcoords="offset points", color=INK, va="center")
    ax.set(xlabel="Share of firms, riskiest first", ylabel="Cumulative share", title="KS chart (test sample)",
           xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(loc="lower right")
    _save(fig, path)


def calibration_plot(dec_table, path):
    t = dec_table.sort_values("mean_pd")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    lim = max(t["mean_pd"].max(), t["default_rate"].max()) * 1.08
    ax.plot([0, lim], [0, lim], color=MUTED, lw=1, ls="--", label="Perfect calibration")
    ax.plot(t["mean_pd"], t["default_rate"], color=BLUE, marker="o", ms=8, mec=SURFACE, mew=2,
            label="PD deciles")
    ax.set(xlabel="Mean predicted PD", ylabel="Observed default rate", title="Calibration (test sample)",
           xlim=(0, lim), ylim=(0, lim))
    ax.legend(loc="upper left")
    _save(fig, path)


def master_scale_plot(gt, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(gt))
    ax.bar(x, gt["mean_pd"] * 100, width=0.55, color=BLUE, alpha=0.85, label="Mean predicted PD")
    obs = gt["observed_dr"].to_numpy() * 100
    lo, hi = gt["dr_ci_low"].to_numpy() * 100, gt["dr_ci_high"].to_numpy() * 100
    floor = max(np.nanmin(lo) / 2, 0.01)
    shown = np.where(obs > 0, obs, floor)          # zero defaults cannot sit on a log axis
    ax.vlines(x, lo, hi, color=INK, lw=1)
    ax.plot(x, shown, ls="none", marker="o", ms=8, color=ORANGE, mec=SURFACE, mew=2,
            label="Observed default rate (95% Jeffreys CI)")
    for xi, o in zip(x, obs):
        if o == 0:
            ax.annotate("0 defaults", (xi, floor), xytext=(0, -14), textcoords="offset points",
                        ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(x, gt["grade"])
    ax.set_yscale("log")
    ax.set_ylim(floor / 2.5, 100)
    ax.set(ylabel="% (log scale)", title="Rating master scale - test sample")
    ax.legend(loc="upper left")
    _save(fig, path)


def iv_plot(iv, selected, path, top=20):
    t = iv[(iv.index < top) | iv["variable"].isin(selected)].iloc[::-1]
    colors = [BLUE if v in selected else "#c9c8c2" for v in t["variable"]]
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.barh(t["variable"], t["iv"], color=colors, height=0.7)
    ax.axvline(0.02, color=MUTED, lw=1, ls="--")
    ax.set(xlabel="Information Value",
           title=f"Top {top} variables by IV + final model variables (blue)")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def monitoring_plot(summary, path, psi_amber, psi_red, drop_amber, drop_red):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    q = summary["quarter"]
    specs = [
        (axes[0], summary["psi"], "Score PSI", [psi_amber, psi_red]),
        (axes[1], summary["gini_relative_drop"] * 100, "Relative Gini drop (%)", [drop_amber * 100, drop_red * 100]),
        (axes[2], summary["binomial_p"], "Calibration: binomial p-value", None),
    ]
    for ax, vals, title, thr in specs:
        ax.plot(q, vals, color=BLUE, marker="o", ms=8, mec=SURFACE, mew=2)
        if thr:
            ax.axhline(thr[0], color=WARNING, lw=1.2, ls="--", label="Amber threshold")
            ax.axhline(thr[1], color=CRITICAL, lw=1.2, ls="--", label="Red threshold")
        else:
            ax.axhline(0.05, color=WARNING, lw=1.2, ls="--", label="Amber (p < 0.05)")
            ax.axhline(0.01, color=CRITICAL, lw=1.2, ls="--", label="Red (p < 0.01)")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Simulated quarterly monitoring", fontweight="bold", color=INK)
    _save(fig, path)
