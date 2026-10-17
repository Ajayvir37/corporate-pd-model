"""Weight-of-Evidence (WoE) binning and Information Value (IV).

Why WoE?
  * Financial ratios are heavy-tailed and often non-linear in risk. Binning them
    and replacing each bin by its WoE makes the relationship with log-odds linear,
    which is exactly what logistic regression assumes.
  * Missing values get their own bin instead of being imputed.
  * Each bin's WoE is easy to explain to credit analysts and validators.

Conventions (standard in credit scoring):
  WoE_bin = ln( %goods_in_bin / %bads_in_bin )      (goods = non-defaults)
  IV      = sum( (%goods - %bads) * WoE )
  A HIGHER WoE therefore means a SAFER bin, and in a default model the logistic
  coefficients on WoE variables should be NEGATIVE.
"""
import numpy as np
import pandas as pd

from . import config


def _woe_iv(goods, bads, total_goods, total_bads):
    # 0.5 smoothing avoids log(0) when a bin has no goods or no bads
    dist_g = (goods + 0.5) / (total_goods + 0.5)
    dist_b = (bads + 0.5) / (total_bads + 0.5)
    woe = np.log(dist_g / dist_b)
    return woe, (dist_g - dist_b) * woe


class WoEBinner:
    """Monotonic WoE binning for one numeric variable."""

    def __init__(self, name, n_prebins=config.N_PREBINS, min_share=config.MIN_BIN_SHARE,
                 min_missing=config.MIN_MISSING_BIN):
        self.name = name
        self.n_prebins = n_prebins
        self.min_share = min_share
        self.min_missing = min_missing

    # ------------------------------------------------------------------ fit
    def fit(self, x: pd.Series, y: pd.Series):
        x = pd.Series(x, dtype=float).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        miss = x.isna()
        xv, yv = x[~miss].to_numpy(), y[~miss].to_numpy()
        self.total_goods = int((y == 0).sum())
        self.total_bads = int((y == 1).sum())
        n_total = len(x)

        # 1. quantile pre-binning -> list of interior cut points
        qs = np.quantile(xv, np.linspace(0, 1, self.n_prebins + 1)[1:-1])
        cuts = list(np.unique(qs))

        def stats(cuts):
            idx = np.searchsorted(cuts, xv, side="right")
            k = len(cuts) + 1
            n = np.bincount(idx, minlength=k)
            b = np.bincount(idx, weights=yv, minlength=k)
            return n, b

        # 2. merge bins that are too small (< min_share of all firms)
        while len(cuts) > 0:
            n, b = stats(cuts)
            small = np.where(n < self.min_share * n_total)[0]
            if len(small) == 0:
                break
            i = small[0]
            # merge with the smaller neighbour
            if i == 0:
                cuts.pop(0)
            elif i == len(n) - 1:
                cuts.pop(-1)
            else:
                cuts.pop(i - 1 if n[i - 1] <= n[i + 1] else i)

        # 3. enforce a monotonic default rate across bins (business intuition +
        #    stability): merge the adjacent pair that breaks the trend most
        n, b = stats(cuts)
        rate = b / np.maximum(n, 1)
        if len(rate) > 1 and np.std(rate) > 0:
            self.trend = 1 if np.corrcoef(np.arange(len(rate)), rate)[0, 1] >= 0 else -1
        else:
            self.trend = 1
        while len(cuts) > 0:
            n, b = stats(cuts)
            rate = b / np.maximum(n, 1)
            diffs = np.diff(rate) * self.trend
            bad_steps = np.where(diffs < 0)[0]
            if len(bad_steps) == 0:
                break
            j = bad_steps[np.argmin(diffs[bad_steps])]
            cuts.pop(j)

        self.cuts = np.array(cuts)
        n, b = stats(self.cuts)
        g = n - b
        woe, iv = _woe_iv(g, b, self.total_goods, self.total_bads)

        lows = np.r_[-np.inf, self.cuts]
        highs = np.r_[self.cuts, np.inf]
        rows = [
            dict(bin=f"[{lo:.4g}, {hi:.4g})", lower=lo, upper=hi, n=int(n_), bads=int(b_),
                 default_rate=b_ / max(n_, 1), woe=w, iv=v)
            for lo, hi, n_, b_, w, v in zip(lows, highs, n, b, woe, iv)
        ]

        # 4. missing values: own bin if material, otherwise the riskiest bin (conservative)
        n_miss = int(miss.sum())
        self.missing_woe = None
        if n_miss > 0:
            b_miss = float(y[miss].sum())
            if n_miss >= self.min_missing:
                w, v = _woe_iv(n_miss - b_miss, b_miss, self.total_goods, self.total_bads)
                self.missing_woe = float(w)
                rows.append(dict(bin="Missing", lower=np.nan, upper=np.nan, n=n_miss, bads=int(b_miss),
                                 default_rate=b_miss / n_miss, woe=w, iv=v))
            else:
                self.missing_woe = float(np.min(woe))
                rows.append(dict(bin="Missing (-> riskiest bin)", lower=np.nan, upper=np.nan, n=n_miss,
                                 bads=int(b_miss), default_rate=b_miss / n_miss,
                                 woe=self.missing_woe, iv=0.0))
        else:
            self.missing_woe = float(np.min(woe))   # unseen missing in future data -> conservative

        self.woe_values = np.asarray(woe)
        self.table_ = pd.DataFrame(rows)
        self.table_.insert(0, "variable", self.name)
        self.iv = float(self.table_["iv"].sum())
        return self

    # ------------------------------------------------------------ transform
    def bin_index(self, x):
        x = np.asarray(x, dtype=float)
        idx = np.searchsorted(self.cuts, x, side="right")
        idx = np.where(np.isnan(x), -1, idx)       # -1 = missing
        return idx

    def transform(self, x):
        idx = self.bin_index(x)
        out = np.where(idx >= 0, self.woe_values[np.clip(idx, 0, None)], self.missing_woe)
        return out

    def bin_labels(self, x):
        """Bin label for each observation - used for CSI monitoring."""
        idx = self.bin_index(x)
        labels = np.array(list(self.table_["bin"][: len(self.woe_values)]) + ["Missing"])
        return labels[np.where(idx >= 0, idx, len(self.woe_values))]


def fit_all(train: pd.DataFrame, variables, target="default_flag"):
    """Fit a WoEBinner per variable. Returns (binners dict, IV summary DataFrame)."""
    binners = {v: WoEBinner(v).fit(train[v], train[target]) for v in variables}
    iv = (pd.DataFrame({"variable": list(binners), "iv": [b.iv for b in binners.values()],
                        "n_bins": [len(b.woe_values) for b in binners.values()]})
          .sort_values("iv", ascending=False).reset_index(drop=True))
    return binners, iv


def transform_all(df: pd.DataFrame, binners: dict) -> pd.DataFrame:
    return pd.DataFrame({v: b.transform(df[v]) for v, b in binners.items()}, index=df.index)


def iv_strength(iv):
    """Conventional rule-of-thumb labels for IV."""
    if iv < 0.02:
        return "not predictive"
    if iv < 0.10:
        return "weak"
    if iv < 0.30:
        return "medium"
    if iv < 0.50:
        return "strong"
    return "very strong (check)"
