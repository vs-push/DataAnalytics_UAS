"""Small extra helpers used by the two Exercise Project 1 report notebooks.

AI DISCLOSURE: this file was written with Claude (Anthropic) as an AI assistant.
Every function here is short on purpose so it can be read and checked by hand.
For the two calculation helpers (ydata_style_alerts, poly_r2) the docstring says
how the result was verified; the others only wrap plotting or report tools.

helper.py (from the course/lecture setup) is still used for everything else.
"""

import numpy as np
import pandas as pd

from helper import _cramers_v  # bias-corrected Cramer's V, already in helper.py


def try_ydata_profile(df, title, out_html):
    """Build a ydata-profiling report and return its alerts as a DataFrame.

    Unlike the original notebooks, this prints the *real* import error instead of
    "not installed": ydata-profiling 4.18 does `import pkg_resources`, which only
    exists when setuptools is installed (Python 3.12 venvs do not include it).
    Fix in the venv:  python -m pip install "setuptools<81"
    """
    try:
        from ydata_profiling import ProfileReport
    except Exception as e:  # show what actually failed
        print(f"ydata-profiling could not be imported: {type(e).__name__}: {e}")
        print(
            'If the message mentions pkg_resources, run:  python -m pip install "setuptools<81"'
        )
        return None
    profile = ProfileReport(df, title=title, explorative=True)
    profile.to_file(out_html)
    print("Saved", out_html)
    alerts = [str(a) for a in profile.get_description().alerts]
    return pd.DataFrame({"ydata-profiling alert": alerts})


def try_sweetviz(df, target, out_html):
    """SweetViz report (optional second overview tool). Prints the real error if it fails."""
    try:
        import sweetviz as sv

        sv.analyze(df, target_feat=target).show_html(out_html, open_browser=False)
        print("Saved", out_html)
    except Exception as e:
        print(f"SweetViz skipped: {type(e).__name__}: {e}")


def ydata_style_alerts(
    df, skew_thr=20, card_thr=50, imb_thr=0.5, corr_thr=0.9, max_cat_levels=100
):
    """Re-implements the alert *rules* of ydata-profiling 4.18 with plain pandas.

    Thresholds are copied from ydata_profiling/config_default.yaml and the rule logic
    from ydata_profiling/model/alerts.py (read from the project's .venv):
      MISSING  p_missing > 1%      ZEROS  p_zeros > 1% (numeric, >=5 distinct values)
      SKEWED   |skew| > 20         HIGH CARDINALITY  > 50 distinct (categorical)
      IMBALANCE  1 - entropy/log2(k) > 0.5
      HIGH CORRELATION  > 0.9 ("auto": Spearman for numeric pairs, Cramer's V for
                        categorical pairs with <= 100 levels)
    Purpose: (1) list the alerts even where ydata-profiling cannot run, and
    (2) Step 4d - check the tool's alerts against an independent calculation.
    It is an approximation: ydata also discretises numeric-categorical pairs.
    """
    rows, n = [], len(df)
    n_dup = int(df.duplicated().sum())
    if n_dup:
        rows.append(("<whole table>", "DUPLICATES", f"{n_dup} duplicate rows"))
    for c in df.columns:
        s = df[c]
        if s.isna().mean() > 0.01:
            rows.append((c, "MISSING", f"{s.isna().sum()} ({s.isna().mean():.1%})"))
        nd = s.nunique()
        if nd == n:
            rows.append((c, "UNIQUE", "every value different"))
        if pd.api.types.is_numeric_dtype(s) and nd >= 5:
            if (s == 0).mean() > 0.01:
                rows.append((c, "ZEROS", f"{(s == 0).sum()} ({(s == 0).mean():.1%})"))
            if abs(s.skew()) > skew_thr:
                rows.append((c, "SKEWED", f"skew = {s.skew():.1f}"))
        elif not pd.api.types.is_datetime64_any_dtype(s):
            if nd > card_thr:
                rows.append((c, "HIGH CARDINALITY", f"{nd} distinct values"))
            p = s.value_counts(normalize=True).to_numpy(float)
            if len(p) > 1:
                imb = 1 - (-(p * np.log2(p)).sum() / np.log2(len(p)))
                if imb > imb_thr:
                    rows.append((c, "IMBALANCE", f"score {imb:.2f}"))
    num = [c for c in df.select_dtypes("number").columns if df[c].nunique() >= 5]
    sp = df[num].corr(method="spearman").abs()
    for i, a in enumerate(num):
        for b in num[i + 1 :]:
            if sp.loc[a, b] > corr_thr:
                rows.append(
                    (a, "HIGH CORRELATION", f"with {b} (Spearman {sp.loc[a, b]:.2f})")
                )
    cats = [
        c
        for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_datetime64_any_dtype(df[c])
        and 1 < df[c].nunique() <= max_cat_levels
    ]
    for i, a in enumerate(cats):
        for b in cats[i + 1 :]:
            v = _cramers_v(df[a], df[b])
            if v > corr_thr:
                rows.append((a, "HIGH CORRELATION", f"with {b} (Cramer's V {v:.2f})"))
    return pd.DataFrame(rows, columns=["column", "alert", "detail"])


def poly_r2(x, y, degrees=(1, 2, 3)):
    """R^2 of polynomial fits y ~ x of several degrees (numpy.polyfit). Checked against
    sklearn's r2_score on the same data while writing the report."""
    out = {}
    for d in degrees:
        p = np.polyfit(x, y, d)
        resid = y - np.polyval(p, x)
        out[f"degree {d}"] = 1 - (resid**2).sum() / ((y - y.mean()) ** 2).sum()
    return pd.Series(out).round(3)


def binned_trend(ax, x, y, bins=15, color="red", label="median per bin"):
    """Draw the median of y inside equal-count bins of x (a robust, assumption-free trend line).
    Used instead of seaborn's lowess=True, which needs statsmodels."""
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    g = d.groupby(pd.qcut(d["x"], bins, duplicates="drop"), observed=True)
    ax.plot(
        g["x"].median(), g["y"].median(), color=color, marker="o", lw=2, label=label
    )
    return ax
