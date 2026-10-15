"""Inferential statistics behind the feature-selection story.

Reproduces (and extends with effect sizes) the ANOVA and chi-squared analysis
of the original study, so the dashboard can show *why* a feature matters rather
than only how much the model leans on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config


def anova(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """One-way ANOVA of every continuous feature against the binary target.

    Cohen's *d* is reported alongside the p-value: with n≈2000 a tiny but real
    difference reaches significance, and d keeps that in perspective.
    """
    features = features or config.AUDIO_FEATURES
    rows = []
    for var in features:
        a = df.loc[df[config.TARGET] == 0, var].astype(float)
        b = df.loc[df[config.TARGET] == 1, var].astype(float)
        f_stat, p_value = stats.f_oneway(a, b)
        pooled = np.sqrt(((a.var(ddof=1) * (len(a) - 1)) + (b.var(ddof=1) * (len(b) - 1))) / (len(a) + len(b) - 2))
        d = float((b.mean() - a.mean()) / pooled) if pooled else 0.0
        rows.append(
            {
                "feature": var,
                "f_statistic": round(float(f_stat), 3),
                "p_value": float(p_value),
                "cohens_d": round(d, 3),
                "mean_disliked": round(float(a.mean()), 4),
                "mean_liked": round(float(b.mean()), 4),
                "significant": bool(p_value < 0.05),
            }
        )
    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


def chi_square(df: pd.DataFrame, bins: int = 4) -> pd.DataFrame:
    """Chi-squared independence test; continuous features are quantile-binned."""
    rows = []
    candidates = config.AUDIO_FEATURES + config.CATEGORICAL_FEATURES + ["artist"]
    for var in candidates:
        series = df[var]
        if var in config.AUDIO_FEATURES:
            series = pd.qcut(series.astype(float), q=bins, duplicates="drop")
        table = pd.crosstab(series, df[config.TARGET])
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        chi2, p_value, dof, _ = stats.chi2_contingency(table)
        n = int(table.to_numpy().sum())
        cramers_v = float(np.sqrt(chi2 / (n * (min(table.shape) - 1)))) if n else 0.0
        rows.append(
            {
                "feature": var,
                "chi2": round(float(chi2), 3),
                "dof": int(dof),
                "p_value": float(p_value),
                "cramers_v": round(cramers_v, 3),
                "significant": bool(p_value < 0.05),
            }
        )
    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix over the audio features plus the target."""
    cols = config.AUDIO_FEATURES + [config.TARGET]
    return df[cols].astype(float).corr().round(3)


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """FDR control: ~20 tests on one dataset deserve a correction."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    m = len(p)
    thresholds = alpha * (np.arange(1, m + 1) / m)
    passed = ranked <= thresholds
    cutoff = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    out = np.zeros(m, dtype=bool)
    out[order[:cutoff]] = True
    return out.tolist()
