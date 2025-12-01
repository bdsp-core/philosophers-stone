"""
survival_utils.py

Reusable utilities for survival analysis with lifelines.

Main capabilities
- Fit CoxPH models from a formula or covariate list
- Compute in-sample C-index from fitted models and baseline age-only C-index
- Run feature-wise Cox models adjusted for covariates, extracting HR, p, CIs, and C-index
- Optional bootstrap confidence intervals for C-index
- KM plotting helpers by quantile groups

This module is designed to reproduce the main parts of step9e_survival_analysis.ipynb
in a refactored, reusable way, including adding C-index reporting alongside hazard ratios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import warnings

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index


# -----------------------------
# Data transforms and safeguards
# -----------------------------

def zscore_inplace(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """
    Z-score standardize specified columns in-place, ignoring all-nan columns.

    Returns the same DataFrame for convenience.
    """
    for col in columns:
        if col not in df.columns:
            continue
        s = df[col]
        if s.notna().sum() == 0:
            continue
        mu = s.mean()
        sd = s.std()
        if sd and np.isfinite(sd) and sd > 0:
            df[col] = (s - mu) / sd
    return df


def dropna_rows(df: pd.DataFrame, required: Sequence[str]) -> pd.DataFrame:
    """Drop rows with NA in any required column; returns a new DataFrame."""
    return df.dropna(subset=list(required))


def _ensure_sequence(covariates: Union[str, Sequence[str]]) -> Sequence[str]:
    if isinstance(covariates, str):
        return [covariates]
    return list(covariates)


# -----------------------------
# Cox fitting helpers
# -----------------------------

def _build_formula(covariates: Union[str, Sequence[str]]) -> str:
    if isinstance(covariates, str):
        return covariates
    return " + ".join(covariates)


# def fit_cox(
#     data: pd.DataFrame,
#     duration_col: str,
#     event_col: str,
#     covariates: Union[str, Sequence[str]],
#     **cox_kwargs,
# ) -> CoxPHFitter:
#     """
#     Fit a Cox proportional hazards model using lifelines.

#     covariates can be a formula string (e.g., "age_z + feature") or a list of names.
#     Returns the fitted CoxPHFitter instance.
#     """
#     formula = _build_formula(covariates)
#     cph = CoxPHFitter(**cox_kwargs)
#     cph.fit(data, duration_col=duration_col, event_col=event_col, formula=formula)
#     return cph

def fit_cox(
    data: pd.DataFrame,
    duration_col: str,
    event_col: str,
    covariates: Union[str, Sequence[str]],
    **cox_kwargs,
) -> CoxPHFitter:
    """Fit a Cox proportional hazards model using lifelines."""
    formula = _build_formula(covariates)
    cph = CoxPHFitter(**cox_kwargs)
    cph.fit(data, duration_col=duration_col, event_col=event_col, formula=formula)
    return cph


def bootstrap_cindex_ci(
    data: pd.DataFrame,
    duration_col: str,
    event_col: str,
    covariates: Union[str, Sequence[str]],
    n_bootstrap: int = 200,
    random_state: Optional[int] = None,
    dropna_required: Optional[Sequence[str]] = None,
    **cox_kwargs,
) -> Tuple[float, float]:
    """
    Estimate a 95% CI for Harrell's C-index via non-parametric bootstrap.

    Parameters
    ----------
    data : pd.DataFrame
        Input data containing duration, event, and covariate columns. It should already
        be filtered to the rows you want to analyse (e.g., dropna performed upstream).
    duration_col : str
        Column representing time-to-event.
    event_col : str
        Column representing event indicator (1=event, 0=censored).
    covariates : Union[str, Sequence[str]]
        Either a formula string or a sequence of covariate names for the Cox model.
    n_bootstrap : int, default 200
        Number of bootstrap samples. If <=0, returns (nan, nan).
    random_state : Optional[int]
        Seed controlling the bootstrap resampling.
    dropna_required : Optional[Sequence[str]]
        Additional columns to drop NA before bootstrapping.

    Returns
    -------
    (lower, upper) tuple with the 2.5th and 97.5th percentile of the bootstrapped C-index.
    Returns (nan, nan) if insufficient successful bootstrap fits are obtained.
    """
    if data is None or data.empty or n_bootstrap <= 0:
        return float("nan"), float("nan")

    required_cols = set(_ensure_sequence(covariates)) | {duration_col, event_col}
    if dropna_required:
        required_cols |= set(dropna_required)

    available_cols = [col for col in required_cols if col in data.columns]
    if available_cols:
        df_local = dropna_rows(data, available_cols)
    else:
        df_local = data.copy()
    if df_local.empty:
        return float("nan"), float("nan")

    rng = np.random.default_rng(random_state)
    successes: List[float] = []
    for _ in range(int(n_bootstrap)):
        boot = df_local.sample(
            frac=1.0,
            replace=True,
            random_state=int(rng.integers(0, 1_000_000_000)),
        ).reset_index(drop=True)

        try:
            cph_b = fit_cox(boot, duration_col, event_col, covariates, **cox_kwargs)
            covars_seq = _ensure_sequence(covariates)
            preds = cph_b.predict_partial_hazard(boot[covars_seq]).squeeze()
            preds_arr = preds.to_numpy() if hasattr(preds, "to_numpy") else np.asarray(preds)
            ci_val = concordance_index(
                boot[duration_col].to_numpy(),
                -preds_arr,
                boot[event_col].astype(bool).to_numpy(),
            )
            successes.append(float(ci_val))
        except Exception:
            continue

    successes_arr = np.asarray([s for s in successes if np.isfinite(s)], dtype=float)
    if successes_arr.size == 0:
        warnings.warn("Bootstrap CI failed: no successful Cox fits; returning NaNs.")
        return float("nan"), float("nan")
    if successes_arr.size < max(10, n_bootstrap // 5):
        warnings.warn(
            f"Bootstrap CI failed: only {successes_arr.size}/{n_bootstrap} successful fits; returning NaNs."
        )
        return float("nan"), float("nan")

    lower = float(np.nanpercentile(successes_arr, 2.5))
    upper = float(np.nanpercentile(successes_arr, 97.5))
    return lower, upper


def fit_age_baseline(
    data: pd.DataFrame,
    duration_col: str,
    event_col: str,
    age_col: str = "age_z",
    **cox_kwargs,
) -> Tuple[CoxPHFitter, float]:
    """
    Fit an age-only Cox model and return (model, in-sample C-index).
    """
    cph = fit_cox(data, duration_col, event_col, age_col, **cox_kwargs)
    return cph, float(cph.concordance_index_)


def cindex_out_of_sample(
    cph: CoxPHFitter,
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
) -> float:
    """
    Compute out-of-sample concordance index using a fitted Cox model on a new DataFrame.

    Uses partial hazards (risk scores) predicted by the Cox model.
    """
    # lifelines' predict_partial_hazard expects the same covariates the model was trained on
    scores = cph.predict_partial_hazard(df)
    return float(concordance_index(df[duration_col].values, scores.values, df[event_col].values))


# ---------------------------------
# Feature-wise Cox with HR and C-index
# ---------------------------------

@dataclass
class FeatureCoxResult:
    hazard_ratio: float
    p_value: float
    ci_lower: float
    ci_upper: float
    c_index: float
    delta_c_index_vs_age: float
    # Optional: age columns (when age is in the adjustment set)
    age_hr: Optional[float] = None
    age_p: Optional[float] = None
    age_ci_lower: Optional[float] = None
    age_ci_upper: Optional[float] = None


def _extract_hr_row(cph: CoxPHFitter, covar: str) -> Tuple[float, float, float, float]:
    """Return HR, p, CI low, CI high for a covariate from a fitted cph."""
    row = cph.summary.loc[covar]
    hr = float(row["exp(coef)"])
    p = float(row["p"])
    ci_lo = float(row["exp(coef) lower 95%"])
    ci_hi = float(row["exp(coef) upper 95%"])
    return hr, p, ci_lo, ci_hi


def _safe_extract_age(cph: CoxPHFitter, age_col: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    try:
        return _extract_hr_row(cph, age_col)
    except Exception:
        return None, None, None, None


def run_featurewise_cox(
    data: pd.DataFrame,
    features: Sequence[str],
    duration_col: str,
    event_col: str,
    adjust: Union[str, Sequence[str]] = ("age_z",),
    age_baseline_c_index: Optional[float] = None,
    age_col: str = "age_z",
    dropna_required: Optional[Sequence[str]] = None,
    bootstrap_cindex: int = 0,
    random_state: Optional[int] = None,
    **cox_kwargs,
) -> pd.DataFrame:
    """
    For each feature, fit a Cox model with covariates = adjust + [feature].

    Returns a DataFrame indexed by feature with columns:
    - hazard_ratio, p_value, ci_lower, ci_upper
    - c_index (in-sample), delta_c_index_vs_age
    - age_hr, age_p, age_ci_lower, age_ci_upper (if age is in the model)

    Parameters
    - bootstrap_cindex: if >0, compute bootstrap CI for the C-index (adds columns c_index_ci_lower/_upper)
    - dropna_required: extra columns to require non-NA before model fitting
    """
    rng = np.random.default_rng(random_state)

    # Determine baseline C-index (age-only) if requested and not provided
    if age_baseline_c_index is None and age_col is not None and (
        (isinstance(adjust, str) and adjust == age_col)
        or (isinstance(adjust, (list, tuple)) and age_col in adjust)
    ):
        _, age_baseline_c_index = fit_age_baseline(data, duration_col, event_col, age_col=age_col, **cox_kwargs)

    records = {}

    for feature in features:
        # Build minimal frame and drop NA rows
        covars = [feature] + (list(adjust) if not isinstance(adjust, str) else [adjust])
        needed = set(covars) | {duration_col, event_col}
        if dropna_required:
            needed |= set(dropna_required)
        df_local = dropna_rows(data[list(needed)], list(needed))

        if df_local.shape[0] == 0:
            continue

        # require both events present and feature variability
        if event_col not in df_local.columns or df_local[event_col].nunique() < 2:
            warnings.warn(f"Skipping {feature!r}: insufficient event variation")
            continue
        if df_local[feature].std(skipna=True) is None or not np.isfinite(df_local[feature].std(skipna=True)) or df_local[feature].std(skipna=True) == 0:
            warnings.warn(f"Skipping {feature!r}: predictor has zero variance")
            continue
        if isinstance(adjust, (list, tuple)) or isinstance(adjust, set):
            adjust_it = adjust
        else:
            adjust_it = [adjust]
        adjust_finite = True
        for adj in adjust_it:
            if adj in (None, "") or adj not in df_local.columns:
                continue
            std_adj = df_local[adj].std(skipna=True)
            if std_adj is not None and np.isfinite(std_adj) and std_adj == 0:
                warnings.warn(f"Skipping {feature!r}: adjust covariate '{adj}' has zero variance")
                adjust_finite = False
                break
        if not adjust_finite:
            continue

        try:
            cph = fit_cox(df_local, duration_col, event_col, covars, **cox_kwargs)
        except Exception as e:
            warnings.warn(f"Cox fit failed for feature {feature!r}: {e}")
            continue

        try:
            hr, p, ci_lo, ci_hi = _extract_hr_row(cph, feature)
        except Exception as e:
            warnings.warn(f"Unable to extract HR for {feature!r}: {e}")
            continue

        c_idx = float(cph.concordance_index_)
        delta = float(c_idx - age_baseline_c_index) if age_baseline_c_index is not None else np.nan

        age_hr = age_p = age_ci_lo = age_ci_hi = None
        # Populate age columns only if age is present in the model
        if (isinstance(adjust, str) and adjust == age_col) or (
            isinstance(adjust, (list, tuple)) and age_col in adjust
        ):
            age_hr, age_p, age_ci_lo, age_ci_hi = _safe_extract_age(cph, age_col)

        rec = {
            "hazard_ratio": hr,
            "p_value": p,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "c_index": c_idx,
            "delta_c_index_vs_age": delta,
            "age_hr": age_hr,
            "age_p": age_p,
            "age_ci_lower": age_ci_lo,
            "age_ci_upper": age_ci_hi,
        }

        # Optional: bootstrap CI for C-index
        if bootstrap_cindex and bootstrap_cindex > 0:
            c_idxs = []
            for _ in range(int(bootstrap_cindex)):
                sample = df_local.sample(frac=1.0, replace=True, random_state=int(rng.integers(0, 1e9))).reset_index(drop=True)
                try:
                    cph_b = fit_cox(sample, duration_col, event_col, covars, **cox_kwargs)
                    preds = cph_b.predict_partial_hazard(sample[[c for c in covars]]).squeeze()
                    preds_arr = preds.to_numpy() if hasattr(preds, "to_numpy") else np.asarray(preds)
                    ci_val = concordance_index(
                        sample[duration_col].to_numpy(),
                        -preds_arr,
                        sample[event_col].astype(bool).to_numpy(),
                    )
                    c_idxs.append(float(ci_val))
                except Exception:
                    # Skip failed bootstrap replicas
                    continue
            if len(c_idxs) >= 5:
                rec["c_index_ci_lower"] = float(np.percentile(c_idxs, 2.5))
                rec["c_index_ci_upper"] = float(np.percentile(c_idxs, 97.5))

        records[feature] = rec

    return pd.DataFrame.from_dict(records, orient="index")


# -----------------------------
# KM plotting utilities
# -----------------------------

def add_quantile_group(
    df: pd.DataFrame,
    feature: str,
    q: Sequence[float] = (0.0, 0.25, 0.75, 1.0),
    labels: Optional[Sequence[str]] = ("0-0.25", "0.25-0.75", "0.75-1"),
    colname: str = "quantile_group",
) -> pd.DataFrame:
    """
    Create a categorical quantile group column for a feature using pd.qcut.
    Returns a DataFrame copy with the new column.
    """
    df2 = df.copy()
    df2[colname] = pd.qcut(df2[feature], q=q, labels=labels)
    return df2


def plot_km_by_quantiles(
    df: pd.DataFrame,
    feature: str,
    duration_col: str,
    event_col: str,
    q: Sequence[float] = (0.0, 0.25, 0.75, 1.0),
    labels: Optional[Sequence[str]] = ("0-0.25", "0.25-0.75", "0.75-1"),
    ax=None,
    title: Optional[str] = None,
):
    """
    Plot Kaplan–Meier survival curves for quantile groups of a feature.
    Returns (fig, ax).
    """
    import matplotlib.pyplot as plt

    df2 = add_quantile_group(df, feature, q=q, labels=labels)
    groups = df2["quantile_group"].dropna().unique()

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
    else:
        fig = ax.figure

    kmf = KaplanMeierFitter()
    for grp in groups:
        mask = df2["quantile_group"] == grp
        kmf.fit(durations=df2.loc[mask, duration_col], event_observed=df2.loc[mask, event_col], label=str(grp))
        kmf.plot_survival_function(ci_show=True, ax=ax)

    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.set_title(title or f"KM by {feature} quantiles")
    ax.grid(True, alpha=0.25)
    return fig, ax


# -----------------------------
# Pretty helpers
# -----------------------------

def format_results_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of results with common numeric formatting applied.
    Intended for quick notebook display.
    """
    cols_fmt2 = [c for c in ["hazard_ratio", "ci_lower", "ci_upper", "age_hr", "age_ci_lower", "age_ci_upper"] if c in df.columns]
    cols_fmt4 = [c for c in ["p_value"] if c in df.columns]
    cols_c = [c for c in ["c_index", "delta_c_index_vs_age", "c_index_ci_lower", "c_index_ci_upper"] if c in df.columns]
    out = df.copy()
    for c in cols_fmt2:
        out[c] = out[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    for c in cols_fmt4:
        out[c] = out[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    for c in cols_c:
        out[c] = out[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    return out
