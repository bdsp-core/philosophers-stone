# conda environment python312

"""
survival_analysis.py

"""

from __future__ import annotations

import os
import sys
from math import erfc, sqrt
from typing import Dict, List, Optional, Sequence, Tuple
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 500)

from PIL import Image, ImageDraw, ImageFont

from survival_utils import (
    format_results_table,
    plot_km_by_quantiles,
    run_featurewise_cox,
    fit_age_baseline,
    zscore_inplace,
    bootstrap_cindex_ci,
)

# -----------------------------
# Config
# -----------------------------

DURATION_COL = "time_fu"
EVENT_COL = "death"
AGE_COL = "age"
AGE_Z_COL = "age_z"

FEATURES_SLEEP_DEFAULT = ["perc_r", "fs_dens_c4m1"]
COLS_COGNITION_DEFAULT = ["cog_fluid", "cog_crystallized", "cog_total"]

CINDEX_BOOTSTRAP_REPS = 7 # for concordance index CIs; paper: 10000
N_BOOTSTRAPS_FOR_CURVE_CI = 7 # for survival curve CIs; paper: 10000
CINDEX_BOOTSTRAP_RANDOM_STATE = 12345

# Resolve base directories relative to this script so outputs stay co-located.
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "survival_outputs"
DATA_DIR = SCRIPT_DIR
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUTPUT_ROOT / "analysis_log.txt"


# -----------------------------
### Logging setup

class Tee:
    """Write all print() outputs to both console and a file."""
    def __init__(self, filename, mode="w"):
        self.file = open(filename, mode)
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

# --- activate logging ---
sys.stdout = Tee(LOG_PATH)


# -----------------------------
# Small helpers
# -----------------------------

def ensure_dir(path: os.PathLike | str) -> None:
    os.makedirs(path, exist_ok=True)


def console_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def safe_read_csv(path: os.PathLike | str, **kwargs) -> Optional[pd.DataFrame]:
    try:
        path = Path(path)
        if not path.is_file():
            print(f"[WARN] File not found: {path}")
            return None
        return pd.read_csv(path, **kwargs)
    except Exception as exc:
        print(f"[WARN] Failed reading {path}: {exc}")
        return None


def ensure_age_z(df: pd.DataFrame) -> bool:
    if AGE_Z_COL in df.columns:
        return True
    if AGE_COL not in df.columns:
        return False
    series = df[AGE_COL]
    std = series.std()
    if series.notna().sum() > 1 and std and np.isfinite(std) and std > 0:
        df[AGE_Z_COL] = (series - series.mean()) / std
    else:
        df[AGE_Z_COL] = series
    return True


def _resolved_brain_health_column(df: pd.DataFrame) -> Optional[str]:
    return "brain_health_score" if "brain_health_score" in df.columns else None


def print_survival_baseline_summary(cohort_label: str, df: pd.DataFrame) -> None:
    
    
    age_min = float(df[AGE_COL].min())
    age_max = float(df[AGE_COL].max())
    age_mean = float(df[AGE_COL].mean())
    age_std = float(df[AGE_COL].std())
    print(f"\nAGE at baseline (sleep study):")
    print(f"  Range:      [{age_min:.1f}, {age_max:.1f}] years")
    print(f"  Mean (SD):  {age_mean:.1f} ({age_std:.1f}) years")
    
    print("\nFOLLOW-UP and SURVIVAL DATA")

    if DURATION_COL not in df.columns or EVENT_COL not in df.columns:
        return
    durations = pd.to_numeric(df[DURATION_COL], errors="coerce").dropna()
    if durations.empty:
        return
    t_min = float(durations.min())
    t_0025 = float(durations.quantile(0.025))
    t_median = float(durations.median())
    t_0975 = float(durations.quantile(0.975))
    t_max = float(durations.max())
    mask = df[DURATION_COL] <= t_median
    mortality = np.nan
    if mask.any():
        events = pd.to_numeric(df.loc[mask, EVENT_COL], errors="coerce")
        mortality = events.mean()
    print(f"  Follow-up (years) 2.5q/median/97.5q       : {t_0025:.2f}/{t_median:.2f}/{t_0975:.2f}")


    from lifelines import KaplanMeierFitter
    km = KaplanMeierFitter()
    km.fit(durations=durations, event_observed=pd.to_numeric(df[EVENT_COL], errors="coerce"))
    s_med = float(km.survival_function_at_times(t_median).ffill().iloc[-1])
    print(f"  KM mortality by median follow-up ({t_median:.2f}y) : {(1-s_med)*100:.1f}%")
    s_max = float(km.survival_function_at_times(t_max).ffill().iloc[-1])
    print(f"  KM mortality by max follow-up ({t_max:.2f}y)    : {(1-s_max)*100:.1f}%")

def _chi2_survival(statistic: float, dof: int) -> float:
    try:
        from scipy.stats import chi2  # type: ignore

        return float(chi2.sf(statistic, dof))
    except Exception:
        try:
            import mpmath as mp  # type: ignore

            return float(mp.gammainc(dof / 2.0, statistic / 2.0, mp.inf) / mp.gamma(dof / 2.0))
        except Exception:
            return float("nan")


def _predict_risk(model, df: pd.DataFrame, horizon: float) -> pd.Series:
    surv = model.predict_survival_function(df, times=[horizon])
    return 1.0 - surv.loc[horizon]


def compute_time_dependent_auc(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    baseline_model,
    full_model,
    horizons: Sequence[float],
    models: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    model_items: List[Tuple[str, object]] = []
    if models:
        model_items.extend(models.items())
    else:
        if baseline_model is not None:
            model_items.append(("age_only", baseline_model))
        if full_model is not None:
            model_items.append(("age_plus_bhs", full_model))
    try:
        from lifelines.utils import concordance_index_ipcw  # type: ignore

        event_times = df[duration_col].to_numpy()
        event_observed = df[event_col].to_numpy().astype(bool)

        for label, model in model_items:
            if model is None:
                continue
            surv = model.predict_survival_function(df, times=horizons)
            for horizon in horizons:
                preds = 1.0 - surv.loc[horizon].to_numpy()
                try:
                    estimate = concordance_index_ipcw(event_times, event_observed, preds, tau=horizon)
                    auc = float(estimate[0] if isinstance(estimate, (tuple, list)) else estimate)
                except Exception:
                    auc = float("nan")
                records.append({"model": label, "horizon": float(horizon), "metric": auc})
    except Exception:
        try:
            from sklearn.metrics import roc_auc_score  # type: ignore
        except Exception:
            print("[WARN] Unable to compute time-dependent AUC: lifelines/sklearn unavailable.")
            return pd.DataFrame()

        for label, model in model_items:
            if model is None:
                continue
            surv = model.predict_survival_function(df, times=horizons)
            for horizon in horizons:
                preds = 1.0 - surv.loc[horizon]
                mask_event = (df[event_col] == 1) & (df[duration_col] <= horizon)
                mask_nonevent = df[duration_col] > horizon
                mask = mask_event | mask_nonevent
                if mask.sum() < 2 or not mask_event.any() or not mask_nonevent.any():
                    auc = float("nan")
                else:
                    labels = mask_event.loc[mask].astype(int)
                    auc = float(roc_auc_score(labels, preds.loc[mask]))
                records.append({"model": label, "horizon": float(horizon), "metric": auc})
    return pd.DataFrame(records)


def compute_absolute_risk_by_quantiles(
    df: pd.DataFrame,
    model,
    feature: str,
    ages: Sequence[float],
    horizon: float,
) -> pd.DataFrame:
    """
    Compute absolute risk predictions across quartiles of a feature.
    
    CRITICAL: This function expects ages in YEARS (e.g., 65, 75).
    If the model was trained on age_z (z-scored age), we need the original
    age column to convert requested ages to z-scores properly.
    """
    if feature not in df.columns:
        return pd.DataFrame()
    
    # Get the subset of data with valid values
    required_cols = [feature]
    required_cols.extend([AGE_COL, AGE_Z_COL])

    df_age = df.dropna(subset=required_cols)
    if df_age.empty:
        return pd.DataFrame()

    # Calculate age statistics from ORIGINAL age column
    age_mean = df_age[AGE_COL].mean()
    age_std = df_age[AGE_COL].std()
    age_min = df_age[AGE_COL].min()
    age_max = df_age[AGE_COL].max()

    if not (np.isfinite(age_std) and age_std > 0):
        age_std = 1.0

    # Validate requested ages are reasonable
    for age in ages:
        if age < age_min or age > age_max:
            print(f"  [WARN] Age {age} is outside data range [{age_min:.1f}, {age_max:.1f}]")
    
    quantile_edges = df_age[feature].quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_numpy()
    records: List[Dict[str, float]] = []
    
    for q_idx in range(4):
        lower, upper = quantile_edges[q_idx], quantile_edges[q_idx + 1]
        if q_idx < 3:
            mask = (df_age[feature] >= lower) & (df_age[feature] < upper)
        else:
            mask = (df_age[feature] >= lower) & (df_age[feature] <= upper)
        if mask.sum() == 0:
            continue
        feature_mean = float(df_age.loc[mask, feature].mean())
        
        for age in ages:
            # Convert requested age (in years) to z-score using ORIGINAL age statistics
            age_z = (age - age_mean) / age_std
   
            # Create prediction profile
            profile = pd.DataFrame({AGE_Z_COL: [age_z], feature: [feature_mean]})
            
            try:
                # Use lifelines' prediction method
                survival = model.predict_survival_function(profile, times=[horizon])
                S_t = float(survival.loc[horizon].values[0])
                risk = float(np.clip(1.0 - S_t, 0.0, 1.0))
                                
            except Exception as e:
                print(f"[ERROR] Prediction failed for Q{q_idx+1}, age={age}: {e}")
                risk = float("nan")
            
            records.append(
                {
                    "quartile": f"Q{q_idx + 1}",
                    "feature_mean": feature_mean,
                    "age": float(age),
                    "age_z": float(age_z),
                    "risk_horizon_years": float(horizon),
                    "predicted_risk": risk,
                }
            )
    
    return pd.DataFrame(records)


def compute_likelihood_metrics(
    baseline_model,
    full_model,
    n_baseline: int,
    n_full: int,
) -> Dict[str, float]:
    ll_base = getattr(baseline_model, "log_likelihood_", float("nan"))
    ll_full = getattr(full_model, "log_likelihood_", float("nan"))
    neg2ll_base = -2.0 * ll_base if np.isfinite(ll_base) else float("nan")
    neg2ll_full = -2.0 * ll_full if np.isfinite(ll_full) else float("nan")
    k_base = len(getattr(baseline_model, "params_", []))
    k_full = len(getattr(full_model, "params_", []))
    delta_neg2ll = (neg2ll_base - neg2ll_full) if np.all(np.isfinite([neg2ll_base, neg2ll_full])) else float("nan")
    lr_stat = 2.0 * (ll_full - ll_base) if np.all(np.isfinite([ll_base, ll_full])) else float("nan")
    df_diff = max(k_full - k_base, 0)
    lr_p = _chi2_survival(lr_stat, df_diff) if (df_diff > 0 and np.isfinite(lr_stat)) else float("nan")
    aic_base = neg2ll_base + 2 * k_base if np.isfinite(neg2ll_base) else float("nan")
    aic_full = neg2ll_full + 2 * k_full if np.isfinite(neg2ll_full) else float("nan")
    bic_base = neg2ll_base + k_base * np.log(n_baseline) if np.isfinite(neg2ll_base) and n_baseline > 0 else float("nan")
    bic_full = neg2ll_full + k_full * np.log(n_full) if np.isfinite(neg2ll_full) and n_full > 0 else float("nan")
    delta_aic = aic_base - aic_full if np.all(np.isfinite([aic_base, aic_full])) else float("nan")
    delta_bic = bic_base - bic_full if np.all(np.isfinite([bic_base, bic_full])) else float("nan")
    return {
        "neg2ll_base": neg2ll_base,
        "neg2ll_full": neg2ll_full,
        "delta_neg2ll": delta_neg2ll,
        "lr_stat": lr_stat,
        "lr_df": df_diff,
        "lr_p": lr_p,
        "aic_base": aic_base,
        "aic_full": aic_full,
        "delta_aic": delta_aic,
        "bic_base": bic_base,
        "bic_full": bic_full,
        "delta_bic": delta_bic,
    }


# ---- Reclassification summary & interpretation helper -----------------------
def _fmt_pct(x, digits=1):
    try:
        if x is None or not np.isfinite(float(x)):
            return "NA"
        return f"{100*float(x):.{digits}f}%"
    except Exception:
        return "NA"

def _fmt_dec(x, digits=4):
    try:
        if x is None or not np.isfinite(float(x)):
            return "NA"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "NA"

def summarize_reclassification(rec_metrics: dict, cohort_name: str, horizon_years: float = 5.0) -> str:
    # expected keys (your function already produces these)
    keys = ["n_used","n_events","n_nonevents","nri_events","nri_nonevents","nri_total",
            "idi_events","idi_nonevents","idi_total"]
    m = {k: rec_metrics.get(k, np.nan) for k in keys}

    n_used = int(m["n_used"]) if np.isfinite(m["n_used"]) else None
    n_events = int(m["n_events"]) if np.isfinite(m["n_events"]) else None
    n_nonevents = int(m["n_nonevents"]) if np.isfinite(m["n_nonevents"]) else None

    # narrative pieces based on direction/magnitude
    def _trend_word(x, pos="increased", neg="decreased", zero="unchanged"):
        if not np.isfinite(x): return "was " + zero
        if x > 0: return f"{pos}"
        if x < 0: return f"{neg}"
        return zero

    # thresholds for “small / moderate / large” (tunable)
    def _size_word(x):
        if not np.isfinite(x): return "NA"
        ax = abs(x)
        if ax < 0.02: return "very small"
        if ax < 0.05: return "small"
        if ax < 0.10: return "moderate"
        return "large"

    # Build interpretation lines
    lines = []
    lines.append(f"Reclassification analysis at {horizon_years:.0f} years ({cohort_name}).")
    lines.append("  • What it measures:")
    lines.append("    – NRI (Net Reclassification Improvement): net proportion moving in the correct risk direction when adding BHS to age.")
    lines.append("      NRI = [P(risk↑|event) − P(risk↓|event)] + [P(risk↓|non-event) − P(risk↑|non-event)]  (category-free).")
    lines.append("    – IDI (Integrated Discrimination Improvement): increase in mean risk for events plus decrease for non-events.")

    # Tabular summary (human-readable)
    lines.append("  • Results:")
    lines.append(f"    – Sample used: n={n_used}  (events={n_events}, non-events={n_nonevents})")
    lines.append(f"    – NRI_events:     { _fmt_pct(m['nri_events']) }")
    lines.append(f"    – NRI_nonevents:  { _fmt_pct(m['nri_nonevents']) }")
    lines.append(f"    – NRI_total:      { _fmt_pct(m['nri_total']) }")
    lines.append(f"    – IDI_events:     { _fmt_dec(m['idi_events']) }")
    lines.append(f"    – IDI_nonevents:  { _fmt_dec(m['idi_nonevents']) }")
    lines.append(f"    – IDI_total:      { _fmt_dec(m['idi_total']) }")

    # Automatic interpretation
    # Events
    ev_trend = _trend_word(m["nri_events"], pos="more deaths moved to higher risk", neg="some deaths moved incorrectly to lower risk")
    ev_size  = _size_word(m["nri_events"])
    lines.append("  • Interpretation:")
    if n_events is not None and n_events < 30:
        lines.append(f"    – Events were few (events={n_events}); NRI_events estimates may be unstable.")

    lines.append(f"    – For events: {ev_trend} ({_fmt_pct(m['nri_events'])} net; {ev_size} net up-classification).")

    # Nonevents
    ne_trend = _trend_word(m["nri_nonevents"], pos="more survivors moved to lower risk", neg="some survivors moved incorrectly to higher risk")
    ne_size  = _size_word(m["nri_nonevents"])
    lines.append(f"    – For non-events: {ne_trend} ({_fmt_pct(m['nri_nonevents'])} net; {ne_size} net down-classification).")

    # Total NRI
    tot_trend = _trend_word(m["nri_total"], pos="overall improvement in individualized risk assignment", neg="overall worsening in individualized risk assignment")
    tot_size  = _size_word(m["nri_total"])
    lines.append(f"    – Overall: {tot_trend} (NRI_total={_fmt_pct(m['nri_total'])}; {tot_size}).")

    # IDI narrative
    idi_ev_trend = _trend_word(m["idi_events"], pos="increased", neg="decreased")
    idi_ne_trend = _trend_word(m["idi_nonevents"], pos="decreased", neg="increased")
    lines.append(f"    – IDI: mean predicted risk for events {idi_ev_trend} by { _fmt_dec(m['idi_events']) }, "
                f"and for non-events {idi_ne_trend} by { _fmt_dec(m['idi_nonevents']) } "
                f"(IDI_total={ _fmt_dec(m['idi_total']) }).")

    # Gentle guidance on interpretation
    lines.append("    – Practical read: positive NRI_total and IDI_total indicate that adding BHS refines individual 5-year risk estimates in the right direction.")
    return "\n".join(lines)


def compute_reclassification_metrics(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    baseline_model,
    full_model,
    horizon: float,
) -> Dict[str, float]:
    risk_base = _predict_risk(baseline_model, df, horizon)
    risk_full = _predict_risk(full_model, df, horizon)

    mask_event = (df[event_col] == 1) & (df[duration_col] <= horizon)
    mask_nonevent = df[duration_col] > horizon
    mask = mask_event | mask_nonevent
    if mask.sum() == 0:
        return {}

    status = mask_event.loc[mask].astype(int)
    rb = risk_base.loc[mask]
    rf = risk_full.loc[mask]

    events = status == 1
    nonevents = status == 0

    metrics: Dict[str, float] = {
        "n_used": float(mask.sum()),
        "n_events": float(events.sum()),
        "n_nonevents": float(nonevents.sum()),
    }

    if events.any():
        p_up = float(((rf > rb) & events).sum()) / events.sum()
        p_down = float(((rf < rb) & events).sum()) / events.sum()
        metrics["nri_events"] = p_up - p_down
        metrics["idi_events"] = float(rf.loc[events].mean() - rb.loc[events].mean())
    else:
        metrics["nri_events"] = float("nan")
        metrics["idi_events"] = float("nan")

    if nonevents.any():
        p_down = float(((rf < rb) & nonevents).sum()) / nonevents.sum()
        p_up = float(((rf > rb) & nonevents).sum()) / nonevents.sum()
        metrics["nri_nonevents"] = p_down - p_up
        metrics["idi_nonevents"] = float(rb.loc[nonevents].mean() - rf.loc[nonevents].mean())
    else:
        metrics["nri_nonevents"] = float("nan")
        metrics["idi_nonevents"] = float("nan")

    if np.all(np.isfinite([metrics.get("nri_events", np.nan), metrics.get("nri_nonevents", np.nan)])):
        metrics["nri_total"] = metrics["nri_events"] + metrics["nri_nonevents"]
    else:
        metrics["nri_total"] = float("nan")

    if np.all(np.isfinite([metrics.get("idi_events", np.nan), metrics.get("idi_nonevents", np.nan)])):
        metrics["idi_total"] = metrics["idi_events"] + metrics["idi_nonevents"]
    else:
        metrics["idi_total"] = float("nan")

    return metrics


def evaluate_brain_health_metrics(
    cohort_key: str,
    df: pd.DataFrame,
    out_dir: os.PathLike | str,
    horizons: Sequence[float],
    risk_ages: Sequence[float],
    risk_horizon: float = 5.0,
) -> None:
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    # --- sanitize key once ---
    cohort_key = (
        cohort_key.lower().replace("/", "_").replace("\\", "_").replace(" ", "_")
    )

    feature = _resolved_brain_health_column(df)
    if feature is None:
        print(f"[WARN] {cohort_key}: No brain health score column available for extended metrics.")
        return

    required_cols = [DURATION_COL, EVENT_COL, AGE_Z_COL, feature]
    df_model = df.dropna(subset=required_cols).copy()
    if df_model.empty:
        print(f"[WARN] {cohort_key}: Insufficient data for extended metrics.")
        return

    try:
        from lifelines import CoxPHFitter  # type: ignore
    except Exception:
        print("[WARN] lifelines unavailable; skipping extended metrics.")
        return

    cph_age = CoxPHFitter()
    cph_age.fit(
        df_model[[DURATION_COL, EVENT_COL, AGE_Z_COL]],
        duration_col=DURATION_COL,
        event_col=EVENT_COL,
        formula=f"{AGE_Z_COL}",
    )

    cph_full = CoxPHFitter()
    cph_full.fit(
        df_model[[DURATION_COL, EVENT_COL, AGE_Z_COL, feature]],
        duration_col=DURATION_COL,
        event_col=EVENT_COL,
        formula=f"{AGE_Z_COL} + {feature}",
    )


    # clamp risk_horizon to support
    tmax = float(np.nanmax(df_model[DURATION_COL]))
    if risk_horizon >= 0.98 * tmax:
        print(f"[WARN] risk_horizon={risk_horizon} ≥ 98% of max follow-up ({tmax:.2f}); clamping.")
        risk_horizon = 0.98 * tmax

    # Baseline survival at t (debug)
    try:
        # find row at/after t
        idx = (cph_full.baseline_cumulative_hazard_.index >= risk_horizon).argmax()
        base_ch = float(cph_full.baseline_cumulative_hazard_.iloc[idx, 0])
        s0 = float(np.exp(-base_ch))
    except Exception:
        raise ValueError("Failed to compute baseline survival.")


    auc_df = compute_time_dependent_auc(df_model, DURATION_COL, EVENT_COL, cph_age, cph_full, horizons)
    if not auc_df.empty:
        path_auc = out_dir / f"time_dependent_auc_{cohort_key}.csv"
        auc_df.to_csv(path_auc, index=False)
        print(f"\n_______________________________________________________________")
        print(f"Time-dependent discrimination (IPCW C(t))")
        print(f'  The C(t) statistic quantifies how well the model discriminates who will die by time t vs. who will survive past t (horizon t in years from baseline).')
        print(f'  It’s analogous to an AUC but adapted to censored survival data using inverse probability of censoring weights (IPCW).')
        print("")
        auc_df_print = auc_df.pivot(index="horizon", columns="model", values="metric")
        auc_df_print = auc_df_print.round(3)
        print(auc_df_print)
        print("\n_______________________________________________________________\n")

    abs_df = compute_absolute_risk_by_quantiles(df_model, cph_full, feature, risk_ages, risk_horizon)
    if not abs_df.empty:
        path_abs = out_dir / f"absolute_risk_{cohort_key}.csv"
        abs_df.to_csv(path_abs, index=False)

        print("Fitted Model interpretation: Absolute risk predictions by Brain Health Score quartile")
        print(f"Model-predicted {risk_horizon}-year absolute mortality risk by Brain Health Score quartile (age-adjusted Cox model).")
        print("Risks represent the predicted probability of death within 5 years, computed from the fitted Cox model at representative ages and the mean Brain Health Score within each quartile (Q1 = lowest, Q4 = highest).")
        print('')
        abs_df_print = abs_df.drop(columns=["age_z"])
        abs_df_print = abs_df_print.rename(columns={"feature_mean": "mean_bhs_in_quartile"})
        # rename predicted_risk to make it clear: predicted_risk_of_death_at_horizon
        abs_df_print = abs_df_print.rename(columns={"predicted_risk": f"predicted_risk_of_death_at_horizon"})
        abs_df_print = abs_df_print.round({"mean_bhs_in_quartile": 2, "predicted_risk_of_death_at_horizon": 3})
        # split the print into age groups:
        for age in risk_ages:
            mask_age = abs_df_print["age"] == age
            if not mask_age.any():
                continue
            print(f"Age: {age} years")
            abs_df_print_age = abs_df_print.loc[mask_age].drop(columns=["age"])
            print(abs_df_print_age)
        print("")

        print("_______________________________________________________________\n")


   # KM sanity check
    from lifelines import KaplanMeierFitter
    km = KaplanMeierFitter()
    q = pd.qcut(df_model[feature], 4, labels=["Q1","Q2","Q3","Q4"])
    km_abs = []
    for qlab in ["Q1","Q2","Q3","Q4"]:
        mask = q == qlab
        if mask.sum() < 5: 
            continue
        km.fit(df_model.loc[mask, DURATION_COL], event_observed=df_model.loc[mask, EVENT_COL])
        s = float(km.survival_function_at_times(risk_horizon).ffill().iloc[-1])
        km_abs.append({"quartile": qlab, "km_risk": 1.0 - s, "horizon": risk_horizon})
    if km_abs:
        km_abs_df = pd.DataFrame(km_abs)
        km_abs_df.to_csv(out_dir / f"km_absolute_risk_{cohort_key}.csv", index=False)
        print(f"Kaplan-Meier (sanity check): Absolute risk at {risk_horizon:.0f} years from sleep study by Brain Health Score quartile")
        print(km_abs_df)
        print("")
        print("_______________________________________________________________\n")

    lik_metrics = compute_likelihood_metrics(cph_age, cph_full, df_model.shape[0], df_model.shape[0])
    path_lik = out_dir / f"likelihood_metrics_{cohort_key}.csv"
    pd.DataFrame([lik_metrics]).to_csv(path_lik, index=False)
    print(f"Analysis on whether adding Brain Health Score improves model fit over age alone\n")
    print(f"Likelihood-ratio test comparing age-only vs. age + Brain Health Score Cox models")
    # do variable-specific prints of main vairables with small explanation:
    print(f"  Likelihood-ratio statistic: {lik_metrics['lr_stat']:.2f} on {int(lik_metrics['lr_df'])} df, p = {lik_metrics['lr_p']:.3g}")
    print(f"  AIC improvement (delta AIC): {lik_metrics['delta_aic']:.2f} (delta of AIC > 10 is considered strong evidence)")
    print(f"  BIC improvement (delta BIC): {lik_metrics['delta_bic']:.2f} (delta of BIC > 10 is considered strong evidence)")
    print(f"\nDetailed metrics:")
    lik_metrics_print = lik_metrics.copy()
    for key in lik_metrics_print:
        if key != "lr_p":
            lik_metrics_print[key] = round(lik_metrics_print[key], 1)
    print(pd.DataFrame([lik_metrics_print]))
    print("_______________________________________________________________\n")


    print(f"Reclassification analysis. Do individual patients get better classified into appropriate risk levels for the event of interest (death within X years)?\n")

    for risk_horizon in [5, 8, 10, 12]:
        rec_metrics = compute_reclassification_metrics(df_model, DURATION_COL, EVENT_COL, cph_age, cph_full, risk_horizon)
        print(summarize_reclassification(rec_metrics, cohort_name=cohort_key.upper(), horizon_years=risk_horizon))
        print('\n')
        
    print("\n_______________________________________________________________\n")

# -----------------------------
# Cohort analysis functions
# -----------------------------

def evaluate_bai_brain_health_comparison(
    cohort_label: str,
    df: pd.DataFrame,
    out_dir: os.PathLike | str,
    horizons: Sequence[float],
) -> None:
    out_dir = Path(out_dir)
    required_cols = [DURATION_COL, EVENT_COL, AGE_Z_COL, "brain_health_score", "BAI"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"[WARN] {cohort_label}: missing columns for combined biomarker analysis: {', '.join(missing_cols)}")
        return

    df_model = df.dropna(subset=required_cols).copy()
    if df_model.empty:
        print(f"[WARN] {cohort_label}: insufficient overlapping data for combined biomarker analysis.")
        return

    try:
        from lifelines import CoxPHFitter  # type: ignore
    except Exception:
        print("[WARN] lifelines unavailable; skipping combined biomarker analysis.")
        return

    console_header(f"{cohort_label}: BAI + brain health score comparisons")

    cph_age = CoxPHFitter()
    cph_age.fit(
        df_model[[DURATION_COL, EVENT_COL, AGE_Z_COL]],
        duration_col=DURATION_COL,
        event_col=EVENT_COL,
        formula=f"{AGE_Z_COL}",
    )

    cph_age_bhs = CoxPHFitter()
    cph_age_bhs.fit(
        df_model[[DURATION_COL, EVENT_COL, AGE_Z_COL, "brain_health_score"]],
        duration_col=DURATION_COL,
        event_col=EVENT_COL,
        formula=f"{AGE_Z_COL} + brain_health_score",
    )

    cph_age_bai = CoxPHFitter()
    cph_age_bai.fit(
        df_model[[DURATION_COL, EVENT_COL, AGE_Z_COL, "BAI"]],
        duration_col=DURATION_COL,
        event_col=EVENT_COL,
        formula=f"{AGE_Z_COL} + BAI",
    )

    cph_age_both = CoxPHFitter()
    cph_age_both.fit(
        df_model[[DURATION_COL, EVENT_COL, AGE_Z_COL, "brain_health_score", "BAI"]],
        duration_col=DURATION_COL,
        event_col=EVENT_COL,
        formula=f"{AGE_Z_COL} + brain_health_score + BAI",
    )

    models = {
        "age": cph_age,
        "age+brain_health_score": cph_age_bhs,
        "age+BAI": cph_age_bai,
        "age+brain_health_score+BAI": cph_age_both,
    }
    covariate_map: Dict[str, Sequence[str]] = {
        "age": [AGE_Z_COL],
        "age+brain_health_score": [AGE_Z_COL, "brain_health_score"],
        "age+BAI": [AGE_Z_COL, "BAI"],
        "age+brain_health_score+BAI": [AGE_Z_COL, "brain_health_score", "BAI"],
    }
    cindex_ci: Dict[str, Tuple[float, float]] = {}
    for idx, (label, covars) in enumerate(covariate_map.items()):
        ci_low, ci_high = bootstrap_cindex_ci(
            df_model,
            DURATION_COL,
            EVENT_COL,
            covars,
            n_bootstrap=CINDEX_BOOTSTRAP_REPS,
            random_state=CINDEX_BOOTSTRAP_RANDOM_STATE + 200 + idx,
        )
        cindex_ci[label] = (ci_low, ci_high)

    n_obs = df_model.shape[0]
    lik_records: List[Dict[str, float]] = []
    lik_records.append(
        {"comparison": "age -> age+brain_health_score", **compute_likelihood_metrics(cph_age, cph_age_bhs, n_obs, n_obs)}
    )
    lik_records.append(
        {"comparison": "age -> age+BAI", **compute_likelihood_metrics(cph_age, cph_age_bai, n_obs, n_obs)}
    )
    lik_records.append(
        {
            "comparison": "age+brain_health_score -> +BAI",
            **compute_likelihood_metrics(cph_age_bhs, cph_age_both, n_obs, n_obs),
        }
    )
    lik_records.append(
        {
            "comparison": "age+BAI -> +brain_health_score",
            **compute_likelihood_metrics(cph_age_bai, cph_age_both, n_obs, n_obs),
        }
    )

    df_lik = pd.DataFrame(lik_records)
    lik_cols = [
        "comparison",
        "lr_stat",
        "lr_df",
        "lr_p",
        "delta_aic",
        "delta_bic",
        "delta_neg2ll",
    ]
    lik_print = df_lik[lik_cols].copy()
    lik_print[["lr_stat", "delta_aic", "delta_bic", "delta_neg2ll"]] = lik_print[
        ["lr_stat", "delta_aic", "delta_bic", "delta_neg2ll"]
    ].round(2)
    lik_print["lr_p"] = lik_print["lr_p"].apply(lambda x: f"{x:.3g}" if np.isfinite(x) else "NA")
    lik_print["lr_df"] = lik_print["lr_df"].astype(int)

    ensure_dir(out_dir)
    lik_path = out_dir / f"bai_bhs_likelihood_{cohort_label.lower()}.csv"
    df_lik.to_csv(lik_path, index=False)
    print("Model fit comparisons (likelihood / information criteria):")
    print(lik_print.to_string(index=False))

    cindex_records = []
    for label, model in models.items():
        ci_low, ci_high = cindex_ci.get(label, (float("nan"), float("nan")))
        cindex_records.append(
            {
                "model": label,
                "concordance_index": float(model.concordance_index_),
                "c_index_ci_lower": ci_low,
                "c_index_ci_upper": ci_high,
            }
        )
    df_cindex = pd.DataFrame(cindex_records).sort_values("model")
    for col in ["concordance_index", "c_index_ci_lower", "c_index_ci_upper"]:
        if col in df_cindex.columns:
            df_cindex[col] = df_cindex[col].round(3)
    print("\nHarrell's C-index (in-sample) for each model:")
    print(df_cindex.to_string(index=False))

    unique_horizons = tuple(dict.fromkeys(float(h) for h in horizons))
    auc_df = compute_time_dependent_auc(
        df_model,
        DURATION_COL,
        EVENT_COL,
        baseline_model=None,
        full_model=None,
        horizons=unique_horizons,
        models=models,
    )
    if not auc_df.empty:
        auc_path = out_dir / f"bai_bhs_time_dependent_auc_{cohort_label.lower()}.csv"
        auc_df.to_csv(auc_path, index=False)
        auc_pivot = auc_df.pivot(index="horizon", columns="model", values="metric").sort_index()
        print("\nTime-dependent discrimination (IPCW C(t)) across models:")
        print(auc_pivot.round(3))

        combined_label = "age+brain_health_score+BAI"
        if combined_label in auc_pivot.columns:
            base_labels = ["age", "age+brain_health_score", "age+BAI"]
            print("\nΔC(t) for combined model relative to simpler models:")
            for horizon in auc_pivot.index:
                combined_val = auc_pivot.loc[horizon, combined_label]
                if not np.isfinite(combined_val):
                    continue
                horizon_str = f"{float(horizon):g}y"
                print(f"  Horizon {horizon_str}: combined C(t) = {combined_val:.3f}")
                for label in base_labels:
                    if label not in auc_pivot.columns:
                        continue
                    base_val = auc_pivot.loc[horizon, label]
                    if np.isfinite(base_val):
                        delta = combined_val - base_val
                        print(f"    Δ vs {label}: {delta:+.3f}")
    else:
        print("[WARN] Unable to compute time-dependent discrimination for combined models.")

    print("")

def _prepare_common_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.replace({np.inf: np.nan, -np.inf: np.nan})
    for col in [AGE_COL, AGE_Z_COL, DURATION_COL, EVENT_COL]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    ensure_age_z(data)
    return data


def _collect_features(data: pd.DataFrame) -> List[str]:
    sleep = [c for c in FEATURES_SLEEP_DEFAULT if c in data.columns]
    cognition = [c for c in COLS_COGNITION_DEFAULT if c in data.columns]
    extras = [c for c in ["brain_health_score", "BAI"] if c in data.columns]
    features = list(dict.fromkeys(sleep + cognition + extras))
    if features:
        zscore_inplace(data, features)
    return features


def analyze_cohort(
    cohort_label: str,
    csv_path: os.PathLike | str,
    horizons: Sequence[float],
    risk_ages: Sequence[float],
    risk_horizon: float,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    console_header(f"\033[93m{cohort_label} -- Survival Analysis\033[0m")
    csv_path = Path(csv_path)
    data = safe_read_csv(csv_path, index_col=0)
    if data is None:
        print(f"[INFO] Skipping {cohort_label}: missing {csv_path}")
        return None

    data.index = data.index.astype(str)
    data = data.dropna(subset=[DURATION_COL, EVENT_COL])
    data = _prepare_common_columns(data)

    bai_raw = data["BAI"].copy() if "BAI" in data.columns else None

    features = _collect_features(data)

    if AGE_Z_COL in data.columns:
        data = data.dropna(subset=[DURATION_COL, EVENT_COL, AGE_Z_COL])
    else:
        data = data.dropna(subset=[DURATION_COL, EVENT_COL])

    if bai_raw is not None:
        bai_raw = bai_raw.loc[data.index]

    if data.empty:
        print(f"[WARN] {cohort_label}: no data after filtering.")
        return None

    print_survival_baseline_summary(cohort_label, data)


    console_header(f"{cohort_label}: Cox models (age + feature)")
    age_model = None
    age_cidx = float("nan")
    age_ci_bounds: Tuple[float, float] = (float("nan"), float("nan"))
    if AGE_Z_COL in data.columns:
        age_model, age_cidx = fit_age_baseline(data, DURATION_COL, EVENT_COL, age_col=AGE_Z_COL)
        age_ci_bounds = bootstrap_cindex_ci(
            data,
            DURATION_COL,
            EVENT_COL,
            [AGE_Z_COL],
            n_bootstrap=CINDEX_BOOTSTRAP_REPS,
            random_state=CINDEX_BOOTSTRAP_RANDOM_STATE,
        )
        if np.all(np.isfinite(age_ci_bounds)):
            print(
                f"{cohort_label} baseline age-only C-index: {age_cidx:.3f} "
                f"(95% CI {age_ci_bounds[0]:.3f}-{age_ci_bounds[1]:.3f})"
            )
        else:
            print(f"{cohort_label} baseline age-only C-index: {age_cidx:.3f}")
    else:
        print(f"[WARN] {cohort_label}: missing age_z; baseline C-index unavailable.")

    if age_model is not None and AGE_Z_COL in age_model.summary.index:
        row = age_model.summary.loc[AGE_Z_COL]
        age_hr = float(row["exp(coef)"])
        age_lo = float(row["exp(coef) lower 95%"])
        age_hi = float(row["exp(coef) upper 95%"])
        age_p = float(row["p"])
        print(f"{cohort_label} age-only HR: {age_hr:.2f} ({age_lo:.2f}-{age_hi:.2f}), p={age_p:.3g}")

    df_res = run_featurewise_cox(
        data=data,
        features=features,
        duration_col=DURATION_COL,
        event_col=EVENT_COL,
        adjust=(AGE_Z_COL,) if AGE_Z_COL in data.columns else (),
        age_baseline_c_index=age_cidx if np.isfinite(age_cidx) else None,
        bootstrap_cindex=CINDEX_BOOTSTRAP_REPS,
        random_state=CINDEX_BOOTSTRAP_RANDOM_STATE,
    )

    out_dir = OUTPUT_ROOT / cohort_label.lower()
    plots_dir = out_dir / "plots"
    ensure_dir(out_dir)
    ensure_dir(plots_dir)

    df_res.to_csv(out_dir / "results.csv")
    if not df_res.empty and "p_value" in df_res.columns:
        print(format_results_table(df_res.sort_values("p_value")).head(15))
    else:
        print(f"[WARN] {cohort_label}: no valid feature models were fitted.")

    if "BAI" in data.columns:
        biomarker_features: List[str] = []
        if "brain_health_score" in data.columns:
            biomarker_features.append("brain_health_score")
        biomarker_features.append("BAI")

        console_header("BAI survival association")
        df_bai = run_featurewise_cox(
            data=data,
            features=biomarker_features,
            duration_col=DURATION_COL,
            event_col=EVENT_COL,
            adjust=(AGE_Z_COL,) if AGE_Z_COL in data.columns else (),
            age_baseline_c_index=age_cidx if np.isfinite(age_cidx) else None,
            bootstrap_cindex=CINDEX_BOOTSTRAP_REPS,
            random_state=CINDEX_BOOTSTRAP_RANDOM_STATE + 101,
        )
        df_bai = df_bai.loc[[feat for feat in biomarker_features if feat in df_bai.index]]
        if not df_bai.empty:
            print(format_results_table(df_bai))
        else:
            print(f"[WARN] {cohort_label}: unable to fit age-adjusted Cox models for BAI/brain_health_score.")

        if "BAI" in df_bai.index:
            bai_row = df_bai.loc["BAI"]
            hr = float(bai_row.get("hazard_ratio", np.nan))
            ci_l = float(bai_row.get("ci_lower", np.nan))
            ci_u = float(bai_row.get("ci_upper", np.nan))
            p_val = float(bai_row.get("p_value", np.nan))
            delta_c = float(bai_row.get("delta_c_index_vs_age", np.nan))
            c_idx = float(bai_row.get("c_index", np.nan))
            c_ci_l = float(bai_row.get("c_index_ci_lower", np.nan))
            c_ci_u = float(bai_row.get("c_index_ci_upper", np.nan))
        else:
            hr = ci_l = ci_u = p_val = delta_c = float("nan")
            c_idx = c_ci_l = c_ci_u = float("nan")

        if bai_raw is not None and bai_raw.notna().sum() > 0:
            bai_valid = bai_raw.dropna()
            q1 = float(bai_valid.quantile(0.25))
            median = float(bai_valid.median())
            q3 = float(bai_valid.quantile(0.75))
            print("\nBAI distribution (raw units) among analyzed participants:")
            print(f"  n = {bai_valid.shape[0]}")
            print(f"  Mean (SD): {float(bai_valid.mean()):.2f} ({float(bai_valid.std()):.2f})")
            print(f"  Median (IQR): {median:.2f} ({q1:.2f}-{q3:.2f})")
            print(f"  Range: [{float(bai_valid.min()):.2f}, {float(bai_valid.max()):.2f}]")
        else:
            print("\n[WARN] BAI raw distribution unavailable after filtering.")

        if np.isfinite(hr) and np.isfinite(ci_l) and np.isfinite(ci_u):
            stat_line = f"Age-adjusted hazard ratio per 1 SD increase in BAI: {hr:.2f} (95% CI {ci_l:.2f}-{ci_u:.2f})"
            if np.isfinite(p_val):
                stat_line += f", p={p_val:.3g}"
            print(stat_line)
        if np.isfinite(c_idx):
            if np.isfinite(c_ci_l) and np.isfinite(c_ci_u):
                print(f"  Age-adjusted C-index: {c_idx:.3f} (95% CI {c_ci_l:.3f}-{c_ci_u:.3f})")
            else:
                print(f"  Age-adjusted C-index: {c_idx:.3f}")
        if np.isfinite(delta_c):
            print(f"  ΔC-index vs age-only: {delta_c:+.3f}")
        print("")

    if {"BAI", "brain_health_score"}.issubset(data.columns):
        evaluate_bai_brain_health_comparison(
            cohort_label=cohort_label,
            df=data,
            out_dir=out_dir,
            horizons=horizons,
        )

    for feature in ["brain_health_score", "perc_r", "fs_dens_c4m1"]:
        if feature in data.columns:
            try:
                fig, ax = plot_km_by_quantiles(data, feature, DURATION_COL, EVENT_COL, title=f"{cohort_label}: {feature}")
                fig.savefig(plots_dir / f"km_{feature}.png", dpi=300, bbox_inches="tight")
            except Exception as exc:
                print(f"[WARN] KM plot failed for {feature}: {exc}")

    evaluate_brain_health_metrics(
        cohort_key=cohort_label.lower(),
        df=data,
        out_dir=OUTPUT_ROOT,
        horizons=horizons,
        risk_ages=risk_ages,
        risk_horizon=risk_horizon,
    )

    return data, df_res


# -----------------------------
# Combined figures and tables
# -----------------------------

def _hr_string(row: pd.Series) -> str:
    try:
        return f"{row['hazard_ratio']:.2f} ({row['ci_lower']:.2f}-{row['ci_upper']:.2f})"
    except Exception:
        return np.nan


FEATURE_FRIENDLY = {
    "perc_r": "REM sleep %",
    "fs_dens_c4m1": "Spindle density",
    "cog_total": "Total cognition (neuropsychological test)",
    "brain_health_score": "Brain health score (Sleep EEG + DL)",
}

COHORT_LABELS = {"mros": "MrOS", "shhs": "FHS", "fhs": "FHS", "mgh": "MGH", "bidmc": "BIDMC"}


def build_results_long_and_matrix(
    mros_res: Optional[pd.DataFrame],
    fhs_res: Optional[pd.DataFrame],
    mgh_res: Optional[pd.DataFrame],
    bidmc_res: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    records: List[Dict[str, object]] = []
    matrices: Dict[str, Dict[str, str]] = {"MrOS": {}, "FHS": {}, "MGH": {}, "BIDMC": {}}

    def ingest(res: Optional[pd.DataFrame], cohort_key: str) -> None:
        if res is None or res.empty:
            return
        label = COHORT_LABELS[cohort_key]
        for raw_feature, friendly in [
            ("perc_r", FEATURE_FRIENDLY["perc_r"]),
            ("fs_dens_c4m1", FEATURE_FRIENDLY["fs_dens_c4m1"]),
            ("cog_total", FEATURE_FRIENDLY["cog_total"]),
            ("brain_health_score", FEATURE_FRIENDLY["brain_health_score"]),
        ]:
            if raw_feature not in res.index:
                continue
            row = res.loc[raw_feature]
            hr_text = _hr_string(row)
            records.append(
                {
                    "cohort": label,
                    "feature": friendly,
                    "hazard_ratio": row.get("hazard_ratio", np.nan),
                    "ci_lower": row.get("ci_lower", np.nan),
                    "ci_upper": row.get("ci_upper", np.nan),
                    "p_value": row.get("p_value", np.nan),
                    "hr_str": hr_text,
                }
            )
            matrices[label][friendly] = hr_text

    ingest(mros_res, "mros")
    ingest(fhs_res, "shhs")
    ingest(mgh_res, "mgh")
    ingest(bidmc_res, "bidmc")

    df_long = pd.DataFrame.from_records(records)
    order = [
        FEATURE_FRIENDLY["perc_r"],
        FEATURE_FRIENDLY["fs_dens_c4m1"],
        FEATURE_FRIENDLY["cog_total"],
        FEATURE_FRIENDLY["brain_health_score"],
    ]
    df_matrix = pd.DataFrame(matrices).reindex(order)
    return df_long, df_matrix


def wald_equal_betas_pvalue(cph, name_a: str, name_b: str) -> float:
    params = cph.params_
    cov = cph.variance_matrix_
    if name_a not in params.index or name_b not in params.index:
        return np.nan
    delta = float(params[name_a] - params[name_b])
    var_delta = float(cov.loc[name_a, name_a] + cov.loc[name_b, name_b] - 2 * cov.loc[name_a, name_b])
    if var_delta <= 0 or not np.isfinite(var_delta):
        return np.nan
    z = delta / sqrt(var_delta)
    return erfc(abs(z) / sqrt(2.0))


def compute_pvals_vs_bhs(
    data_mros: Optional[pd.DataFrame],
    data_fhs: Optional[pd.DataFrame],
    data_mgh: Optional[pd.DataFrame],
    data_bidmc: Optional[pd.DataFrame],
) -> pd.DataFrame:
    feats = [
        ("perc_r", FEATURE_FRIENDLY["perc_r"]),
        ("fs_dens_c4m1", FEATURE_FRIENDLY["fs_dens_c4m1"]),
        ("cog_total", FEATURE_FRIENDLY["cog_total"]),
    ]
    results = pd.DataFrame(
        index=[friendly for _, friendly in feats],
        columns=["MrOS", "FHS", "MGH", "BIDMC"],
        dtype=float,
    )

    for dataset, cohort_name in [
        (data_mros, "MrOS"),
        (data_fhs, "FHS"),
        (data_mgh, "MGH"),
        (data_bidmc, "BIDMC"),
    ]:
        if dataset is None or dataset.empty:
            continue
        bhs_col = _resolved_brain_health_column(dataset)
        if bhs_col is None or AGE_Z_COL not in dataset.columns:
            print(f"[WARN] {cohort_name}: brain health score or age_z missing for hazard comparison.")
            continue
        try:
            from lifelines import CoxPHFitter  # type: ignore
        except Exception:
            print("[WARN] lifelines unavailable; skipping hazard comparisons.")
            break
        for raw_feature, friendly in feats:
            if raw_feature not in dataset.columns:
                continue
            cols_needed = [DURATION_COL, EVENT_COL, AGE_Z_COL, raw_feature, bhs_col]
            df_fit = dataset.dropna(subset=cols_needed)
            if df_fit.empty:
                continue
            cph = CoxPHFitter()
            try:
                cph.fit(
                    df_fit,
                    duration_col=DURATION_COL,
                    event_col=EVENT_COL,
                    formula=f"{AGE_Z_COL} + {raw_feature} + {bhs_col}",
                )
                results.loc[friendly, cohort_name] = wald_equal_betas_pvalue(cph, raw_feature, bhs_col)
            except Exception:
                results.loc[friendly, cohort_name] = np.nan
    return results


# Consistent layout parameters
LAYOUT_KW = dict(
    left=0.08,
    right=0.995,
    bottom=0.16,
    top=0.82,
    wspace=0.25,
)

TIGHT_KW = dict(pad=0.3, w_pad=0.2, h_pad=0.2)

def plot_survival_curves_part_a(
    datasets: Dict[str, pd.DataFrame],
    plot_savedir: os.PathLike | str,
    n_bootstraps: int = 200,
    fontsize_title: int = 11,
    fontsize_subtitle: int = 10,
    fontsize_label: int = 8,
) -> None:
    plot_savedir = Path(plot_savedir)
    try:
        from lifelines import CoxPHFitter  # type: ignore
    except Exception:
        print("[WARN] lifelines unavailable; skipping Part A survival plots.")
        return

    fig, ax = plt.subplots(1, 4, figsize=(7, 2.5))
    cohort_order = ["mros", "shhs", "mgh", "bidmc"]
    colors = ["#E69F00", "#56B4E9", "#009E73"]

    for i, cohort_name in enumerate(cohort_order):
        if cohort_name not in datasets:
            ax[i].axis("off")
            continue
        data = datasets[cohort_name].copy()
        main_feature = _resolved_brain_health_column(data)
        if main_feature is None or AGE_COL not in data.columns:
            continue

        df_fit = data.dropna(subset=[DURATION_COL, EVENT_COL, AGE_COL, main_feature])
        if df_fit.empty:
            continue

        cph = CoxPHFitter()
        cph.fit(
            df_fit,
            duration_col=DURATION_COL,
            event_col=EVENT_COL,
            formula=f"{AGE_COL} + {main_feature}",
        )

        median_age = data[AGE_COL].median()
        with pd.option_context("mode.chained_assignment", None):
            data["quantile_group"] = pd.qcut(
                data[main_feature],
                q=[0, 0.25, 0.75, 1],
                labels=["0-0.25", "0.25-0.75", "0.75-1"],
            )

        quantiles = ["0-0.25", "0.25-0.75", "0.75-1"]
        for j, quantile in enumerate(quantiles):
            data_q = data.loc[data["quantile_group"] == quantile]
            profile = pd.DataFrame({AGE_COL: [median_age], main_feature: [data_q[main_feature].mean()]})
            survival = cph.predict_survival_function(profile)
            ax[i].plot(survival.index, survival.values, label=f"Quantile {quantile}", color=colors[j])

            x_vals = survival.index
            survivors = []
            for _ in range(int(n_bootstraps)):
                boot = data.sample(frac=1, replace=True)
                cph_boot = CoxPHFitter()
                try:
                    cph_boot.fit(
                        boot.dropna(subset=[DURATION_COL, EVENT_COL, AGE_COL, main_feature]),
                        duration_col=DURATION_COL,
                        event_col=EVENT_COL,
                        formula=f"{AGE_COL} + {main_feature}",
                    )
                    profile_boot = pd.DataFrame({AGE_COL: [median_age], main_feature: [data_q[main_feature].mean()]})
                    surv_boot = cph_boot.predict_survival_function(profile_boot)
                    surv_boot = surv_boot.reindex(x_vals).interpolate()
                    survivors.append(surv_boot)
                except Exception:
                    continue
            if len(survivors) >= 5:
                surv_cat = pd.concat(survivors, axis=1)
                ci_lower = surv_cat.quantile(0.025, axis=1)
                ci_upper = surv_cat.quantile(0.975, axis=1)
                ax[i].fill_between(survival.index, ci_lower, ci_upper, color=colors[j], alpha=0.20)


        # x-axis from 0 to 0.999 percentile:
        x_max = df_fit[DURATION_COL].quantile(0.999)
        ax[i].set_xlim(0, x_max)
        ax[i].tick_params(axis="both", which="major", labelsize=fontsize_label, pad=1, length=2)
        ax[i].set_title(f"{COHORT_LABELS.get(cohort_name, cohort_name)}", fontsize=fontsize_subtitle)
        ax[i].grid(True, alpha=0.3)
        if i == 0:
            ax[i].set_ylabel("Survival Probability (age adjusted)", fontsize=fontsize_label, labelpad=1)
            ax[i].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
        else:
            ax[i].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))

        if i == 1:
            ax[i].legend(title=None, loc="upper left", bbox_to_anchor=(-0.80, 1.30), ncol=3, frameon=False, fontsize=fontsize_label)
            # ax[i].set_xlabel("Time (years)", fontsize=fontsize_label, labelpad=1, 
            ax[i].text(1, -0.14, "Time (years)", transform=ax[i].transAxes, fontsize=fontsize_label, ha="center", va="center")
            
        try:
            n = df_fit.shape[0]
            deaths = int(df_fit[EVENT_COL].sum())
            perc_death = deaths / max(n, 1) * 100.0
            hr = float(cph.summary.loc[main_feature, "exp(coef)"])
            p_val = float(cph.summary.loc[main_feature, "p"])
            c_index = float(getattr(cph, "concordance_index_", np.nan))
            c_ci_l, c_ci_u = bootstrap_cindex_ci(
                df_fit,
                DURATION_COL,
                EVENT_COL,
                [AGE_COL, main_feature],
                n_bootstrap=CINDEX_BOOTSTRAP_REPS,
                random_state=CINDEX_BOOTSTRAP_RANDOM_STATE + i,
            )
            if p_val < 0.0001:
                p_txt = "<0.0001"
            elif p_val < 0.01:
                p_txt = f"{p_val:.4f}"
            else:
                p_txt = f"{p_val:.2f}"
            if np.isfinite(c_index):
                if np.isfinite(c_ci_l) and np.isfinite(c_ci_u):
                    c_txt = f"C-index: {c_index:.2f} [{c_ci_l:.2f}, {c_ci_u:.2f}]"
                else:
                    c_txt = f"C-index: {c_index:.2f}"
            else:
                c_txt = "C-index: NA"
            ax[i].text(
                0.03,
                0.14,
                f"N: {n:,}\nMortality: {perc_death:.0f}%\nHR: {hr:.2f} (p: {p_txt})\n{c_txt}",
                transform=ax[i].transAxes,
                fontsize=fontsize_label-1,
                va="center",
                ha="left",
            )

            # if cohort_name.lower() in ["fhs", "shhs", "mgh", "bidmc"]:
                # ax[i].set_ylim(0.75, 1)
            # else:
            #     ax[i].set_ylim(0, 1)
            # keep lower lim but set upper lim to 1:
            ax[i].set_ylim(bottom=ax[i].get_ylim()[0], top=1.0)
        except Exception:
            pass

    # fig.suptitle("Brain health score predicts mortality risk", y=1, fontsize=fontsize_title, x=0.50)
    ensure_dir(plot_savedir)
    plt.tight_layout(**TIGHT_KW)
    LAYOUT_A_KW = LAYOUT_KW.copy()
    LAYOUT_A_KW["bottom"] = LAYOUT_A_KW["bottom"] - 0.03
    fig.subplots_adjust(**LAYOUT_A_KW)

    fig.savefig(plot_savedir / "figure_brain_health_score_survival_curves_part_a.png", dpi=300, bbox_inches=None)
    fig.savefig(plot_savedir / "figure_brain_health_score_survival_curves_part_a.pdf", dpi=600, bbox_inches=None)


def plot_comparative_bars_part_b(
    df_results_hr_matrix: pd.DataFrame,
    p_vals_hazard_comparisons: pd.DataFrame,
    plot_savedir: os.PathLike | str,
    fontsize_title: int = 11,
    fontsize_subtitle: int = 10,
    fontsize_label: int = 8,
) -> None:
    plot_savedir = Path(plot_savedir)
    features = [
        FEATURE_FRIENDLY["perc_r"],
        FEATURE_FRIENDLY["fs_dens_c4m1"],
        FEATURE_FRIENDLY["cog_total"],
        FEATURE_FRIENDLY["brain_health_score"],
    ]
    cohorts = list(df_results_hr_matrix.columns)
    n_cols = max(len(cohorts), 4)
    fig, axes = plt.subplots(1, n_cols, figsize=(7, 2.5), sharey=True)
    axes = np.atleast_1d(axes)
    bar_colors = ["#E0E0E0", "#A0A0A0", "#606060", "#303030"]

    for i, cohort in enumerate(cohorts):
        ax = axes[i]
        cohort_data = df_results_hr_matrix[cohort]
        means, yerr_lower, yerr_upper = [], [], []
        for feat in features:
            hr_string = cohort_data.get(feat, np.nan)
            if pd.notna(hr_string):
                try:
                    val = float(hr_string.split()[0])
                    low = float(hr_string.split("(")[1].split("-")[0])
                    high = float(hr_string.split("-")[1].split(")")[0])
                except Exception:
                    val = low = high = np.nan
            else:
                val = low = high = np.nan
            if np.isnan(val):
                means.append(np.nan)
                yerr_lower.append(np.nan)
                yerr_upper.append(np.nan)
            else:
                means.append(val)
                yerr_lower.append(val - low)
                yerr_upper.append(high - val)
        x_positions = np.arange(len(features))
        bars = ax.bar(x_positions, means, yerr=[yerr_lower, yerr_upper], color=bar_colors, alpha=0.8, capsize=5)

        for j, feat in enumerate(features):
            if feat == FEATURE_FRIENDLY["brain_health_score"]:
                continue
            p_val = p_vals_hazard_comparisons.loc[feat, cohort] if feat in p_vals_hazard_comparisons.index else np.nan
            if pd.isna(p_val):
                continue
            if p_val < 0.001:
                p_txt = "***"
            elif p_val < 0.01:
                p_txt = "**"
            elif p_val < 0.05:
                p_txt = "*"
            elif p_val < 0.10:
                p_txt = f"p={p_val:.2f}"
            else:
                p_txt = np.nan
            if pd.notna(p_txt):
                bhs_idx = features.index(FEATURE_FRIENDLY["brain_health_score"])
                y_base = 1.1 - 0.1 * j
                ax.plot([j, bhs_idx], [y_base, y_base], color="#303030", lw=1)
                ax.text((j + bhs_idx) / 2, y_base, p_txt, ha="center", va="bottom", fontsize=fontsize_label, color="#303030")

        ax.set_title(cohort, fontsize=fontsize_subtitle)
        ax.set_xticks([])
        ax.set_ylim(0, 1.2)
        ax.tick_params(axis="both", which="major", labelsize=fontsize_label, pad=1, length=2)
        if i == 0:
            ax.set_ylabel("Hazard ratio per 1 SD\n(age adjusted)", fontsize=fontsize_label, labelpad=1)
            
            
            ax.legend(bars, features, loc='lower center', bbox_to_anchor=(0.50, -0.03), ncol=2, frameon=False, 
                      fontsize=fontsize_label,
                      bbox_transform=ax.figure.transFigure)
            # below legend, add text "Comparison to brain health score: ***: p<0.001
            ax.text(0.8, -0.06, '***: p<0.001', horizontalalignment='center', verticalalignment='bottom',
                    transform=ax.figure.transFigure, fontsize=fontsize_label)
        if pd.isna(cohort_data.get(FEATURE_FRIENDLY["cog_total"], np.nan)):
            ax.text(0.59, 0.09, "N/A", transform=ax.transAxes, fontsize=fontsize_label, rotation=90, color="#303030")

    for j in range(len(cohorts), n_cols):
        axes[j].axis("off")

    fig.suptitle("Comparative hazard ratios", fontsize=fontsize_title, y=1.08, x=0.50, ha="center")
    ensure_dir(plot_savedir)
    plt.tight_layout(**TIGHT_KW)
    LAYOUT_KW_B = LAYOUT_KW.copy()
    LAYOUT_KW_B["top"] = LAYOUT_KW_B["top"] + 0.05
    fig.subplots_adjust(**LAYOUT_KW_B)
    fig.savefig(plot_savedir / "figure_brain_health_score_survival_curves_part_b.png", dpi=300, bbox_inches=None)
    fig.savefig(plot_savedir / "figure_brain_health_score_survival_curves_part_b.pdf", dpi=600, bbox_inches=None)


# part c plot, just a suptitle '    # fig.suptitle("Brain health score predicts mortality risk", y=1, fontsize=fontsize_title, x=0.50)'
# same dimensions as part a and b:

def part_c_suptitle(
    plot_savedir: os.PathLike | str,
    fontsize_title: int = 11,
) -> None:
    plot_savedir = Path(plot_savedir)
    fig, ax = plt.subplots(1, 1, figsize=(7, 0.35))
    ax.axis("off")
    fig.suptitle("Brain health score predicts mortality risk", y=0.90, fontsize=fontsize_title, x=0.50)
    ensure_dir(plot_savedir)
    plt.tight_layout(**TIGHT_KW)
    LAYOUT_KW_C = LAYOUT_KW.copy()
    LAYOUT_KW_C["bottom"] = 0.1
    LAYOUT_KW_C["top"] = 1.0
    fig.subplots_adjust(**LAYOUT_KW_C)
    fig.savefig(plot_savedir / "figure_brain_health_score_survival_curves_part_c.png", dpi=300, bbox_inches=None)
    fig.savefig(plot_savedir / "figure_brain_health_score_survival_curves_part_c.pdf", dpi=600, bbox_inches=None)

def plot_single_cohort_bars_mros(
    df_results_hr_matrix: pd.DataFrame,
    p_vals_hazard_comparisons: pd.DataFrame,
    plot_savedir: os.PathLike | str,
    fontsize_title: int = 11,
    fontsize_subtitle: int = 10,
    fontsize_label: int = 8,
) -> None:
    plot_savedir = Path(plot_savedir)
    cohort = "MrOS"
    if cohort not in df_results_hr_matrix.columns:
        return

    features = ["REM sleep %", "Spindle density", "Neuropsychological test score", "Brain health score"]
    bar_colors = ["#E0E0E0", "#A0A0A0", "#606060", "#303030"]
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    cohort_data = df_results_hr_matrix[cohort].rename(
        index={
            FEATURE_FRIENDLY["cog_total"]: "Neuropsychological test score",
            FEATURE_FRIENDLY["brain_health_score"]: "Brain health score",
        }
    )
    p_vals = p_vals_hazard_comparisons.rename(
        index={
            FEATURE_FRIENDLY["cog_total"]: "Neuropsychological test score",
            FEATURE_FRIENDLY["brain_health_score"]: "Brain health score",
        }
    )

    means, yerr_lower, yerr_upper = [], [], []
    for feat in features:
        hr_string = cohort_data.get(feat, np.nan)
        if pd.notna(hr_string):
            try:
                val = float(hr_string.split()[0])
                low = float(hr_string.split("(")[1].split("-")[0])
                high = float(hr_string.split("-")[1].split(")")[0])
            except Exception:
                val = low = high = np.nan
        else:
            val = low = high = np.nan
        if np.isnan(val):
            means.append(np.nan)
            yerr_lower.append(np.nan)
            yerr_upper.append(np.nan)
        else:
            means.append(val)
            yerr_lower.append(val - low)
            yerr_upper.append(high - val)

    x_positions = np.arange(len(features))
    bars = ax.bar(x_positions, means, yerr=[yerr_lower, yerr_upper], color=bar_colors, alpha=0.8, capsize=5)

    for j, feat in enumerate(features):
        if feat == "Brain health score":
            continue
        p_val = p_vals.loc[feat, cohort] if feat in p_vals.index else np.nan
        if pd.isna(p_val):
            continue
        if p_val < 0.001:
            p_txt = "***"
        elif p_val < 0.01:
            p_txt = "**"
        elif p_val < 0.05:
            p_txt = "*"
        elif p_val < 0.10:
            p_txt = f"p={p_val:.2f}"
        else:
            p_txt = np.nan
        if pd.notna(p_txt):
            brain_health_idx = features.index("Brain health score")
            y_base = 1.1 - 0.1 * j
            ax.plot([j, brain_health_idx], [y_base, y_base], color="#303030", lw=1)
            ax.text((j + brain_health_idx) / 2, y_base, p_txt, ha="center", va="bottom", fontsize=fontsize_label, color="#303030")

    ax.set_xticks([])
    ax.set_ylim(0, 1.2)
    ax.tick_params(axis="both", which="major", labelsize=fontsize_label, pad=1, length=2)
    ax.set_ylabel("Hazard ratio (per SD)", fontsize=fontsize_label, labelpad=1)
    ax.legend(
        bars,
        features,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=1,
        frameon=False,
        fontsize=fontsize_label,
        bbox_transform=ax.figure.transFigure,
    )
    ax.text(0.75, -0.08, "***: p<0.001", transform=ax.figure.transFigure, ha="center", va="bottom", fontsize=fontsize_label)
    fig.suptitle("Hazard ratios by feature\n(MrOS, age-adjusted)", fontsize=fontsize_title, y=0.97, x=0.50, ha="center")
    plt.tight_layout()
    ensure_dir(plot_savedir)
    fig.savefig(plot_savedir / "figure_brain_health_score_hazard_ratios_mros.png", dpi=300, bbox_inches="tight")


def combine_part_a_b_c_images(plot_savedir: os.PathLike | str) -> None:
    plot_savedir = Path(plot_savedir)
    path_a = plot_savedir / "figure_brain_health_score_survival_curves_part_a.png"
    path_b = plot_savedir / "figure_brain_health_score_survival_curves_part_b.png"
    path_c = plot_savedir / "figure_brain_health_score_survival_curves_part_c.png"
    if not (path_a.is_file() and path_b.is_file() and path_c.is_file()):
        print("[WARN] Cannot combine figures: missing part A, B, or C.")
        return
    img_a = Image.open(path_a)
    img_b = Image.open(path_b)
    img_c = Image.open(path_c)
    try:
        font_path = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
        font = ImageFont.truetype(font_path, 52)
    except Exception:
        font = ImageFont.load_default()
    ImageDraw.Draw(img_a).text((0, 0), "A", fill="black", font=font)
    ImageDraw.Draw(img_b).text((0, 0), "B", fill="black", font=font)
    
    combined_width = max(img_a.width, img_b.width, img_c.width)
    combined_height = img_a.height + img_b.height + img_c.height
    combined_img = Image.new("RGB", (combined_width, combined_height), color=(255, 255, 255))
    combined_img.paste(img_c, (0, 0))
    combined_img.paste(img_a, (0, img_c.height))
    combined_img.paste(img_b, (0, img_c.height + img_a.height))
    out_path = plot_savedir / "figure_brain_health_score_survival_curves.png"
    combined_img.save(out_path)
    print(f"Saved combined figure: {out_path}")


# -----------------------------
# Main
# -----------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    out_root = OUTPUT_ROOT
    ensure_dir(out_root)

    data_mros = res_mros = None
    data_fhs = res_fhs = None
    data_mgh = res_mgh = None
    data_bidmc = res_bidmc = None

    try:
        result = analyze_cohort(
            cohort_label="MrOS",
            csv_path=DATA_DIR / "data_mros_survival.csv",
            horizons=[5.0, 10.0, 15.0],
            risk_ages=[70, 80],
            risk_horizon=5.0,
        )
        if result is not None:
            data_mros, res_mros = result
    except Exception as exc:
        print(f"[WARN] MrOS analysis failed: {exc}")

    try:
        result = analyze_cohort(
            cohort_label="FHS",
            csv_path=DATA_DIR / "data_fhs_survival.csv",
            horizons=[5.0, 10.0],
            risk_ages=[65, 75],
            risk_horizon=5.0,
        )
        if result is not None:
            data_fhs, res_fhs = result
    except Exception as exc:
        print(f"[WARN] FHS analysis failed: {exc}")

    try:
        result = analyze_cohort(
            cohort_label="MGH",
            csv_path=DATA_DIR / "data_mgh_survival.csv",
            horizons=[5.0, 10.0],
            risk_ages=[65, 75],
            risk_horizon=5.0,
        )
        if result is not None:
            data_mgh, res_mgh = result
    except Exception as exc:
        print(f"[WARN] mgh analysis failed: {exc}")

    try:
        result = analyze_cohort(
            cohort_label="BIDMC",
            csv_path=DATA_DIR / "data_bidmc_survival.csv",
            horizons=[5.0, 10.0],
            risk_ages=[65, 75],
            risk_horizon=5.0,
        )
        if result is not None:
            data_bidmc, res_bidmc = result
    except Exception as exc:
        print(f"[WARN] BIDMC analysis failed: {exc}")

    df_results_hr, df_results_hr_matrix = build_results_long_and_matrix(res_mros, res_fhs, res_mgh, res_bidmc)
    try:
        df_results_hr.to_csv(out_root / "df_results_hr.csv", index=False)
        df_results_hr_matrix.to_csv(out_root / "df_results_hr_matrix.csv")
    except Exception as exc:
        print(f"[WARN] Failed to save results tables: {exc}")

    p_vals_hazard_comparisons = compute_pvals_vs_bhs(data_mros, data_fhs, data_mgh, data_bidmc)
    print("\nP-values for hazard ratio comparisons vs. brain health score:")
    print(p_vals_hazard_comparisons)
    try:
        p_vals_hazard_comparisons.to_csv(out_root / "p_vals_hazard_comparisons.csv")
    except Exception:
        pass

    datasets = {}
    if data_mros is not None:
        datasets["mros"] = data_mros
    if data_fhs is not None:
        datasets["shhs"] = data_fhs
    if data_mgh is not None:
        datasets["mgh"] = data_mgh
    if data_bidmc is not None:
        datasets["bidmc"] = data_bidmc

    if datasets:
        try:
            plot_survival_curves_part_a(datasets, out_root, n_bootstraps=N_BOOTSTRAPS_FOR_CURVE_CI)
        except Exception as exc:
            print(f"[WARN] Part A plotting failed: {exc}")

    if not df_results_hr_matrix.empty:
        try:
            plot_comparative_bars_part_b(df_results_hr_matrix, p_vals_hazard_comparisons, out_root)
        except Exception as exc:
            print(f"[WARN] Part B plotting failed: {exc}")

    try:
        if not df_results_hr_matrix.empty and "MrOS" in df_results_hr_matrix.columns:
            plot_single_cohort_bars_mros(df_results_hr_matrix, p_vals_hazard_comparisons, out_root)
    except Exception as exc:
        print(f"[WARN] MrOS single-cohort plotting failed: {exc}")

    try:
        part_c_suptitle(out_root)
    except Exception as exc:
        print(f"[WARN] Part C suptitle failed: {exc}")

    try:
        combine_part_a_b_c_images(out_root)
    except Exception as exc:
        print(f"[WARN] Combining images failed: {exc}")


if __name__ == "__main__":
    import time
    start_time = time.time()
    main(sys.argv[1:])
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nTotal runtime: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
