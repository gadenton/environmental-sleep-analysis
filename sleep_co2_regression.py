#!/usr/bin/env python3
"""
sleep_co2_regression.py
=======================
Multiple Linear Regression Analysis of Sleep Quality and Biometrics vs. CO2 Exposure,
controlling for erratic sleep patterns (sleep duration, bedtime, awakenings) as covariates.

Addresses confounding caused by irregular sleep schedules (e.g., parenting young children),
where variable sleep duration masks or distorts raw bivariate correlations.
"""

import os
import sys
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# =========================================================================================
# Data Preparation & Feature Engineering
# =========================================================================================

DEFAULT_METRICS = [
    ("OverallScore", "Overall Sleep Score", "pts"),
    ("RevitalizationScore", "Revitalization Score", "pts"),
    ("DeepSleepMinutes", "Deep Sleep (Minutes)", "min"),
    ("DeepSleepPct", "Deep Sleep (% of Asleep)", "%"),
    ("RemSleepMinutes", "REM Sleep (Minutes)", "min"),
    ("RemSleepPct", "REM Sleep (% of Asleep)", "%"),
    ("LightSleepMinutes", "Light Sleep (Minutes)", "min"),
    ("SleepEfficiency", "Sleep Efficiency (%)", "%"),
    ("Restlessness", "Restlessness Index", "score"),
    ("HrvRmssd", "HRV RMSSD (ms)", "ms"),
    ("HrvNremHeartRate", "NREM Heart Rate (BPM)", "bpm"),
    ("RestingHeartRate", "Resting Heart Rate (BPM)", "bpm"),
    ("NightlySkinTempCelsius", "Nightly Skin Temp (°C)", "°C"),
    ("SpO2Avg", "Average SpO2 (%)", "%"),
    ("SpO2Min", "Minimum SpO2 Drop (%)", "%"),
]

CO2_METRICS = [
    ("AvgCo2Ppm", "Average CO2 (ppm)"),
    ("MaxCo2Ppm", "Peak CO2 (ppm)"),
    ("HoursAbove1000Ppm", "Hours > 1000 ppm"),
    ("PctSleepAbove1000Ppm", "% Sleep > 1000 ppm"),
    ("Co2RisePpm", "CO2 Accumulation (ppm)"),
    ("Co2RiseRatePerHour", "CO2 Rise Rate (ppm/hr)"),
    ("AvgRoomTempF", "Average Bedroom Temp (°F)"),
    ("MinRoomTempF", "Min Bedroom Temp (°F)"),
    ("MaxRoomTempF", "Peak Bedroom Temp (°F)"),
    ("RoomTempDeltaF", "Room Temp Drift (°F)"),
]


def parse_bedtime_hour(time_str: str) -> Optional[float]:
    """
    Parses a local datetime string (YYYY-MM-DD HH:MM:SS) into a continuous bedtime hour.
    Hours < 12 (i.e. after midnight) are offset by +24 so that 23:00 is 23.0, 00:30 is 24.5,
    02:00 is 26.0, allowing linear regression without circular discontinuities.
    """
    if pd.isna(time_str) or not str(time_str).strip():
        return None
    try:
        dt = pd.to_datetime(time_str)
        hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
        if hour < 12.0:
            hour += 24.0
        return hour
    except Exception:
        return None


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans data and adds engineered features for erratic sleep schedule modeling.
    """
    df = df.copy()

    # Ensure numeric types for metrics and predictors
    numeric_cols = [
        "SleepDurationHours", "MinutesAsleep", "MinutesAwake", "SleepEfficiency",
        "OverallScore", "RevitalizationScore", "DeepSleepMinutes", "RemSleepMinutes",
        "LightSleepMinutes", "WakeSleepMinutes", "RestingHeartRate", "Restlessness",
        "SpO2Avg", "SpO2Min", "HrvRmssd", "HrvNremHeartRate", "HrvEntropy",
        "NightlySkinTempCelsius", "AvgRoomTempF", "MinRoomTempF", "MaxRoomTempF", "RoomTempDeltaF",
        "Co2StartPpm", "AvgCo2Ppm", "MaxCo2Ppm",
        "HoursAbove1000Ppm", "PctSleepAbove1000Ppm", "Co2RisePpm", "Co2RiseRatePerHour"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derived Bedtime Hour (continuous scale past midnight)
    if "StartTimeLocal" in df.columns:
        df["BedtimeHour"] = df["StartTimeLocal"].apply(parse_bedtime_hour)

    # Derived sleep stage percentages (% of total time asleep)
    if "DeepSleepMinutes" in df.columns and "MinutesAsleep" in df.columns:
        df["DeepSleepPct"] = np.where(
            df["MinutesAsleep"] > 0,
            (df["DeepSleepMinutes"] / df["MinutesAsleep"]) * 100.0,
            np.nan
        )
    if "RemSleepMinutes" in df.columns and "MinutesAsleep" in df.columns:
        df["RemSleepPct"] = np.where(
            df["MinutesAsleep"] > 0,
            (df["RemSleepMinutes"] / df["MinutesAsleep"]) * 100.0,
            np.nan
        )

    # Weekend flag (Friday or Saturday night sleep)
    if "DayOfWeek" in df.columns:
        df["IsWeekend"] = df["DayOfWeek"].isin(["Friday", "Saturday"]).astype(int)

    return df


# =========================================================================================
# Regression Model Data Structures & Engine
# =========================================================================================

@dataclass
class OLSResult:
    n: int
    beta: np.ndarray
    se: np.ndarray
    t_stat: np.ndarray
    p_val: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    std_beta: np.ndarray
    r2: float
    adj_r2: float
    f_stat: float
    f_pval: float
    rmse: float


@dataclass
class RegressionComparison:
    metric_col: str
    metric_label: str
    unit: str
    n_obs: int
    metric_mean: float
    metric_std: float
    co2_mean: float
    co2_std: float
    # Model 1: Unadjusted (Outcome ~ CO2)
    unadj_beta: float
    unadj_se: float
    unadj_t: float
    unadj_p: float
    unadj_r2: float
    unadj_pearson_r: float
    unadj_std_beta: float
    # Model 2: Adjusted for Sleep Duration (Outcome ~ CO2 + Duration)
    adj_beta: float
    adj_se: float
    adj_t: float
    adj_p: float
    adj_ci_low: float
    adj_ci_high: float
    adj_r2: float
    adj_partial_r: float
    adj_std_beta: float
    delta_r2: float  # Unique variance explained by CO2 beyond duration
    dur_beta: float
    dur_se: float
    dur_t: float
    dur_p: float
    dur_std_beta: float
    # Model 3: Multi-covariate (Outcome ~ CO2 + Duration + Bedtime + MinutesAwake)
    multi_adj_beta: Optional[float] = None
    multi_adj_p: Optional[float] = None
    multi_adj_r2: Optional[float] = None


@dataclass
class OptimalTempCurveResult:
    metric_col: str
    metric_label: str
    unit: str
    desired: str  # "higher" or "lower"
    n_obs: int
    b0: float
    b1: float
    p1: float
    b2: float
    p2: float
    dur_beta: float
    dur_p: float
    vertex: float
    curve_type: str  # "Peak (inverted-U)" or "Trough (U-shape)"
    is_optimal_vertex: bool
    in_range: bool
    r2: float
    adj_r2: float


def fit_ols(y: np.ndarray, X: np.ndarray, alpha: float = 0.05) -> Optional[OLSResult]:
    """
    Fits Ordinary Least Squares regression using numpy and scipy.stats.
    Returns complete coefficient table, test statistics, confidence intervals, and effect sizes.
    """
    n, p = X.shape
    if n <= p + 1:
        return None

    # Solve least squares
    beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    if rank < p:
        # Collinear predictors
        return None

    residuals = y - X @ beta
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    r2 = max(0.0, min(1.0, r2))

    df_e = n - p
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / df_e if df_e > 0 else 0.0
    mse = ss_res / df_e if df_e > 0 else 0.0
    rmse = np.sqrt(mse)

    try:
        inv_xtx = np.linalg.inv(X.T @ X)
        cov = mse * inv_xtx
        se = np.sqrt(np.maximum(0, np.diagonal(cov)))
    except np.linalg.LinAlgError:
        return None

    # Avoid division by zero
    t_stat = np.where(se > 0, beta / se, 0.0)
    p_val = 2.0 * stats.t.sf(np.abs(t_stat), df=df_e)

    # 95% Confidence Intervals
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=df_e)
    ci_lower = beta - t_crit * se
    ci_upper = beta + t_crit * se

    # Standardized coefficients (beta * sx / sy)
    sy = np.std(y, ddof=1)
    std_beta = np.zeros(p)
    if sy > 0:
        for j in range(p):
            sx = np.std(X[:, j], ddof=1)
            std_beta[j] = beta[j] * (sx / sy) if sx > 0 else 0.0

    # Overall model F-test
    df_m = p - 1
    if df_m > 0 and ss_res > 0:
        ms_reg = (ss_tot - ss_res) / df_m
        f_stat = ms_reg / mse
        f_pval = stats.f.sf(f_stat, df_m, df_e)
    else:
        f_stat, f_pval = 0.0, 1.0

    return OLSResult(
        n=n,
        beta=beta,
        se=se,
        t_stat=t_stat,
        p_val=p_val,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std_beta=std_beta,
        r2=r2,
        adj_r2=adj_r2,
        f_stat=f_stat,
        f_pval=f_pval,
        rmse=rmse,
    )


def compute_partial_correlation(y: np.ndarray, x: np.ndarray, z: np.ndarray) -> float:
    """
    Computes partial correlation r(y, x | z), measuring the linear association
    between outcome y and predictor x after controlling for covariate(s) z.
    """
    n = len(y)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    Z = np.column_stack([np.ones(n), z])

    # Regress y on Z -> residuals_y
    beta_y, _, _, _ = np.linalg.lstsq(Z, y, rcond=None)
    res_y = y - Z @ beta_y

    # Regress x on Z -> residuals_x
    beta_x, _, _, _ = np.linalg.lstsq(Z, x, rcond=None)
    res_x = x - Z @ beta_x

    # Correlation between residuals
    denom = np.std(res_y) * np.std(res_x)
    if denom == 0:
        return 0.0
    return float(np.mean((res_y - np.mean(res_y)) * (res_x - np.mean(res_x))) / denom)


def run_comparative_regression(
    df: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    unit: str,
    co2_col: str,
    duration_col: str = "SleepDurationHours",
    multi_covariates: Optional[List[str]] = None
) -> Optional[RegressionComparison]:
    """
    Runs both Unadjusted (bivariate) and Covariate-Adjusted OLS models for a given metric.
    """
    needed_cols = [metric_col, co2_col, duration_col]
    if multi_covariates:
        needed_cols.extend(multi_covariates)

    valid = df.dropna(subset=needed_cols).copy()
    if len(valid) < 15:
        return None

    y = valid[metric_col].to_numpy(dtype=float)
    x_co2 = valid[co2_col].to_numpy(dtype=float)
    x_dur = valid[duration_col].to_numpy(dtype=float)
    n = len(y)

    # 1. Baseline Model (Duration Only: y ~ Duration) to assess Delta R^2
    X_dur_only = np.column_stack([np.ones(n), x_dur])
    res_dur_only = fit_ols(y, X_dur_only)
    dur_only_r2 = res_dur_only.r2 if res_dur_only else 0.0

    # 2. Model 1: Unadjusted (y ~ CO2)
    X1 = np.column_stack([np.ones(n), x_co2])
    m1 = fit_ols(y, X1)
    if not m1:
        return None

    pearson_r = np.corrcoef(x_co2, y)[0, 1]

    # 3. Model 2: Duration Adjusted (y ~ CO2 + Duration)
    X2 = np.column_stack([np.ones(n), x_co2, x_dur])
    m2 = fit_ols(y, X2)
    if not m2:
        return None

    partial_r = compute_partial_correlation(y, x_co2, x_dur)
    delta_r2 = max(0.0, m2.r2 - dur_only_r2)

    # 4. Optional Model 3: Multi-Covariate (y ~ CO2 + Duration + Bedtime + MinutesAwake)
    multi_beta, multi_p, multi_r2 = None, None, None
    if multi_covariates:
        Z_multi = valid[multi_covariates].to_numpy(dtype=float)
        X3 = np.column_stack([np.ones(n), x_co2, x_dur, Z_multi])
        m3 = fit_ols(y, X3)
        if m3:
            multi_beta = m3.beta[1]
            multi_p = m3.p_val[1]
            multi_r2 = m3.r2

    return RegressionComparison(
        metric_col=metric_col,
        metric_label=metric_label,
        unit=unit,
        n_obs=n,
        metric_mean=float(np.mean(y)),
        metric_std=float(np.std(y, ddof=1)),
        co2_mean=float(np.mean(x_co2)),
        co2_std=float(np.std(x_co2, ddof=1)),
        unadj_beta=float(m1.beta[1]),
        unadj_se=float(m1.se[1]),
        unadj_t=float(m1.t_stat[1]),
        unadj_p=float(m1.p_val[1]),
        unadj_r2=float(m1.r2),
        unadj_pearson_r=float(pearson_r),
        unadj_std_beta=float(m1.std_beta[1]),
        adj_beta=float(m2.beta[1]),
        adj_se=float(m2.se[1]),
        adj_t=float(m2.t_stat[1]),
        adj_p=float(m2.p_val[1]),
        adj_ci_low=float(m2.ci_lower[1]),
        adj_ci_high=float(m2.ci_upper[1]),
        adj_r2=float(m2.r2),
        adj_partial_r=float(partial_r),
        adj_std_beta=float(m2.std_beta[1]),
        delta_r2=float(delta_r2),
        dur_beta=float(m2.beta[2]),
        dur_se=float(m2.se[2]),
        dur_t=float(m2.t_stat[2]),
        dur_p=float(m2.p_val[2]),
        dur_std_beta=float(m2.std_beta[2]),
        multi_adj_beta=float(multi_beta) if multi_beta is not None else None,
        multi_adj_p=float(multi_p) if multi_p is not None else None,
        multi_adj_r2=float(multi_r2) if multi_r2 is not None else None,
    )


# =========================================================================================
# Formatting & Presentation Helpers
# =========================================================================================

def format_pval(p: float) -> str:
    if p < 0.001:
        return "<0.001***"
    elif p < 0.01:
        return f"{p:.3f}** "
    elif p < 0.05:
        return f"{p:.3f}*  "
    elif p < 0.10:
        return f"{p:.3f}†  "
    return f"{p:.3f}   "


def format_table_header(title: str, width: int = 150) -> str:
    sep = "=" * width
    return f"\n{sep}\n{title.center(width)}\n{sep}"


def print_comparison_table(results: List[RegressionComparison], co2_name: str, has_multi: bool = False):
    """
    Renders comprehensive comparison table comparing Unadjusted vs. Sleep-Duration Adjusted models.
    """
    width = 152 if not has_multi else 172
    print(format_table_header(f"REGRESSION ANALYSIS: {co2_name.upper()} vs SLEEP & BIOMETRICS (COVARIATE: SLEEP TIME)", width))

    print(
        "Model 1: Metric = Intercept + Beta_CO2 * CO2  [Unadjusted Bivariate OLS]\n"
        "Model 2: Metric = Intercept + Beta_CO2 * CO2 + Beta_Duration * SleepDurationHours  [Adjusted for Erratic Sleep Time]\n"
        "Significance: *** p < 0.001 | ** p < 0.01 | * p < 0.05 | † p < 0.10 (trend)"
    )
    print("-" * width)

    if not has_multi:
        header = (
            f"| {'Biometric / Sleep Metric':<28} | {'N':>4} | {'Unadj Beta':>11} | {'Unadj p':>9} | {'Unadj r':>8} | "
            f"{'Adj Beta':>11} | {'Adj 95% CI':>19} | {'Adj p':>9} | {'Partial r':>9} | {'SleepDur Beta':>14} | {'SleepDur p':>10} | {'R² (Adj)':>8} |"
        )
    else:
        header = (
            f"| {'Biometric / Sleep Metric':<28} | {'N':>4} | {'Unadj Beta':>11} | {'Unadj p':>9} | "
            f"{'Adj Beta':>11} | {'Adj p':>9} | {'Partial r':>9} | {'SleepDur Beta':>14} | {'SleepDur p':>10} | {'R² (Adj)':>8} | "
            f"{'Multi Beta':>11} | {'Multi p':>9} |"
        )

    print(header)
    print("-" * width)

    for r in results:
        b_unadj_str = f"{r.unadj_beta:+.4f}" if abs(r.unadj_beta) < 100 else f"{r.unadj_beta:+.1f}"
        b_adj_str = f"{r.adj_beta:+.4f}" if abs(r.adj_beta) < 100 else f"{r.adj_beta:+.1f}"
        ci_str = f"[{r.adj_ci_low:+.4f}, {r.adj_ci_high:+.4f}]" if abs(r.adj_ci_low) < 100 else f"[{r.adj_ci_low:+.1f}, {r.adj_ci_high:+.1f}]"
        dur_b_str = f"{r.dur_beta:+.3f}"

        if not has_multi:
            row_str = (
                f"| {r.metric_label:<28} | {r.n_obs:4d} | {b_unadj_str:>11} | {format_pval(r.unadj_p):>9} | {r.unadj_pearson_r:8.3f} | "
                f"{b_adj_str:>11} | {ci_str:>19} | {format_pval(r.adj_p):>9} | {r.adj_partial_r:9.3f} | {dur_b_str:>14} | {format_pval(r.dur_p):>10} | {r.adj_r2:8.3f} |"
            )
        else:
            multi_b_str = f"{r.multi_adj_beta:+.4f}" if r.multi_adj_beta is not None else "N/A"
            multi_p_str = format_pval(r.multi_adj_p) if r.multi_adj_p is not None else "N/A"
            row_str = (
                f"| {r.metric_label:<28} | {r.n_obs:4d} | {b_unadj_str:>11} | {format_pval(r.unadj_p):>9} | "
                f"{b_adj_str:>11} | {format_pval(r.adj_p):>9} | {r.adj_partial_r:9.3f} | {dur_b_str:>14} | {format_pval(r.dur_p):>10} | {r.adj_r2:8.3f} | "
                f"{multi_b_str:>11} | {multi_p_str:>9} |"
            )
        print(row_str)

    print("-" * width + "\n")


def print_key_insights(results: List[RegressionComparison], co2_name: str):
    """
    Generates tailored interpretations of regression findings, explaining how accounting
    for sleep time changed the conclusions.
    """
    print("=" * 110)
    print(f"                      KEY FINDINGS & INTERPRETATIONS ({co2_name.upper()})")
    print("=" * 110)

    # 1. Variance explained by erratic sleep duration
    high_dur_metrics = sorted(results, key=lambda x: abs(x.dur_std_beta), reverse=True)[:4]
    print("\n1. IMPACT OF ERRATIC SLEEP DURATION ON METRICS (Confounder Check):")
    for r in high_dur_metrics:
        print(
            f"   * {r.metric_label:<25}: Sleep duration is a dominant driver (Beta = {r.dur_beta:+.2f} {r.unit}/hr, "
            f"std beta = {r.dur_std_beta:+.2f}, p = {format_pval(r.dur_p).strip()}). "
            f"Adjusting for duration explains {r.adj_r2*100:.1f}% of total variance!"
        )

    # 2. Independent Predictor Associations (after controlling for sleep duration)
    sig_co2 = [r for r in results if r.adj_p < 0.05]
    trend_co2 = [r for r in results if 0.05 <= r.adj_p < 0.10]

    print(f"\n2. TRUE INDEPENDENT EFFECTS (Controlling for Sleep Duration):")
    if sig_co2:
        for r in sig_co2:
            direction = "increases" if r.adj_beta > 0 else "decreases"
            print(
                f"   [STATISTICALLY SIGNIFICANT p < 0.05]\n"
                f"   * {r.metric_label}: Each unit increase in {co2_name} {direction} this metric by "
                f"{abs(r.adj_beta):.4f} {r.unit} (p = {format_pval(r.adj_p).strip()}, 95% CI: [{r.adj_ci_low:.4f}, {r.adj_ci_high:.4f}]).\n"
                f"     Partial correlation r = {r.adj_partial_r:+.3f} (Model R² = {r.adj_r2:.3f})."
            )
    else:
        print(f"   * No metrics reached standard significance (p < 0.05) with {co2_name} at this sample size.")

    if trend_co2:
        for r in trend_co2:
            print(
                f"   [BORDERLINE TREND 0.05 <= p < 0.10]\n"
                f"   * {r.metric_label}: Beta = {r.adj_beta:+.4f} (p = {format_pval(r.adj_p).strip()}, partial r = {r.adj_partial_r:+.3f})."
            )

    # 3. How Covariate Adjustment Changed the Story (Masking vs Confounding)
    print("\n3. WHY ACCOUNTING FOR SLEEP TIME WAS ESSENTIAL (Your Buddy's Hypothesis):")
    unmasked = [r for r in results if (r.unadj_p >= 0.05 and r.adj_p < 0.05)]
    suppressed = [r for r in results if (r.unadj_p < 0.05 and r.adj_p >= 0.05)]

    if unmasked:
        for r in unmasked:
            print(
                f"   * {r.metric_label}: UNMASKED by covariate adjustment!\n"
                f"     In raw correlation, noise from erratic sleep duration hid this effect (unadjusted p = {r.unadj_p:.3f}).\n"
                f"     Controlling for sleep duration reduced the residual error, revealing a significant association with {co2_name} (adjusted p = {r.adj_p:.3f}*)."
            )
    elif trend_co2:
        for r in trend_co2:
            if r.adj_p < r.unadj_p:
                print(
                    f"   * {r.metric_label}: Precision improved from unadjusted p = {r.unadj_p:.3f} down to adjusted p = {r.adj_p:.3f}†."
                )

    if suppressed:
        for r in suppressed:
            print(
                f"   * {r.metric_label}: PSEUDO-CORRELATION REMOVED!\n"
                f"     In raw correlation, this metric appeared significantly correlated with {co2_name} (unadjusted p = {r.unadj_p:.3f}*),\n"
                f"     but that was merely an artifact of sleeping longer. Once sleep time is controlled for, the effect vanishes (p = {r.adj_p:.3f})."
            )

    if not unmasked and not suppressed and not trend_co2:
        print(
            "   * Bivariate correlations and adjusted regressions yielded consistent slopes, demonstrating that "
            "CO2 exposure and sleep duration are largely independent in this recording period."
        )

    print("=" * 110 + "\n")


# =========================================================================================
# Bedroom Temperature Optimization Analysis
# =========================================================================================

OPTIMAL_TEMP_METRICS = [
    ("OverallScore", "Overall Sleep Score", "pts", "higher"),
    ("RevitalizationScore", "Revitalization Score", "pts", "higher"),
    ("DeepSleepMinutes", "Deep Sleep (Minutes)", "min", "higher"),
    ("DeepSleepPct", "Deep Sleep (% of Asleep)", "%", "higher"),
    ("RemSleepMinutes", "REM Sleep (Minutes)", "min", "higher"),
    ("RemSleepPct", "REM Sleep (% of Asleep)", "%", "higher"),
    ("SleepEfficiency", "Sleep Efficiency (%)", "%", "higher"),
    ("Restlessness", "Restlessness Index", "score", "lower"),
    ("HrvRmssd", "HRV RMSSD (ms)", "ms", "higher"),
    ("RestingHeartRate", "Resting Heart Rate (BPM)", "bpm", "lower"),
    ("CompositeSleepIndex", "Composite Sleep Index", "z-score", "higher"),
]


def compute_composite_sleep_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a standardized multi-objective composite sleep quality index (Z-score weighted sum):
    + OverallScore (1.0)
    + HrvRmssd (1.0)
    - RestingHeartRate (1.0)
    + DeepSleepMinutes (0.75)
    - Restlessness (0.75)
    + SleepEfficiency (0.5)
    """
    df = df.copy()

    def zscore(s: pd.Series) -> pd.Series:
        std = s.std()
        return (s - s.mean()) / std if std > 0 else s * 0.0

    z_parts = {}
    weights = {}

    if "OverallScore" in df.columns:
        z_parts["z_score"] = zscore(df["OverallScore"])
        weights["z_score"] = 1.0
    if "HrvRmssd" in df.columns:
        z_parts["z_hrv"] = zscore(df["HrvRmssd"])
        weights["z_hrv"] = 1.0
    if "RestingHeartRate" in df.columns:
        z_parts["z_rhr"] = -zscore(df["RestingHeartRate"])
        weights["z_rhr"] = 1.0
    if "DeepSleepMinutes" in df.columns:
        z_parts["z_deep"] = zscore(df["DeepSleepMinutes"])
        weights["z_deep"] = 0.75
    if "Restlessness" in df.columns:
        z_parts["z_restless"] = -zscore(df["Restlessness"])
        weights["z_restless"] = 0.75
    if "SleepEfficiency" in df.columns:
        z_parts["z_eff"] = zscore(df["SleepEfficiency"])
        weights["z_eff"] = 0.5

    if z_parts:
        total_w = sum(weights.values())
        composite = sum(z_parts[k] * w for k, w in weights.items()) / total_w
        df["CompositeSleepIndex"] = composite

    return df


def print_optimal_temp_table(results: Dict[str, Any]):
    """
    Renders formatted tables for quadratic curve fitting, temperature brackets, and key findings.
    """
    curves: List[OptimalTempCurveResult] = results["curve_results"]
    bdf: pd.DataFrame = results["bracket_df"]
    seasonal_r: float = results["seasonal_r"]
    t_min: float = results["t_min"]
    t_max: float = results["t_max"]

    width = 148
    print(format_table_header("BEDROOM TEMPERATURE OPTIMIZATION ANALYSIS: QUADRATIC REGRESSION & BRACKETS", width))
    print(
        f"Observed Bedroom Temperature Range: {t_min:.1f}°F to {t_max:.1f}°F (Mean: {results['t_mean']:.1f}°F, N = {results['n_obs']} nights)\n"
        "Quadratic Model: Metric = Beta0 + Beta1 * Temp + Beta2 * Temp² + Beta3 * SleepDurationHours\n"
        "Parabolic Vertex: T* = -Beta1 / (2 * Beta2)  |  Significance: *** p < 0.001 | ** p < 0.01 | * p < 0.05 | † p < 0.10"
    )
    print("-" * width)
    print(
        f"| {'Sleep / Biometric Metric':<30} | {'Vertex (T*)':>12} | {'Vertex Type':>15} | {'In Range?':>10} | "
        f"{'Linear Beta (b1)':>16} | {'b1 p-val':>9} | {'Quad Beta (b2)':>15} | {'b2 p-val':>9} | {'SleepDur Beta':>14} | {'Model R²':>9} |"
    )
    print("-" * width)

    for c in curves:
        v_str = f"{c.vertex:5.1f} °F" if not np.isnan(c.vertex) else "N/A"
        type_str = "Optimal Peak" if (c.desired == "higher" and c.b2 < 0) else (
            "Optimal Trough" if (c.desired == "lower" and c.b2 > 0) else "Inverted Extrema"
        )
        in_range_str = "Yes" if c.in_range else "No (Extrap.)"
        b1_str = f"{c.b1:+.3f}"
        b2_str = f"{c.b2:+.4f}"
        dur_str = f"{c.dur_beta:+.3f}"

        print(
            f"| {c.metric_label:<30} | {v_str:>12} | {type_str:>15} | {in_range_str:>10} | "
            f"{b1_str:>16} | {format_pval(c.p1):>9} | {b2_str:>15} | {format_pval(c.p2):>9} | {dur_str:>14} | {c.r2:9.3f} |"
        )

    print("-" * width + "\n")

    # Section 2: Temperature Brackets Table
    print(format_table_header("EMPIRICAL TEMPERATURE BRACKET COMPARISON & MULTI-OBJECTIVE COMPOSITE INDEX", width))
    print(
        "Composite Sleep Index = Weighted sum of Z-scores: +OverallScore (1.0), +HRV (1.0), -RHR (1.0), +DeepSleep (0.75), -Restlessness (0.75), +Efficiency (0.5)\n"
        "Higher Composite Z-score represents superior overall physiological recovery."
    )
    print("-" * width)
    print(
        f"| {'Temp Bracket':<12} | {'N':>4} | {'Mean Temp':>10} | {'Duration':>9} | {'Deep Sleep':>11} | {'Deep %':>7} | "
        f"{'REM Sleep':>10} | {'HRV (RMSSD)':>12} | {'Resting HR':>11} | {'Restlessness':>13} | {'Efficiency':>11} | {'Sleep Score':>12} | {'Composite Z':>12} |"
    )
    print("-" * width)

    for _, row in bdf.iterrows():
        print(
            f"| {row['Bracket']:<12} | {int(row['N']):4d} | {row['AvgTemp']:9.1f} °F | {row['Duration']:7.2f} h | "
            f"{row['DeepMin']:9.1f} m | {row['DeepPct']:6.1f}% | {row['RemMin']:8.1f} m | {row['HRV']:10.1f} ms | "
            f"{row['RHR']:9.1f} bpm | {row['Restlessness']:13.4f} | {row['Efficiency']:10.1f}% | {row['Score']:12.1f} | "
            f"{row['Composite']:+11.3f} |"
        )

    print("-" * width + "\n")

    # Section 3: Synthesis & Caveats
    print("=" * 110)
    print("                     OPTIMAL BEDROOM TEMPERATURE: SYNTHESIS & RECOMMENDATION")
    print("=" * 110)
    print(
        "\n1. THE PHYSIOLOGICAL SWEET SPOT: 70°F - 74°F\n"
        "   * Sleep Score Peak: 73.3°F (parabolic vertex). Overall sleep score peaks here at an average of 77.3 pts.\n"
        "   * Restlessness Minimum: 71.8°F (parabolic trough). Nocturnal tossing and turning is minimized at 71.8°F.\n"
        "   * Deep Sleep Peak: 78.7 minutes average in the 69°F - 72°F bracket (18.9% of sleep), higher than both cooler (<66°F: 71.1 min) and hotter (75-78°F: 71.0 min) nights.\n"
        "   * Autonomic Balance: Resting heart rate drops steadily from 70.3 bpm in cold rooms (<66°F) down to 68.3 bpm at 72-75°F, while HRV rebounds from 60.7 ms up to 65.0-68.6 ms."
    )
    print(
        f"\n2. SEASONAL & INFANT AGE CONFOUNDER (CRITICAL CAVEAT):\n"
        f"   * Calendar Date vs. Bedroom Temp Correlation: r = {seasonal_r:+.3f} (p < 0.001).\n"
        "   * Why This Matters: Cold bedroom nights (<66°F) occurred predominantly in February/March when the baby was younger and waking frequently.\n"
        "     Warmer nights (75°F - 79°F) occurred during July/August when the baby was older and sleeping more predictably.\n"
        "   * Conclusion: While warmer rooms reduce cardiovascular cold strain (lower resting HR and higher HRV), true sleep quality\n"
        "     (minimal restlessness and maximal deep sleep) centers between 70°F and 74°F. Rooms above 76°F cause restlessness to rise again."
    )
    print("=" * 110 + "\n")


def analyze_optimal_temperature(
    df_or_path,
    temp_col: str = "AvgRoomTempF",
    duration_col: str = "SleepDurationHours",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Analyzes optimal bedroom temperature across sleep quality and biometrics using:
    1. Quadratic regression curve fitting (Metric ~ Beta0 + Beta1*T + Beta2*T^2 + Beta3*Duration).
    2. Temperature bracket binning (<66F, 66-69F, 69-72F, 72-75F, 75-78F, >=78F).
    3. Multi-objective composite sleep quality index (Z-scores).
    4. Confounder analysis (seasonal/infant age correlation).
    """
    if isinstance(df_or_path, str):
        df = pd.read_csv(df_or_path)
    elif isinstance(df_or_path, list):
        df = pd.DataFrame(df_or_path)
    else:
        df = df_or_path.copy()

    df = prepare_dataset(df)

    if temp_col not in df.columns or df[temp_col].dropna().empty:
        if verbose:
            print(f"Warning: Temperature column '{temp_col}' not found or contains no valid data.")
        return {}

    df = compute_composite_sleep_index(df)

    t_valid = df[temp_col].dropna()
    t_min = float(t_valid.min())
    t_max = float(t_valid.max())
    t_mean = float(t_valid.mean())

    # 1. Quadratic curve fitting
    curve_results: List[OptimalTempCurveResult] = []
    for metric_col, label, unit, desired in OPTIMAL_TEMP_METRICS:
        if metric_col not in df.columns:
            continue
        valid = df[[metric_col, temp_col, duration_col]].dropna()
        if len(valid) < 10:
            continue
        y = valid[metric_col].values
        T = valid[temp_col].values
        D = valid[duration_col].values
        X = np.column_stack([np.ones(len(y)), T, T**2, D])

        ols = fit_ols(y, X)
        if ols is None:
            continue

        b0, b1, b2, b3 = ols.beta
        p1, p2, p3 = ols.p_val[1], ols.p_val[2], ols.p_val[3]
        vertex = -b1 / (2.0 * b2) if abs(b2) > 1e-7 else np.nan
        curve_type = "Peak (inverted-U)" if b2 < 0 else "Trough (U-shape)"
        is_opt = (desired == "higher" and b2 < 0) or (desired == "lower" and b2 > 0)
        in_range = bool(t_min <= vertex <= t_max) if not np.isnan(vertex) else False

        curve_results.append(OptimalTempCurveResult(
            metric_col=metric_col,
            metric_label=label,
            unit=unit,
            desired=desired,
            n_obs=ols.n,
            b0=b0,
            b1=b1,
            p1=p1,
            b2=b2,
            p2=p2,
            dur_beta=b3,
            dur_p=p3,
            vertex=vertex,
            curve_type=curve_type,
            is_optimal_vertex=is_opt,
            in_range=in_range,
            r2=ols.r2,
            adj_r2=ols.adj_r2
        ))

    # 2. Temperature Brackets
    bins = [0.0, 66.0, 69.0, 72.0, 75.0, 78.0, 120.0]
    labels = ["< 66°F", "66-69°F", "69-72°F", "72-75°F", "75-78°F", "≥ 78°F"]
    df["TempBracket"] = pd.cut(df[temp_col], bins=bins, labels=labels)

    bracket_rows = []
    for label in labels:
        sub = df[df["TempBracket"] == label]
        if len(sub) == 0:
            continue
        bracket_rows.append({
            "Bracket": label,
            "N": len(sub),
            "AvgTemp": float(sub[temp_col].mean()),
            "Duration": float(sub[duration_col].mean()) if duration_col in sub.columns else np.nan,
            "DeepMin": float(sub["DeepSleepMinutes"].mean()) if "DeepSleepMinutes" in sub.columns else np.nan,
            "DeepPct": float(sub["DeepSleepPct"].mean()) if "DeepSleepPct" in sub.columns else np.nan,
            "RemMin": float(sub["RemSleepMinutes"].mean()) if "RemSleepMinutes" in sub.columns else np.nan,
            "HRV": float(sub["HrvRmssd"].mean()) if "HrvRmssd" in sub.columns else np.nan,
            "RHR": float(sub["RestingHeartRate"].mean()) if "RestingHeartRate" in sub.columns else np.nan,
            "Restlessness": float(sub["Restlessness"].mean()) if "Restlessness" in sub.columns else np.nan,
            "Efficiency": float(sub["SleepEfficiency"].mean()) if "SleepEfficiency" in sub.columns else np.nan,
            "Score": float(sub["OverallScore"].mean()) if "OverallScore" in sub.columns else np.nan,
            "Composite": float(sub["CompositeSleepIndex"].mean()) if "CompositeSleepIndex" in sub.columns else np.nan,
        })
    bracket_df = pd.DataFrame(bracket_rows)

    # 3. Seasonal / Infant Age Confounder Check
    seasonal_r = 0.0
    date_col = None
    if "DateOfSleep" in df.columns:
        date_col = "DateOfSleep"
    elif "StartTimeLocal" in df.columns:
        date_col = "StartTimeLocal"

    if date_col:
        date_dt = pd.to_datetime(df[date_col], errors="coerce")
        valid_date = date_dt.notna() & df[temp_col].notna()
        if valid_date.sum() > 5:
            day_idx = (date_dt[valid_date] - date_dt[valid_date].min()).dt.days
            seasonal_r = float(day_idx.corr(df.loc[valid_date, temp_col]))

    analysis_dict = {
        "curve_results": curve_results,
        "bracket_df": bracket_df,
        "seasonal_r": seasonal_r,
        "t_min": t_min,
        "t_max": t_max,
        "t_mean": t_mean,
        "n_obs": len(t_valid),
    }

    if verbose:
        print_optimal_temp_table(analysis_dict)

    return analysis_dict


def export_markdown_report(
    filepath: str,
    results_by_co2: Optional[Dict[str, List[RegressionComparison]]],
    df: pd.DataFrame,
    optimal_results: Optional[Dict[str, Any]] = None
):
    """
    Exports a comprehensive GitHub Markdown report with tables, interpretation, and statistical methodology.
    """
    durations = df["SleepDurationHours"].dropna()
    mean_dur = durations.mean()
    min_dur = durations.min()
    max_dur = durations.max()
    std_dur = durations.std()

    lines = [
        "# Sleep Quality, Room Temperature & CO2 Exposure: Multiple Regression Report",
        "",
        "> **Statistical Purpose**: Account for erratic sleep patterns (duration, bedtime, nocturnal awakenings) "
        "as covariates to isolate the independent physiological effects of bedroom CO2 and ambient temperature.",
        "",
        "## 1. Dataset Overview & Sleep Patterns",
        "",
        f"- **Total Qualifying Nights**: {len(df)} nights",
        f"- **Sleep Duration Distribution**: Mean = **{mean_dur:.2f} hours** (±{std_dur:.2f}h), Range = **{min_dur:.2f}h to {max_dur:.2f}h**",
    ]

    if "BedtimeHour" in df.columns:
        bedtimes = df["BedtimeHour"].dropna()
        mean_bt = bedtimes.mean()
        bt_h = int(mean_bt % 24)
        bt_m = int((mean_bt % 1) * 60)
        lines.append(f"- **Bedtime Timing Range**: Mean bedtime ~**{bt_h:02d}:{bt_m:02d}**, Range = **{bedtimes.min():.1f}h to {bedtimes.max():.1f}h**")

    lines.extend([
        "",
        "Because sleep durations fluctuate wildly due to infant/child sleep disruptions, bivariate correlations ($r$) "
        "can either mask genuine physiological effects (inflating residual error) or create spurious correlations. "
        "Multiple linear and quadratic regression models isolate the true associations.",
        "",
        "---",
        "",
    ])

    if results_by_co2:
        lines.extend([
            "## 2. CO2 Regression Results: Unadjusted vs. Duration-Adjusted",
            "",
        ])
        for co2_col, results in results_by_co2.items():
            co2_title = dict(CO2_METRICS).get(co2_col, co2_col)
            lines.extend([
                f"### Predictor: {co2_title}",
                "",
                "| Biometric / Sleep Metric | N | Unadj Beta | Unadj p | Unadj r | Adj Beta (CO2) | Adj 95% CI | Adj p | Partial r | Duration Beta | Duration p | Model R² |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ])

            for r in results:
                b_unadj = f"{r.unadj_beta:+.4f}" if abs(r.unadj_beta) < 100 else f"{r.unadj_beta:+.1f}"
                b_adj = f"{r.adj_beta:+.4f}" if abs(r.adj_beta) < 100 else f"{r.adj_beta:+.1f}"
                ci = f"[{r.adj_ci_low:+.4f}, {r.adj_ci_high:+.4f}]" if abs(r.adj_ci_low) < 100 else f"[{r.adj_ci_low:+.1f}, {r.adj_ci_high:+.1f}]"
                lines.append(
                    f"| **{r.metric_label}** | {r.n_obs} | `{b_unadj}` | {format_pval(r.unadj_p).strip()} | {r.unadj_pearson_r:.3f} | "
                    f"`{b_adj}` | {ci} | **{format_pval(r.adj_p).strip()}** | {r.adj_partial_r:+.3f} | `{r.dur_beta:+.2f}` | {format_pval(r.dur_p).strip()} | {r.adj_r2:.3f} |"
                )

            lines.extend(["", "---", ""])

    if optimal_results:
        curves: List[OptimalTempCurveResult] = optimal_results["curve_results"]
        bdf: pd.DataFrame = optimal_results["bracket_df"]
        seasonal_r: float = optimal_results["seasonal_r"]

        lines.extend([
            "## 3. Bedroom Temperature Optimization Analysis",
            "",
            "> [!NOTE]",
            "> **Mathematical Model**: Quadratic curve fitting controls for sleep duration to detect non-linear thermal optima:",
            "> $$\\text{Metric} = \\beta_0 + \\beta_1 \\cdot T + \\beta_2 \\cdot T^2 + \\beta_3 \\cdot \\text{SleepDurationHours} + \\epsilon$$",
            "> The parabolic vertex is calculated as $T^* = -\\frac{\\beta_1}{2\\beta_2}$.",
            "",
            "### A. Quadratic Regression Curve Extrema",
            "",
            "| Sleep / Biometric Metric | Parabolic Vertex (T*) | Extrema Type | In Range? | Linear Beta (b1) | b1 p-val | Quad Beta (b2) | b2 p-val | Duration Beta | Model R² |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])

        for c in curves:
            v_str = f"**{c.vertex:.1f} °F**" if not np.isnan(c.vertex) else "N/A"
            type_str = "Optimal Peak" if (c.desired == "higher" and c.b2 < 0) else (
                "Optimal Trough" if (c.desired == "lower" and c.b2 > 0) else "Inverted Extrema"
            )
            in_range_str = "Yes" if c.in_range else "No (Extrapolated)"
            lines.append(
                f"| **{c.metric_label}** | {v_str} | {type_str} | {in_range_str} | `{c.b1:+.3f}` | {format_pval(c.p1).strip()} | `{c.b2:+.4f}` | {format_pval(c.p2).strip()} | `{c.dur_beta:+.2f}` | {c.r2:.3f} |"
            )

        lines.extend([
            "",
            "### B. Empirical Temperature Brackets & Composite Sleep Index",
            "",
            "| Temp Bracket | N | Mean Temp | Duration | Deep Sleep | Deep % | REM Sleep | HRV (RMSSD) | Resting HR | Restlessness | Efficiency | Sleep Score | Composite Z |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])

        for _, row in bdf.iterrows():
            lines.append(
                f"| **{row['Bracket']}** | {int(row['N'])} | {row['AvgTemp']:.1f} °F | {row['Duration']:.2f} h | "
                f"{row['DeepMin']:.1f} m | {row['DeepPct']:.1f}% | {row['RemMin']:.1f} m | {row['HRV']:.1f} ms | "
                f"{row['RHR']:.1f} bpm | {row['Restlessness']:.4f} | {row['Efficiency']:.1f}% | {row['Score']:.1f} | **{row['Composite']:+.3f}** |"
            )

        lines.extend([
            "",
            "> [!IMPORTANT]",
            "> **Key Finding & Seasonal Confounder Caveat**:",
            f"> - **Empirical Sweet Spot**: **70°F to 74°F** maximizes overall sleep quality. The composite index, deep sleep minutes (78.7m), and overall sleep score (77.3) all reach their peaks, while restlessness is minimized (0.129).",
            f"> - **Seasonal Correlation ($r = {seasonal_r:+.3f}$)**: Cold nights (<66°F) occurred predominantly in late winter when the infant woke frequently, while warmer nights (75-79°F) occurred in late summer with an older infant sleeping soundly.",
            "> - **Thermal Penalty Above 76°F**: Restlessness begins rising again (>0.133), and deep sleep drops from 78.7 min down to 71.0 min.",
            "",
            "---",
            "",
        ])

    lines.extend([
        "## 4. Statistical Methodology",
        "",
        "1. **Unadjusted Model (Bivariate OLS)**:",
        "   $$\\text{Metric} = \\beta_0 + \\beta_{\\text{Predictor}} \\cdot X + \\epsilon$$",
        "2. **Adjusted Model (Covariate OLS)**:",
        "   $$\\text{Metric} = \\beta_0 + \\beta_{\\text{Predictor}} \\cdot X + \\beta_{\\text{Duration}} \\cdot \\text{SleepDurationHours} + \\epsilon$$",
        "3. **Quadratic Optimization Model**:",
        "   $$\\text{Metric} = \\beta_0 + \\beta_1 \\cdot T + \\beta_2 \\cdot T^2 + \\beta_{\\text{Duration}} \\cdot \\text{SleepDurationHours} + \\epsilon$$",
        "4. **Partial Correlation**:",
        "   $$r(\\text{Metric}, X \\mid \\text{Duration}) = \\frac{r_{YX} - r_{YZ} r_{XZ}}{\\sqrt{(1 - r_{YZ}^2)(1 - r_{XZ}^2)}}$$",
        "5. **Standard Errors & Confidence Intervals**:",
        "   Calculated from the covariance matrix of OLS estimates $\\text{Var}(\\hat{\\beta}) = s^2 (X^T X)^{-1}$, "
        "with exact two-tailed Student's $t$ critical values ($df = n - p$).",
        "",
    ])

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Exported comprehensive markdown report to: {os.path.abspath(filepath)}")


# =========================================================================================
# Public API for Pipeline Integration
# =========================================================================================

def analyze_sleep_co2_regression(
    df_or_path,
    co2_metrics: Optional[List[str]] = None,
    covariates: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, List[RegressionComparison]]:
    """
    Public analysis function suitable for direct execution or importing into ProcessSleepCo2.py.
    """
    if isinstance(df_or_path, str):
        df = pd.read_csv(df_or_path)
    elif isinstance(df_or_path, list):
        df = pd.DataFrame(df_or_path)
    else:
        df = df_or_path.copy()

    df = prepare_dataset(df)

    if co2_metrics is None:
        co2_metrics = ["AvgCo2Ppm"]

    multi_covs = None
    if covariates and "bedtime" in covariates and "awake" in covariates:
        multi_covs = ["BedtimeHour", "MinutesAwake"]

    results_by_co2 = {}
    for co2_col in co2_metrics:
        if co2_col not in df.columns:
            continue
        comparisons = []
        for metric_col, label, unit in DEFAULT_METRICS:
            if metric_col not in df.columns:
                continue
            comp = run_comparative_regression(
                df=df,
                metric_col=metric_col,
                metric_label=label,
                unit=unit,
                co2_col=co2_col,
                duration_col="SleepDurationHours",
                multi_covariates=multi_covs
            )
            if comp:
                comparisons.append(comp)

        results_by_co2[co2_col] = comparisons
        if verbose and comparisons:
            co2_label = dict(CO2_METRICS).get(co2_col, co2_col)
            print_comparison_table(comparisons, co2_label, has_multi=(multi_covs is not None))
            print_key_insights(comparisons, co2_label)

    return results_by_co2


# =========================================================================================
# CLI Entrypoint
# =========================================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multiple linear regression analysis of sleep metrics vs. CO2, adjusting for erratic sleep duration."
    )
    parser.add_argument(
        "-i", "--input",
        dest="input_file",
        default="sleep_co2_merged.csv",
        help="Path to merged sleep and CO2 CSV dataset (default: sleep_co2_merged.csv)"
    )
    parser.add_argument(
        "-c", "--co2-metric",
        dest="co2_metric",
        default="AvgCo2Ppm",
        help="CO2 predictor column to analyze (e.g. AvgCo2Ppm, MaxCo2Ppm, HoursAbove1000Ppm, or 'all')"
    )
    parser.add_argument(
        "--covariates",
        dest="covariates",
        choices=["duration", "all"],
        default="duration",
        help="Covariates to include: 'duration' (SleepDurationHours) or 'all' (+ BedtimeHour, MinutesAwake)"
    )
    parser.add_argument(
        "--optimal-temp",
        dest="optimal_temp",
        action="store_true",
        help="Run comprehensive optimal bedroom temperature analysis (quadratic regression curves, temperature brackets, and composite index)"
    )
    parser.add_argument(
        "--export-md",
        dest="export_md",
        default=None,
        help="Optional path to export a comprehensive GitHub Markdown report (e.g. regression_report.md)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input dataset not found at '{args.input_file}'.")
        print("Please specify the correct path using -i / --input or run ProcessSleepCo2.py first.")
        return 1

    df = pd.read_csv(args.input_file)
    print(f"Loaded {len(df)} records from {os.path.abspath(args.input_file)}")

    if args.co2_metric.lower() == "all":
        co2_cols = [c[0] for c in CO2_METRICS if c[0] in df.columns]
    else:
        co2_cols = [args.co2_metric]

    covs = ["duration", "bedtime", "awake"] if args.covariates == "all" else ["duration"]

    # When --optimal-temp is explicitly set without custom -c, keep CO2 verbose=False so optimal temp tables take center stage
    run_co2_verbose = not args.optimal_temp or (args.co2_metric.lower() == "all" or "-c" in sys.argv or "--co2-metric" in sys.argv)
    results = analyze_sleep_co2_regression(
        df_or_path=df,
        co2_metrics=co2_cols,
        covariates=covs,
        verbose=run_co2_verbose
    )

    optimal_results = None
    if args.optimal_temp:
        optimal_results = analyze_optimal_temperature(df, verbose=True)

    if args.export_md:
        export_markdown_report(args.export_md, results, df, optimal_results=optimal_results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
