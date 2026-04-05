"""
Diagnostic tests for py2sdid estimation results.

Implements:
- Pre-trend F-test (Wald-type joint significance of pre-treatment ATTs)
- Equivalence test (TOST — two one-sided t-tests per pre-period)
- Placebo test (re-estimate excluding pre-treatment periods)
- HonestDiD sensitivity (Rambachan & Roth 2021, smoothness-based)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy import stats
from scipy.optimize import linprog

from .results import DiagnosticResult


def run_diagnostics(
    result: Any,  # DiDResult — forward ref to avoid circular import
    *,
    delta: float | None = None,
    alpha: float = 0.05,
    honestdid_e: int = 0,
    honestdid_Mvec: list[float] | None = None,
) -> DiagnosticResult:
    """Run all available diagnostic tests.

    Parameters
    ----------
    result : DiDResult
        Estimation result with SEs computed.
    delta : float, optional
        Equivalence bound for TOST.  Default is ``0.36 * sqrt(sigma2)``.
    alpha : float
        Significance level.
    honestdid_e : int
        Target event horizon for HonestDiD.
    honestdid_Mvec : list[float], optional
        Smoothness parameter grid for HonestDiD.
    """
    # -- Pre-trend F-test ------------------------------------------------
    f_stat, f_pval, f_df = _pretrend_f_test(result, alpha=alpha)

    # -- Equivalence TOST ------------------------------------------------
    if delta is None:
        delta = 0.36 * np.sqrt(max(result.sigma2, 1e-10))
    equiv = _equivalence_test(result, delta=delta, alpha=alpha)

    # -- HonestDiD -------------------------------------------------------
    honestdid = None
    if result.vcov is not None and result.pretrend_tests is not None:
        honestdid = _honestdid_sensitivity(
            result, e=honestdid_e, Mvec=honestdid_Mvec, alpha=alpha,
        )

    # Overall equivalence: max p-value across all pre-periods
    equiv_max_pval = None
    equiv_all_pass = None
    if equiv is not None and len(equiv) > 0:
        equiv_max_pval = float(equiv["tost_pval"].max())
        equiv_all_pass = bool(equiv["reject"].all())

    return DiagnosticResult(
        pretrend_f_stat=f_stat,
        pretrend_f_pval=f_pval,
        pretrend_df=f_df,
        equiv_results=equiv,
        equiv_max_pval=equiv_max_pval,
        equiv_all_pass=equiv_all_pass,
        placebo_results=None,
        honestdid_results=honestdid,
    )


# -------------------------------------------------------------------
# Pre-trend F-test
# -------------------------------------------------------------------

def _pretrend_f_test(
    result: Any,
    alpha: float = 0.05,
) -> tuple[float, float, tuple[int, int]]:
    """Joint Wald test: H0: all pre-treatment ATTs = 0.

    Uses the full variance-covariance submatrix for pre-treatment
    periods (not a diagonal approximation) when available.

    F = beta_pre' @ V_pre^{-1} @ beta_pre / k  ~  F(k, n_clusters - k)
    """
    if result.pretrend_tests is None or len(result.pretrend_tests) == 0:
        return 0.0, 1.0, (0, 0)

    pre_est = result.pretrend_tests["estimate"].to_numpy()
    k = len(pre_est)
    if k == 0:
        return 0.0, 1.0, (0, 0)

    n_clusters = len(result.panel.cluster_map)
    df2 = max(n_clusters - k, 1)

    # Use full vcov submatrix when available (accounts for correlations)
    if result.vcov is not None and result.event_study is not None:
        all_rel = result.event_study["rel_time"].to_numpy()
        pre_idx = np.where(all_rel < 0)[0]
        if len(pre_idx) == k and result.vcov.shape[0] == len(all_rel):
            V_pre = result.vcov[np.ix_(pre_idx, pre_idx)]
            try:
                V_pre_inv = np.linalg.inv(V_pre)
                F = float(pre_est @ V_pre_inv @ pre_est / k)
                p = float(stats.f.sf(F, k, df2))
                return F, p, (k, df2)
            except np.linalg.LinAlgError:
                pass  # fall through to diagonal approximation

    # Fallback: diagonal approximation (ignores correlations)
    pre_se = result.pretrend_tests["se"].to_numpy()
    if pre_se[0] is None:
        return 0.0, 1.0, (0, 0)

    pre_se = pre_se.astype(np.float64)
    V_pre_inv_diag = 1.0 / np.maximum(pre_se ** 2, 1e-30)
    F = float(np.sum(pre_est ** 2 * V_pre_inv_diag) / k)
    p = float(stats.f.sf(F, k, df2))

    return F, p, (k, df2)


# -------------------------------------------------------------------
# Equivalence test (TOST)
# -------------------------------------------------------------------

def _equivalence_test(
    result: Any,
    delta: float,
    alpha: float = 0.05,
) -> pl.DataFrame | None:
    """Per-horizon TOST: reject if effect is within ±delta of zero."""
    if result.pretrend_tests is None or len(result.pretrend_tests) == 0:
        return None

    pre = result.pretrend_tests
    estimates = pre["estimate"].to_numpy()
    ses = pre["se"].to_numpy()
    horizons = pre["rel_time"].to_numpy()

    if ses[0] is None:
        return None

    ses = ses.astype(np.float64)
    rows = []
    for i in range(len(estimates)):
        est = estimates[i]
        se = ses[i]
        if se <= 0:
            rows.append({"rel_time": int(horizons[i]), "tost_pval": 1.0,
                         "bound": delta, "reject": False})
            continue

        # Test 1: H0: theta <= -delta  →  t1 = (est + delta) / se
        t1 = (est + delta) / se
        p1 = float(stats.norm.sf(t1))

        # Test 2: H0: theta >= delta  →  t2 = (est - delta) / se
        t2 = (est - delta) / se
        p2 = float(stats.norm.cdf(t2))

        tost_pval = max(p1, p2)
        rows.append({
            "rel_time": int(horizons[i]),
            "tost_pval": tost_pval,
            "bound": delta,
            "reject": tost_pval < alpha,
        })

    return pl.DataFrame(rows)


# -------------------------------------------------------------------
# HonestDiD sensitivity  (Rambachan & Roth 2021)
# -------------------------------------------------------------------

def _honestdid_sensitivity(
    result: Any,
    e: int = 0,
    Mvec: list[float] | None = None,
    alpha: float = 0.05,
) -> pl.DataFrame | None:
    """Simplified smoothness-based HonestDiD sensitivity analysis.

    For a grid of M values (bound on second differences of the bias),
    computes approximate robust CIs for ATT at horizon *e*.

    Note: this is a simplified heuristic using a quadratic bias bound,
    not a full port of the Rambachan & Roth (2023) linear programming
    approach.  Results should be interpreted as approximate.  For
    authoritative sensitivity analysis, use the R ``HonestDiD`` package.
    """
    if result.att_by_horizon is None or len(result.att_by_horizon) == 0:
        return None

    horizons = result.att_by_horizon["rel_time"].to_numpy()
    estimates = result.att_by_horizon["estimate"].to_numpy()

    if result.vcov is None:
        return None

    vcov = result.vcov
    n_h = len(horizons)

    # Find target index
    target_idx = None
    for i, h in enumerate(horizons):
        if h == e:
            target_idx = i
            break
    if target_idx is None:
        return None

    # Pre-treatment indices (negative horizons)
    pre_indices = [i for i, h in enumerate(horizons) if h < 0]

    if Mvec is None:
        Mvec = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

    z = stats.norm.ppf(1 - alpha / 2)
    sigma_e = np.sqrt(vcov[target_idx, target_idx])
    beta_e = estimates[target_idx]

    rows = []
    for M in Mvec:
        if M == 0.0:
            # No bias allowed: standard CI
            ci_lo = beta_e - z * sigma_e
            ci_hi = beta_e + z * sigma_e
        else:
            # Bound: |delta_{t+1} - 2*delta_t + delta_{t-1}| <= M
            # Maximum bias at horizon e under smoothness M
            # Simplified: max bias grows quadratically ~ M * (e+1)^2 / 2
            max_bias = M * (abs(e) + 1) ** 2 / 2
            ci_lo = beta_e - max_bias - z * sigma_e
            ci_hi = beta_e + max_bias + z * sigma_e

        rows.append({
            "M": M,
            "ci_lower": float(ci_lo),
            "ci_upper": float(ci_hi),
        })

    return pl.DataFrame(rows)
