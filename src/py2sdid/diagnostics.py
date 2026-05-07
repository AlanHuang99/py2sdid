"""
Diagnostic tests for py2sdid estimation results.

Implements:
- Pre-trend F-test (Wald-type joint significance of pre-treatment ATTs)
- Equivalence test (TOST — two one-sided t-tests per pre-period)
- Placebo test (average ATT over a pre-treatment window)
- HonestDiD sensitivity (Rambachan & Roth 2021, smoothness-based)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from .results import (
    DiagnosticResult,
    HonestDiDResult,
    PlaceboResult,
    PretrendFResult,
    TostResult,
)


_VALID_DIAG_NAMES = {"pretrend_f", "tost", "placebo", "honestdid"}
_VALID_OPTION_KEYS = {
    "delta", "alpha", "placebo_period", "honestdid_e", "honestdid_Mvec",
}
_DEFAULT_FULL_DIAGNOSTICS = ("pretrend_f", "tost", "honestdid")


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and np.isfinite(value)
    )


def _is_positive_finite_number(value: Any) -> bool:
    return _is_finite_number(value) and value > 0


def _validate_period_option(name: str, value: Any) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or not _is_finite_number(value[0])
        or not _is_finite_number(value[1])
    ):
        raise ValueError(
            f"diagnostics_options['{name}'] must be a (start, end) tuple "
            "of finite numeric periods."
        )
    start = int(value[0])
    end = int(value[1])
    if start > end:
        raise ValueError(
            f"diagnostics_options['{name}'] must have start <= end."
        )
    return start, end


def _validate_diagnostics_options(options: dict[str, Any]) -> None:
    if "delta" in options and options["delta"] is not None:
        if not _is_positive_finite_number(options["delta"]):
            raise ValueError(
                "diagnostics_options['delta'] must be a positive finite float."
            )

    if "alpha" in options:
        alpha = options["alpha"]
        if not _is_finite_number(alpha) or not 0 < alpha < 1:
            raise ValueError(
                "diagnostics_options['alpha'] must be a finite float in (0, 1)."
            )

    if "placebo_period" in options and options["placebo_period"] is not None:
        _validate_period_option("placebo_period", options["placebo_period"])

    if "honestdid_e" in options:
        honestdid_e = options["honestdid_e"]
        if isinstance(honestdid_e, bool) or not isinstance(
            honestdid_e, (int, np.integer)
        ):
            raise ValueError("diagnostics_options['honestdid_e'] must be an int.")

    if "honestdid_Mvec" in options and options["honestdid_Mvec"] is not None:
        Mvec = options["honestdid_Mvec"]
        if not isinstance(Mvec, list) or not Mvec:
            raise ValueError(
                "diagnostics_options['honestdid_Mvec'] must be a non-empty list."
            )
        if any(not _is_finite_number(M) or M < 0 for M in Mvec):
            raise ValueError(
                "diagnostics_options['honestdid_Mvec'] values must be "
                "finite non-negative floats."
            )


def validate_diagnostics_request(
    diagnostics: Any,
    diagnostics_options: dict | None,
    se: bool,
) -> list[str] | None:
    """Validate fit-time diagnostics before expensive estimation starts."""
    if diagnostics_options is not None and not isinstance(diagnostics_options, dict):
        raise ValueError("diagnostics_options must be a dict or None.")

    opts: dict[str, Any] = diagnostics_options or {}
    bad_keys = set(opts) - _VALID_OPTION_KEYS
    if bad_keys:
        raise ValueError(
            f"unknown diagnostics_options key(s): {sorted(bad_keys)}; "
            f"valid: {sorted(_VALID_OPTION_KEYS)}"
        )

    if diagnostics == "none":
        _validate_diagnostics_options(opts)
        return None

    if not se:
        raise ValueError("diagnostics requires se=True.")

    if diagnostics == "full":
        requested = list(_DEFAULT_FULL_DIAGNOSTICS)
        if opts.get("placebo_period") is not None:
            requested.append("placebo")
    elif isinstance(diagnostics, list):
        if len(diagnostics) == 0:
            raise ValueError(
                "diagnostics=[] is invalid; pass 'none' or a non-empty list."
            )
        bad = [name for name in diagnostics if name not in _VALID_DIAG_NAMES]
        if bad:
            raise ValueError(
                f"Unknown diagnostic(s): {bad}; valid: {sorted(_VALID_DIAG_NAMES)}"
            )
        if "placebo" in diagnostics and opts.get("placebo_period") is None:
            raise ValueError(
                "diagnostics list includes 'placebo' but "
                "diagnostics_options['placebo_period'] is missing."
            )
        requested = list(diagnostics)
    else:
        raise ValueError(
            f"diagnostics must be 'none', 'full', or a list of names; got "
            f"{diagnostics!r}"
        )

    _validate_diagnostics_options(opts)
    return requested


def run_diagnostics(
    result: Any,  # DiDResult — forward ref to avoid circular import
    *,
    delta: float | None = None,
    alpha: float = 0.05,
    placebo_period: tuple[int, int] | None = None,
    honestdid_e: int = 0,
    honestdid_Mvec: list[float] | None = None,
    _requested: list[str] | None = None,
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
    placebo_period : tuple[int, int], optional
        Inclusive pre-treatment relative-time window to average for a
        placebo test.
    honestdid_e : int
        Target event horizon for HonestDiD.
    honestdid_Mvec : list[float], optional
        Smoothness parameter grid for HonestDiD.
    """
    options: dict[str, Any] = {
        "delta": delta,
        "alpha": alpha,
        "placebo_period": placebo_period,
        "honestdid_e": honestdid_e,
        "honestdid_Mvec": honestdid_Mvec,
    }
    _validate_diagnostics_options(options)

    requested = set(_requested) if _requested is not None else _VALID_DIAG_NAMES.copy()
    if bad := requested - _VALID_DIAG_NAMES:
        raise ValueError(f"Unknown diagnostic(s): {sorted(bad)}")

    # -- Pre-trend F-test ------------------------------------------------
    diag = DiagnosticResult()

    if "pretrend_f" in requested:
        f_stat, f_pval, f_df = _pretrend_f_test(result, alpha=alpha)
        diag.pretrend_f = PretrendFResult(
            f_stat=f_stat,
            p_value=f_pval,
            df1=f_df[0],
            df2=f_df[1],
        )

    # -- Equivalence TOST ------------------------------------------------
    if delta is None:
        delta = 0.36 * np.sqrt(max(result.sigma2, 1e-10))
    if "tost" in requested:
        diag.tost = _equivalence_test(result, delta=delta, alpha=alpha)

    # -- Placebo ---------------------------------------------------------
    if "placebo" in requested and placebo_period is not None:
        diag.placebo = _placebo_test(
            result, placebo_period=placebo_period, delta=delta,
        )

    # -- HonestDiD -------------------------------------------------------
    if (
        "honestdid" in requested
        and result.vcov is not None
        and result.pretrend_tests is not None
    ):
        diag.honestdid = _honestdid_sensitivity(
            result, e=honestdid_e, Mvec=honestdid_Mvec, alpha=alpha,
        )

    diag.options = {
        "delta": float(delta),
        "alpha": float(alpha),
        "placebo_period": (
            (int(placebo_period[0]), int(placebo_period[1]))
            if placebo_period is not None else None
        ),
        "honestdid_e": int(honestdid_e),
        "honestdid_Mvec": (
            [float(M) for M in honestdid_Mvec]
            if honestdid_Mvec is not None else None
        ),
    }

    return diag


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

    n_clusters = getattr(result, "n_clusters", None)
    panel = getattr(result, "panel", None)
    if n_clusters is None and panel is not None:
        n_clusters = len(panel.cluster_map)
    if n_clusters is None:
        n_clusters = k + 1
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
) -> TostResult | None:
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
    pvals = np.full(len(estimates), np.nan, dtype=np.float64)
    for i in range(len(estimates)):
        est = estimates[i]
        se = ses[i]
        if se <= 0:
            pvals[i] = 1.0
            continue

        # Test 1: H0: theta <= -delta  →  t1 = (est + delta) / se
        t1 = (est + delta) / se
        p1 = float(stats.norm.sf(t1))

        # Test 2: H0: theta >= delta  →  t2 = (est - delta) / se
        t2 = (est - delta) / se
        p2 = float(stats.norm.cdf(t2))

        pvals[i] = max(p1, p2)

    finite = pvals[np.isfinite(pvals)]
    max_pval = float(finite.max()) if finite.size else float("nan")
    all_pass = bool(finite.size and (finite < alpha).all())
    return TostResult(
        pvals=pvals,
        periods=horizons.astype(int),
        threshold=float(delta),
        max_pval=max_pval,
        all_pass=all_pass,
    )


# -------------------------------------------------------------------
# Placebo test
# -------------------------------------------------------------------

def _placebo_test(
    result: Any,
    *,
    placebo_period: tuple[int, int],
    delta: float,
) -> PlaceboResult | None:
    """Average pre-period ATT in a placebo window.

    Uses the bootstrap distribution when available. Falls back to the
    event-study covariance matrix so analytic-SE fits can still report a
    Wald-style placebo result.
    """
    if result.event_study is None or len(result.event_study) == 0:
        return None

    start, end = int(placebo_period[0]), int(placebo_period[1])
    horizons = result.event_study["rel_time"].to_numpy()
    estimates = result.event_study["estimate"].to_numpy()
    mask = (horizons >= start) & (horizons <= end) & (horizons < 0)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None

    estimate = float(np.mean(estimates[idx]))
    se = float("nan")
    p_value = float("nan")
    equiv_p_value = float("nan")

    boot_dist = getattr(result, "boot_dist", None)
    if boot_dist is not None:
        boot_arr = np.asarray(boot_dist, dtype=np.float64)
        if boot_arr.ndim == 2 and boot_arr.shape[1] == len(horizons):
            boot_window = boot_arr[:, idx]
            valid = np.all(np.isfinite(boot_window), axis=1)
            boot_placebo = np.mean(boot_window[valid], axis=1)
            if boot_placebo.size:
                p_value = min(
                    2 * float(np.mean(boot_placebo >= 0)),
                    2 * float(np.mean(boot_placebo <= 0)),
                    1.0,
                )
            if boot_placebo.size > 1:
                se = float(np.std(boot_placebo, ddof=1))
                if se > 0:
                    df = boot_placebo.size - 1
                    t_upper = (estimate - delta) / se
                    t_lower = (estimate + delta) / se
                    equiv_p_value = max(
                        float(stats.t.cdf(t_upper, df)),
                        float(1 - stats.t.cdf(t_lower, df)),
                    )

    elif result.vcov is not None:
        vcov = np.asarray(result.vcov, dtype=np.float64)
        if vcov.ndim == 2 and vcov.shape == (len(horizons), len(horizons)):
            weights = np.zeros(len(horizons), dtype=np.float64)
            weights[idx] = 1.0 / len(idx)
            var = float(weights @ vcov @ weights)
            if np.isfinite(var) and var > 0:
                se = float(np.sqrt(var))
                p_value = float(2 * stats.norm.sf(abs(estimate) / se))
                t_upper = (estimate - delta) / se
                t_lower = (estimate + delta) / se
                equiv_p_value = max(
                    float(stats.norm.cdf(t_upper)),
                    float(1 - stats.norm.cdf(t_lower)),
                )

    return PlaceboResult(
        estimate=estimate,
        se=se,
        p_value=p_value,
        equiv_p_value=equiv_p_value,
        period=(start, end),
    )


# -------------------------------------------------------------------
# HonestDiD sensitivity  (Rambachan & Roth 2021)
# -------------------------------------------------------------------

def _honestdid_sensitivity(
    result: Any,
    e: int = 0,
    Mvec: list[float] | None = None,
    alpha: float = 0.05,
) -> HonestDiDResult | None:
    """Simplified smoothness-based HonestDiD sensitivity analysis.

    For a grid of M values (bound on second differences of the bias),
    computes approximate robust CIs for ATT at horizon *e*.

    Note: this is a simplified heuristic using a quadratic bias bound,
    not a full port of the Rambachan & Roth (2023) linear programming
    approach.  Results should be interpreted as approximate.  For
    authoritative sensitivity analysis, use the R ``HonestDiD`` package.
    """
    if result.event_study is None or len(result.event_study) == 0:
        return None

    horizons = result.event_study["rel_time"].to_numpy()
    estimates = result.event_study["estimate"].to_numpy()

    if result.vcov is None:
        return None

    vcov = result.vcov
    # Find target index
    target_idx = None
    for i, h in enumerate(horizons):
        if h == e:
            target_idx = i
            break
    if target_idx is None:
        return None

    if Mvec is None:
        Mvec = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

    z = stats.norm.ppf(1 - alpha / 2)
    sigma_e = np.sqrt(vcov[target_idx, target_idx])
    beta_e = estimates[target_idx]

    M_values = []
    ci_lowers = []
    ci_uppers = []
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

        M_values.append(float(M))
        ci_lowers.append(float(ci_lo))
        ci_uppers.append(float(ci_hi))

    return HonestDiDResult(
        M=np.array(M_values, dtype=np.float64),
        ci_lower=np.array(ci_lowers, dtype=np.float64),
        ci_upper=np.array(ci_uppers, dtype=np.float64),
    )
