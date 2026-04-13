"""
Result dataclasses for py2sdid.

All estimation outputs are immutable(ish) dataclasses with typed fields,
following the pyfector convention. DiDResult provides ``.summary()``,
``.plot()``, and ``.diagnose()`` methods for interactive use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from scipy import sparse


# ---------------------------------------------------------------------------
# PanelData
# ---------------------------------------------------------------------------

@dataclass
class PanelData:
    """Structured panel data ready for estimation.

    All arrays are observation-level (length ``n_obs``) in long format.
    Integer-coded identifiers map back to original labels via the
    ``unit_map``, ``time_map``, and ``cluster_map`` dictionaries.
    """

    # Core observation-level arrays (all length n_obs)
    Y: np.ndarray                   # outcome vector
    D: np.ndarray                   # binary treatment indicator (0/1)
    unit_ids: np.ndarray            # integer-coded unit identifiers
    time_ids: np.ndarray            # integer-coded time period identifiers
    cohort: np.ndarray              # treatment cohort (0 = never-treated)
    event_time: np.ndarray          # relative time = t - g (inf for never-treated)

    # Optional arrays
    X: np.ndarray | None            # (n_obs, K) covariates matrix
    W: np.ndarray | None            # (n_obs,) observation weights
    cluster: np.ndarray             # (n_obs,) cluster identifiers (integer-coded)

    # Dimensions
    n_units: int
    n_periods: int
    n_treated: int
    n_control: int
    n_obs: int
    cohort_sizes: dict[int, int]    # cohort value -> count of units

    # Masks
    is_treated: np.ndarray          # (n_obs,) bool: D_it == 1
    is_control: np.ndarray          # (n_obs,) bool: D_it == 0

    # Mappings (integer code -> original label)
    unit_map: dict[int, Any]
    time_map: dict[int, Any]
    cluster_map: dict[int, Any]

    # Fixed-effect structure (defaults match panel mode for backward compat)
    fe_ids: np.ndarray | None = None        # integer-coded FE identifiers (= unit_ids for panel, group_ids for RCS)
    n_fe_levels: int | None = None          # number of FE levels (= n_units for panel, n_groups for RCS)
    fe_map: dict[int, Any] | None = None    # FE code -> original label
    is_rcs: bool = False                    # True when groupname was provided

    # Singleton detection (observations excluded from ATT and inference)
    is_singleton: np.ndarray | None = None  # (n_obs,) bool: True for singleton obs
    n_singletons: int = 0                   # count of singleton observations dropped

    def __post_init__(self) -> None:
        """Default fe_ids to unit_ids for backward compatibility."""
        if self.fe_ids is None:
            self.fe_ids = self.unit_ids
        if self.n_fe_levels is None:
            self.n_fe_levels = self.n_units
        if self.fe_map is None:
            self.fe_map = self.unit_map
        if self.is_singleton is None:
            self.is_singleton = np.zeros(self.n_obs, dtype=bool)


# ---------------------------------------------------------------------------
# FirstStageResult
# ---------------------------------------------------------------------------

@dataclass
class FirstStageResult:
    """Output from first-stage fixed-effects estimation on untreated obs.

    The first stage estimates unit FE + time FE (+ covariates) using only
    the untreated subsample, then predicts Y(0) for all observations.
    """

    y_hat: np.ndarray               # (n_obs,) predicted Y(0) for all obs
    residuals: np.ndarray           # (n_obs,) first-stage residuals (Y - y_hat)
    beta: np.ndarray                # (n_units + n_periods + K,) full coefficient vector
    unit_fe: np.ndarray             # (n_units,) unit fixed effects
    time_fe: np.ndarray             # (n_periods,) time fixed effects
    covar_coefs: np.ndarray | None  # (K,) covariate coefficients
    sigma2: float                   # error variance estimate
    design_full: sparse.csc_matrix  # (n_obs, p) full-sample design matrix
    design_ctrl: sparse.csc_matrix  # (n_ctrl, p) control-sample design matrix


# ---------------------------------------------------------------------------
# EffectsResult
# ---------------------------------------------------------------------------

@dataclass
class EffectsResult:
    """Output from treatment effect computation and aggregation.

    Contains both the overall ATT and the dynamic (event-study) ATT
    broken down by event horizon.
    """

    att_avg: float                          # overall ATT
    att_by_horizon: np.ndarray              # (n_horizons,) ATT per event horizon
    horizons: np.ndarray                    # (n_horizons,) horizon labels
    counts: np.ndarray                      # (n_horizons,) obs count per horizon
    effects: np.ndarray                     # (n_treated_obs,) individual tau_it

    # Pre-trend estimates (horizons < 0, excluding reference period)
    pretrend_att: np.ndarray | None = None          # (n_pre,) pre-trend ATTs
    pretrend_horizons: np.ndarray | None = None     # (n_pre,) pre-trend horizon labels

    # Weights for SE computation
    weights_matrix: np.ndarray | None = None        # treatment weight vectors


# ---------------------------------------------------------------------------
# InferenceResult
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    """Standard errors, confidence intervals, and p-values.

    Supports both analytic (influence function / BJS formula) and
    bootstrap inference.
    """

    # Overall ATT inference
    att_avg_se: float
    att_avg_ci: tuple[float, float]
    att_avg_pval: float

    # Per-horizon inference
    horizon_se: np.ndarray              # (n_horizons,)
    horizon_ci_lower: np.ndarray        # (n_horizons,)
    horizon_ci_upper: np.ndarray        # (n_horizons,)
    horizon_pval: np.ndarray            # (n_horizons,)

    # Full variance-covariance and bootstrap distribution
    vcov: np.ndarray | None = None              # (n_horizons, n_horizons)
    boot_dist: np.ndarray | None = None         # (n_bootstraps, n_horizons)


# ---------------------------------------------------------------------------
# DiagnosticResult
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticResult:
    """Container for diagnostic test results.

    Populated by ``diagnostics.run_diagnostics()`` and accessed via
    ``DiDResult.diagnose()``.
    """

    # Pre-trend F-test
    pretrend_f_stat: float
    pretrend_f_pval: float
    pretrend_df: tuple[int, int]

    # Equivalence test (TOST)
    equiv_results: pl.DataFrame | None = None   # rel_time, tost_pval, bound, reject
    equiv_max_pval: float | None = None         # max TOST p-value across all pre-periods
    equiv_all_pass: bool | None = None          # True if all pre-periods pass equivalence

    # Placebo test
    placebo_results: pl.DataFrame | None = None  # horizon, estimate, se, pval

    # HonestDiD sensitivity
    honestdid_results: pl.DataFrame | None = None  # horizon, M, ci_lower, ci_upper

    def summary(self) -> str:
        """Return human-readable summary of diagnostic tests."""
        lines = ["Diagnostic Tests", "=" * 50]

        # Pre-trend F-test
        df1, df2 = self.pretrend_df
        lines.append(
            f"Pre-trend F-test: F({df1}, {df2}) = {self.pretrend_f_stat:.4f}, "
            f"p = {self.pretrend_f_pval:.4f}"
        )

        # Equivalence TOST
        if self.equiv_results is not None:
            lines.append("")
            if self.equiv_max_pval is not None:
                status = "PASS" if self.equiv_all_pass else "FAIL"
                lines.append(
                    f"Equivalence test (TOST): {status} "
                    f"(max p = {self.equiv_max_pval:.4f})"
                )
            lines.append("Per-period equivalence:")
            for row in self.equiv_results.iter_rows(named=True):
                status = "PASS" if row["reject"] else "fail"
                lines.append(
                    f"  e={row['rel_time']:+d}: p={row['tost_pval']:.4f} "
                    f"(bound={row['bound']:.4f}) [{status}]"
                )

        # Placebo
        if self.placebo_results is not None:
            lines.append("")
            lines.append("Placebo test:")
            for row in self.placebo_results.iter_rows(named=True):
                lines.append(
                    f"  e={row['rel_time']:+d}: est={row['estimate']:.4f}, "
                    f"se={row['se']:.4f}, p={row['pval']:.4f}"
                )

        # HonestDiD
        if self.honestdid_results is not None:
            lines.append("")
            lines.append("HonestDiD sensitivity:")
            for row in self.honestdid_results.iter_rows(named=True):
                lines.append(
                    f"  M={row['M']:.3f}: CI=[{row['ci_lower']:.4f}, "
                    f"{row['ci_upper']:.4f}]"
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DiDResult
# ---------------------------------------------------------------------------

@dataclass
class DiDResult:
    """Main result container for py2sdid estimation.

    Returned by ``py2sdid.ts_did()`` and ``py2sdid.bjs_did()``.
    Provides ``.summary()``, ``.plot()``, and ``.diagnose()`` methods.
    """

    # Method info
    method: str                                     # "ts_did" or "bjs_did"

    # Point estimates
    att_avg: float                                  # overall ATT
    att_avg_se: float | None                        # SE of overall ATT
    att_avg_ci: tuple[float, float] | None          # confidence interval
    att_avg_pval: float | None                      # p-value

    # Event-study results (ALL relative time periods, pre + post)
    event_study: pl.DataFrame                       # rel_time, estimate, se, ci_lower, ci_upper, pval, count

    # First-stage components
    unit_fe: np.ndarray                             # (n_units,) estimated unit FE
    time_fe: np.ndarray                             # (n_periods,) estimated time FE
    beta: np.ndarray | None                         # (K,) covariate coefficients
    covariate_names: list[str] | None               # covariate column names

    # Treatment effects
    effects: np.ndarray                             # (n_treated_obs,) individual tau_it
    y_hat: np.ndarray                               # (n_obs,) first-stage Y(0) predictions

    # Inference details
    vcov: np.ndarray | None                         # variance-covariance of ATT estimates
    boot_dist: np.ndarray | None                    # (n_bootstraps, n_horizons) bootstrap dist

    # Metadata
    panel: PanelData
    sigma2: float                                   # error variance from first stage
    seed: int | None

    @property
    def att_by_horizon(self) -> pl.DataFrame:
        """Post-treatment event-study estimates (rel_time >= 0)."""
        return self.event_study.filter(pl.col("rel_time") >= 0)

    @property
    def pretrend_tests(self) -> pl.DataFrame | None:
        """Pre-treatment event-study estimates (rel_time < 0)."""
        pre = self.event_study.filter(pl.col("rel_time") < 0)
        return pre if len(pre) > 0 else None

    def summary(self) -> str:
        """Return formatted summary table of estimation results."""
        method_label = {
            "ts_did": "Two-Stage DiD (Gardner 2021)",
            "bjs_did": "BJS Imputation (Borusyak, Jaravel, Spiess 2024)",
        }.get(self.method, self.method)

        p = self.panel
        n_cohorts = len([c for c in p.cohort_sizes if c != 0])
        n_treated_units = sum(v for k, v in p.cohort_sizes.items() if k != 0)

        n_clusters = len(p.cluster_map)
        cluster_label = f"{n_clusters} clusters"

        lines = []
        lines.append(f"py2sdid estimation results ({self.method})")
        lines.append("=" * 60)
        lines.append(f"Method: {method_label}")
        if p.is_rcs:
            lines.append(
                f"Observations: {p.n_obs:,}  (G={p.n_fe_levels}, T={p.n_periods})"
            )
            lines.append(
                f"Treated: {p.n_treated} obs     "
                f"Cohorts: {n_cohorts}  (repeated cross-section)"
            )
        else:
            lines.append(
                f"Observations: {p.n_obs:,}  (N={p.n_units}, T={p.n_periods})"
            )
            lines.append(
                f"Treated: {n_treated_units} units ({p.n_treated} obs)     "
                f"Cohorts: {n_cohorts}"
            )
        lines.append(f"Clustering: {cluster_label}")
        lines.append("")

        # ATT
        lines.append(f"ATT (average):  {self.att_avg:.4f}")
        if self.att_avg_se is not None:
            lines.append(f"  SE:           {self.att_avg_se:.4f}")
        if self.att_avg_ci is not None:
            lines.append(
                f"  95% CI:       [{self.att_avg_ci[0]:.4f}, "
                f"{self.att_avg_ci[1]:.4f}]"
            )
        if self.att_avg_pval is not None:
            lines.append(f"  p-value:      {self.att_avg_pval:.4f}")

        # Per-period event-study table
        if len(self.event_study) > 0:
            lines.append("")
            lines.append("Per-period estimates (event study):")
            lines.append(
                "  Rel.Time    ATT       SE       CI              "
                "p-val   Count"
            )
            lines.append(
                "  --------  -------  -------  ----------------  "
                "------  ------"
            )
            for row in self.event_study.iter_rows(named=True):
                h = row["rel_time"]
                est = row["estimate"]
                se = row.get("se")
                ci_lo = row.get("ci_lower")
                ci_hi = row.get("ci_upper")
                pval = row.get("pval")
                count = row.get("count")

                est_s = f"{est:8.4f}" if est is not None else "       ."
                se_s = f"{se:8.4f}" if se is not None else "       ."
                if ci_lo is not None and ci_hi is not None:
                    ci_s = f"[{ci_lo:7.4f}, {ci_hi:7.4f}]"
                else:
                    ci_s = "       .        "
                pval_s = f"{pval:6.3f}" if pval is not None else "     ."
                count_s = f"{count:6d}" if count is not None else "     ."

                lines.append(
                    f"  {h:8d} {est_s} {se_s}  {ci_s}  {pval_s}  {count_s}"
                )

        # Covariate coefficients
        if self.beta is not None and self.covariate_names:
            lines.append("")
            lines.append("Covariates:")
            for i, name in enumerate(self.covariate_names):
                lines.append(f"  {name}: {self.beta[i]:.6f}")

        lines.append("=" * 60)
        lines.append(f"Sigma^2: {self.sigma2:.4f}")
        if self.seed is not None:
            lines.append(f"Seed: {self.seed}")

        return "\n".join(lines)

    def plot(self, kind: str = "event_study", **kwargs: Any) -> Any:
        """Create a plot of the estimation results.

        Parameters
        ----------
        kind : str
            Plot type. One of ``"event_study"``, ``"pretrends"``,
            ``"treatment_status"``, ``"counterfactual"``, ``"honestdid"``,
            ``"calendar"``.
        **kwargs
            Passed to the plotting function.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from .plotting import plot as _plot

        return _plot(self, kind=kind, **kwargs)

    def diagnose(self, **kwargs: Any) -> DiagnosticResult:
        """Run diagnostic tests on the estimation results.

        Parameters
        ----------
        **kwargs
            Passed to ``diagnostics.run_diagnostics()``.

        Returns
        -------
        DiagnosticResult
        """
        from .diagnostics import run_diagnostics

        return run_diagnostics(self, **kwargs)
