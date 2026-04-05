"""
Core orchestration — ties together panel, first-stage, effects, and inference.

Provides the two public entry points ``ts_did()`` and ``bjs_did()``.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import polars as pl

from .effects import compute_effects
from .first_stage import estimate_first_stage
from .inference import compute_se_bjs, compute_se_did2s, run_bootstrap
from .panel import prepare_panel
from .results import DiDResult, EffectsResult, InferenceResult, PanelData


# ===================================================================
# Public API
# ===================================================================

def ts_did(
    data: pl.DataFrame,
    yname: str,
    idname: str,
    tname: str,
    gname: str,
    *,
    xformla: list[str] | None = None,
    wname: str | None = None,
    cluster_var: str | None = None,
    se: bool = True,
    bootstrap: bool = False,
    n_bootstraps: int = 500,
    seed: int | None = None,
    n_jobs: int | None = None,
    verbose: bool = True,
) -> DiDResult:
    """Two-stage Difference-in-Differences (Gardner 2021).

    Estimates per-period treatment effects for all relative time periods
    by regressing the residualized outcome on treatment indicators.
    Standard errors are computed via GMM influence functions that correct
    for the generated-regressor problem.

    Parameters
    ----------
    data : pl.DataFrame
        Panel data in long format (one row per unit per period).
    yname : str
        Outcome variable column name.
    idname : str
        Unit identifier column name.
    tname : str
        Time period column name (integer-valued).
    gname : str
        Treatment cohort column name. Each value indicates the time
        period when treatment begins for that unit. Use ``0`` or
        ``null`` for never-treated units.
    xformla : list[str], optional
        Time-varying covariate column names to include in the first
        stage alongside unit and time fixed effects.
    wname : str, optional
        Observation weight column name. Weights are applied in both
        the first-stage estimation and the treatment effect aggregation.
    cluster_var : str, optional
        Column to cluster standard errors on. Defaults to ``idname``
        (unit-level clustering).
    se : bool
        Whether to compute standard errors, confidence intervals, and
        p-values.
    bootstrap : bool
        If ``True``, use cluster bootstrap instead of analytic
        influence-function standard errors.
    n_bootstraps : int
        Number of bootstrap replications (only used when ``bootstrap=True``).
    seed : int, optional
        Random seed for bootstrap reproducibility.
    n_jobs : int, optional
        Number of parallel workers for bootstrap. Defaults to the
        number of available CPU cores.
    verbose : bool
        Print step-by-step progress to stdout.

    Returns
    -------
    DiDResult
        Estimation results including per-period ATT estimates,
        standard errors, and methods for summary, plotting, and
        diagnostics.
    """
    return _run_estimation(
        data=data,
        yname=yname,
        idname=idname,
        tname=tname,
        gname=gname,
        method_name="ts_did",
        se_method_fn=compute_se_did2s,
        xformla=xformla,
        wname=wname,
        cluster_var=cluster_var,
        se=se,
        bootstrap=bootstrap,
        n_bootstraps=n_bootstraps,
        seed=seed,
        n_jobs=n_jobs,
        verbose=verbose,
    )


def bjs_did(
    data: pl.DataFrame,
    yname: str,
    idname: str,
    tname: str,
    gname: str,
    *,
    xformla: list[str] | None = None,
    wname: str | None = None,
    cluster_var: str | None = None,
    se: bool = True,
    bootstrap: bool = False,
    n_bootstraps: int = 500,
    seed: int | None = None,
    n_jobs: int | None = None,
    verbose: bool = True,
) -> DiDResult:
    """BJS Imputation Estimator (Borusyak, Jaravel, Spiess 2024).

    Same estimation procedure as :func:`ts_did` — produces identical
    point estimates. Differs only in the analytic standard error
    formula, which uses the BJS imputation approach (Equations 6, 8,
    10 of the paper) instead of GMM influence functions.

    Parameters
    ----------
    Same as :func:`ts_did`.
    """
    return _run_estimation(
        data=data,
        yname=yname,
        idname=idname,
        tname=tname,
        gname=gname,
        method_name="bjs_did",
        se_method_fn=compute_se_bjs,
        xformla=xformla,
        wname=wname,
        cluster_var=cluster_var,
        se=se,
        bootstrap=bootstrap,
        n_bootstraps=n_bootstraps,
        seed=seed,
        n_jobs=n_jobs,
        verbose=verbose,
    )


# ===================================================================
# Shared pipeline
# ===================================================================

def _run_estimation(
    *,
    data: pl.DataFrame,
    yname: str,
    idname: str,
    tname: str,
    gname: str,
    method_name: str,
    se_method_fn: Any,
    xformla: list[str] | None,
    wname: str | None,
    cluster_var: str | None,
    se: bool,
    bootstrap: bool,
    n_bootstraps: int,
    seed: int | None,
    n_jobs: int | None,
    verbose: bool,
) -> DiDResult:
    """Shared estimation pipeline for both ts_did and bjs_did."""

    if n_jobs is None:
        n_jobs = os.cpu_count() or 1

    # Step 1: Panel preparation
    if verbose:
        print(f"py2sdid [{method_name}]: preparing panel...", flush=True)

    panel = prepare_panel(
        data,
        yname=yname,
        idname=idname,
        tname=tname,
        gname=gname,
        xformla=xformla,
        wname=wname,
        cluster_var=cluster_var,
    )

    if verbose:
        n_cohorts = len([c for c in panel.cohort_sizes if c != 0])
        print(
            f"  {panel.n_units} units, {panel.n_periods} periods, "
            f"{panel.n_treated} treated obs, {n_cohorts} cohort(s)",
            flush=True,
        )

    # Step 2: First stage
    if verbose:
        print(
            f"  first_stage (sparse OLS, {panel.is_control.sum()} control obs)...",
            flush=True,
        )
    fs = estimate_first_stage(panel)

    # Step 3: Effects — always compute ALL per-period coefficients
    if verbose:
        print("  compute_effects...", flush=True)
    eff = compute_effects(panel, fs)
    if verbose:
        print(f"  ATT = {eff.att_avg:.4f}", flush=True)

    # Step 4: Inference
    inf_result: InferenceResult | None = None
    if se:
        if bootstrap:
            if verbose:
                print(
                    f"  bootstrap ({n_bootstraps} reps, {n_jobs} workers)...",
                    flush=True,
                )
            inf_result = run_bootstrap(
                panel, fs, eff,
                n_bootstraps=n_bootstraps, seed=seed, n_jobs=n_jobs,
            )
        else:
            if verbose:
                label = "did2s influence functions" if method_name == "ts_did" else "BJS imputation"
                n_cl = len(np.unique(panel.cluster))
                print(f"  analytic SEs ({label}, {n_cl} clusters)...", flush=True)
            inf_result = se_method_fn(panel, fs, eff)

    if verbose:
        if inf_result is not None:
            print(f"  SE(ATT) = {inf_result.att_avg_se:.4f}", flush=True)

    # Step 5: Assemble result
    return _build_result(method_name, panel, fs, eff, inf_result, seed, xformla)


def _build_result(
    method: str,
    panel: PanelData,
    fs: Any,
    eff: EffectsResult,
    inf: InferenceResult | None,
    seed: int | None,
    xformla: list[str] | None = None,
) -> DiDResult:
    """Assemble DiDResult from estimation components."""

    all_h = eff.horizons
    all_est = eff.att_by_horizon
    all_counts = eff.counts

    # Build the single event_study DataFrame with ALL per-period estimates
    es_data: dict[str, Any] = {
        "rel_time": all_h.tolist(),
        "estimate": all_est.tolist(),
        "count": all_counts.tolist(),
    }
    if inf is not None:
        es_data["se"] = inf.horizon_se.tolist()
        es_data["ci_lower"] = inf.horizon_ci_lower.tolist()
        es_data["ci_upper"] = inf.horizon_ci_upper.tolist()
        es_data["pval"] = inf.horizon_pval.tolist()
    else:
        n_h = len(all_h)
        es_data["se"] = [None] * n_h
        es_data["ci_lower"] = [None] * n_h
        es_data["ci_upper"] = [None] * n_h
        es_data["pval"] = [None] * n_h

    event_study = pl.DataFrame(es_data)

    # Compute overall ATT vcov using only post-treatment horizons
    post_mask = all_h >= 0
    vcov_full = inf.vcov if inf else None

    return DiDResult(
        method=method,
        att_avg=eff.att_avg,
        att_avg_se=inf.att_avg_se if inf else None,
        att_avg_ci=inf.att_avg_ci if inf else None,
        att_avg_pval=inf.att_avg_pval if inf else None,
        event_study=event_study,
        unit_fe=fs.unit_fe,
        time_fe=fs.time_fe,
        beta=fs.covar_coefs,
        covariate_names=xformla or None,
        effects=eff.effects,
        y_hat=fs.y_hat,
        vcov=vcov_full,
        boot_dist=inf.boot_dist if inf else None,
        panel=panel,
        sigma2=fs.sigma2,
        seed=seed,
    )
