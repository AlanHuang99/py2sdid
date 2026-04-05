"""
First-stage fixed-effects estimation on untreated observations.

Estimates unit FE + time FE (+ covariates) using only control/not-yet-
treated observations, then predicts Y(0) for the full sample.  This is
the shared first stage for both the two-stage DiD and BJS imputation
estimators.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse.linalg as spla

from .linalg import build_design_matrix
from .results import FirstStageResult, PanelData


def estimate_first_stage(
    panel: PanelData,
    *,
    weighted: bool = True,
) -> FirstStageResult:
    """Run first-stage OLS on the untreated subsample.

    Steps
    -----
    1. Build sparse design matrix ``Z = [unit_FE | time_FE | X]``
    2. Subset to control observations (``D == 0``)
    3. Solve weighted least-squares ``Z_ctrl beta = Y_ctrl``
    4. Predict ``Y_hat(0)`` for all observations
    5. Compute residuals and error variance

    Parameters
    ----------
    panel : PanelData
        Structured panel from ``prepare_panel()``.
    weighted : bool
        Apply observation weights if available.
    """
    # -- Design matrix (full sample) -------------------------------------
    Z_full = build_design_matrix(
        panel.unit_ids,
        panel.n_units,
        panel.time_ids,
        panel.n_periods,
        panel.X,
    )

    # -- Subset to control -----------------------------------------------
    ctrl = panel.is_control
    Z_ctrl = Z_full[ctrl]
    Y_ctrl = panel.Y[ctrl].copy()

    # -- Weights ---------------------------------------------------------
    if weighted and panel.W is not None:
        w_sqrt = np.sqrt(panel.W[ctrl])
        # Weight both sides of the normal equations
        Z_ctrl_w = Z_ctrl.multiply(w_sqrt[:, None])
        Y_ctrl_w = Y_ctrl * w_sqrt
    else:
        Z_ctrl_w = Z_ctrl
        Y_ctrl_w = Y_ctrl

    # -- Solve via LSQR (handles rank deficiency) ------------------------
    beta, *_ = spla.lsqr(
        Z_ctrl_w.astype(np.float64),
        Y_ctrl_w,
        atol=1e-14,
        btol=1e-14,
    )

    # -- Predict full sample ---------------------------------------------
    y_hat = np.asarray(Z_full @ beta).ravel()
    residuals = panel.Y - y_hat

    # -- Extract components from beta ------------------------------------
    n_u = panel.n_units
    n_t = panel.n_periods
    unit_fe = beta[:n_u]
    time_fe = beta[n_u : n_u + n_t]
    covar_coefs = beta[n_u + n_t :] if panel.X is not None else None

    # -- Error variance (on control residuals) ---------------------------
    ctrl_resid = residuals[ctrl]
    dof = max(int(ctrl.sum()) - len(beta), 1)
    sigma2 = float(np.sum(ctrl_resid ** 2) / dof)

    return FirstStageResult(
        y_hat=y_hat,
        residuals=residuals,
        beta=beta,
        unit_fe=unit_fe,
        time_fe=time_fe,
        covar_coefs=covar_coefs,
        sigma2=sigma2,
        design_full=Z_full,
        design_ctrl=Z_ctrl,
    )
