"""
Inference: standard errors, confidence intervals, p-values.

Three methods:
- ``compute_se_did2s``: Gardner (2021) GMM influence-function SEs
- ``compute_se_bjs``: Borusyak, Jaravel, Spiess (2024) imputation SEs
- ``run_bootstrap``: cluster bootstrap (works with either estimator)

The did2s influence-function approach is ported from
``ref/did2s/R/did2s.R`` lines 132-213.

The BJS SE approach is ported from
``ref/didimputation/R/did_imputation.R`` lines 224-268, 380-428
(Equations 6, 8, 10 of the paper).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from scipy import stats

from .linalg import build_design_matrix, robust_solve, sparse_crossprod
from .results import (
    EffectsResult,
    FirstStageResult,
    InferenceResult,
    PanelData,
)


def _robust_factorized(A_sp):
    """Factorize a sparse symmetric matrix for repeated solves.

    Strategy (cascading):
    1. Sparse LU on full matrix (fastest when full-rank).
    2. Drop zero-diagonal columns (singleton FEs), retry sparse LU on
       the reduced system.
    3. Sparse LSQR on the reduced system (handles any remaining rank
       deficiency without dense allocation).

    Returns a callable ``solve(b)`` that computes ``A^{-1} b``
    (or the minimum-norm solution when rank-deficient).
    """
    import scipy.sparse.linalg as _spla
    from scipy.sparse.linalg import factorized as _factorized

    n_full = A_sp.shape[0]

    # Try 1: sparse LU on full matrix
    try:
        return _factorized(A_sp)
    except RuntimeError:
        pass

    # Drop zero-diagonal columns (groups with no control obs)
    diag = np.array(A_sp.diagonal()).ravel()
    live = np.where(diag > 0)[0]

    if len(live) == 0:
        return lambda b: np.zeros(n_full, dtype=np.float64)

    A_reduced = A_sp[np.ix_(live, live)].tocsc()

    # Try 2: sparse LU on reduced matrix
    try:
        _solve_reduced = _factorized(A_reduced)
        def _solve_padded(b):
            b_arr = np.asarray(b).ravel()
            x_reduced = _solve_reduced(b_arr[live])
            x_full = np.zeros(n_full, dtype=np.float64)
            x_full[live] = x_reduced
            return x_full
        return _solve_padded
    except RuntimeError:
        pass

    # Try 3: sparse LSQR on reduced matrix (handles any rank deficiency)
    def _solve_lsqr(b):
        b_arr = np.asarray(b).ravel()
        x_reduced = _spla.lsqr(A_reduced, b_arr[live],
                                atol=1e-10, btol=1e-10)[0]
        x_full = np.zeros(n_full, dtype=np.float64)
        x_full[live] = x_reduced
        return x_full

    return _solve_lsqr


# ===================================================================
# did2s influence-function SEs  (Gardner 2021)
# ===================================================================

def compute_se_did2s(
    panel: PanelData,
    fs: FirstStageResult,
    eff: EffectsResult,
) -> InferenceResult:
    """Analytic SEs via Gardner (2021) influence functions.

    Ports the R did2s SE computation:

    .. math::
        IF_i = (X_2' X_2)^{-1}
               [X_{2i} v_i - X_2' X_1 (X_{10}' X_{10})^{-1} X_{10,i} u_i]

    where X1 = first-stage design, X10 = X1 with treated rows zeroed,
    X2 = second-stage design, v = second-stage residuals,
    u = first-stage residuals.
    """
    n_obs = panel.n_obs
    treated = panel.is_treated
    y_tilde = panel.Y - fs.y_hat

    # -- Weights (must be applied before any matrix computation) ---------
    if panel.W is not None:
        w_sqrt = np.sqrt(panel.W)
    else:
        w_sqrt = np.ones(n_obs, dtype=np.float64)

    # -- Second-stage design matrix X2 -----------------------------------
    X2 = _build_second_stage_design(panel, eff)  # (n_obs, K2)
    X2d = X2.toarray() if sp.issparse(X2) else X2
    X2w = X2d * w_sqrt[:, None]  # weighted X2

    # -- Second-stage weighted OLS (matching R fixest behavior) ----------
    X2tX2 = X2w.T @ X2w
    X2tX2_inv = robust_solve(X2tX2, np.eye(X2tX2.shape[0]))
    y_w = y_tilde * w_sqrt
    beta2 = X2tX2_inv @ (X2w.T @ y_w)

    # -- Residuals (raw, not weighted — matches R residuals()) -----------
    v = y_tilde - X2d @ beta2
    u = fs.residuals.copy()
    u[treated] = 0.0  # zero out treated (as in R did2s)

    # -- Weight remaining matrices ---------------------------------------
    X1 = fs.design_full  # (n_obs, p1) sparse
    X1w = X1.multiply(w_sqrt[:, None]) if sp.issparse(X1) else X1 * w_sqrt[:, None]
    v_w = v * w_sqrt
    u_w = u * w_sqrt

    # -- X10: first-stage design with treated rows zeroed ----------------
    # Build X10w by multiplying X1w by a diagonal mask (0 for treated,
    # 1 for control).  This avoids the catastrophically slow LIL row
    # zeroing which dominates runtime at scale.
    ctrl_mask = (~treated).astype(np.float64)
    if sp.issparse(X1w):
        X10w = X1w.multiply(ctrl_mask[:, None]).tocsc()
    else:
        X10w = X1w * ctrl_mask[:, None]

    # -- Influence functions (K2 x n_obs) --------------------------------
    # IF_ss = X2tX2_inv @ X2w.T @ diag(v_w)
    # Each column i: X2tX2_inv @ X2w[i] * v_w[i]
    IF_ss = X2tX2_inv @ (X2w * v_w[:, None]).T  # (K2, n_obs)

    # gamma_hat = solve(X10w'X10w, X1w' @ X2w)
    # Key optimization: X10'X10 is very sparse (fill ~0.2% for large N).
    # Use sparse LU factorization instead of dense Cholesky — this is
    # O(nnz^1.5) instead of O(N^3), giving 100-200x speedup at scale.
    # Falls back to pseudo-inverse for rank-deficient systems (RCS data).
    X10tX10_sp = (X10w.T @ X10w).tocsc()
    X1tX2 = X1w.T @ X2w
    if sp.issparse(X1tX2):
        X1tX2 = X1tX2.toarray() if hasattr(X1tX2, 'toarray') else np.asarray(X1tX2)

    _solve = _robust_factorized(X10tX10_sp)
    gamma_hat = np.column_stack([_solve(X1tX2[:, j]) for j in range(X1tX2.shape[1])])

    # IF_fs: first-stage contribution via sparse matmul
    X10w_u = X10w.multiply(u_w[:, None]) if sp.issparse(X10w) else X10w * u_w[:, None]
    temp = np.asarray(X10w_u @ gamma_hat)  # sparse @ dense → dense
    IF_fs = X2tX2_inv @ temp.T  # (K2, K2) @ (K2, n_obs) = (K2, n_obs)

    IF = IF_fs - IF_ss  # (K2, n_obs)

    # -- Cluster the influence functions ---------------------------------
    vcov = _cluster_vcov(IF, panel.cluster)

    # -- Overall ATT SE using a single static design column ---------------
    # Mirrors did2s second-stage with a single treatment indicator D_it.
    # Reuses the sparse factorization from above.
    X2_static = treated.astype(np.float64).reshape(-1, 1)
    X2s = X2_static * w_sqrt[:, None]  # weighted
    X2stX2s = X2s.T @ X2s
    X2stX2s_inv = robust_solve(X2stX2s, np.eye(1))
    beta2_static = X2stX2s_inv @ (X2s.T @ y_w)  # weighted OLS
    v_static = y_tilde - X2_static @ beta2_static  # raw residuals

    IF_ss_static = X2stX2s_inv @ (X2s * (v_static * w_sqrt)[:, None]).T
    X1tX2_static = X1w.T @ X2s
    if sp.issparse(X1tX2_static):
        X1tX2_static = np.asarray(X1tX2_static.toarray()).ravel()
    else:
        X1tX2_static = np.asarray(X1tX2_static).ravel()
    gamma_hat_static = _solve(X1tX2_static).reshape(-1, 1)  # reuse factorization
    temp_static = np.asarray(X10w_u @ gamma_hat_static)
    IF_fs_static = X2stX2s_inv @ temp_static.T
    IF_static = IF_fs_static - IF_ss_static
    vcov_static = _cluster_vcov(IF_static, panel.cluster)
    att_avg_se = float(np.sqrt(vcov_static[0, 0]))

    # -- Extract SEs, CIs, p-values --------------------------------------
    return _build_inference(beta2, vcov, eff, att_avg_se_override=att_avg_se)


# ===================================================================
# BJS imputation SEs  (Borusyak, Jaravel, Spiess 2021)
# ===================================================================

def compute_se_bjs(
    panel: PanelData,
    fs: FirstStageResult,
    eff: EffectsResult,
) -> InferenceResult:
    """Analytic SEs via BJS (2024) imputation formula.

    Ports R didimputation SE computation (Equations 6, 8, 10).

    .. math::
        v^*_{it} = -Z (Z_0' Z_0)^{-1} Z_1' w_1   \\quad [\\text{Eq 6}]

        \\text{Var} = \\sum_c \\left[\\sum_{i \\in c} v^*_{it} (adj_{it} - \\bar\\tau_{et})\\right]^2
        \\quad [\\text{Eq 8}]
    """
    n_obs = panel.n_obs
    treated = panel.is_treated
    ctrl = panel.is_control

    # Observation weights
    w = panel.W if panel.W is not None else np.ones(n_obs, dtype=np.float64)

    # Full design matrix weighted by observation weights
    Z = fs.design_full
    if sp.issparse(Z):
        Z = Z.tocsc()  # ensure indexable format
        Z_w = Z.multiply(w[:, None])
    else:
        Z_w = Z * w[:, None]

    # Treatment weight matrix (n_obs, n_wtr)
    wtr_mat = eff.weights_matrix  # from effects.py
    n_wtr = wtr_mat.shape[1]

    # Split into treated/control rows
    if sp.issparse(Z_w):
        Z_w = Z_w.tocsr()
    Z1 = Z_w[treated]  # treated rows
    Z0 = Z_w[ctrl]     # control rows

    # wtr for treated rows only
    wtr_treated = wtr_mat[treated]

    # Eq 6: v* = -Z @ solve(Z0'Z0, Z1' @ wtr_treated)
    # Use sparse factorized solve — Z0'Z0 is very sparse (same structure
    # as X10'X10 in did2s) and this avoids the O(N^3) dense Cholesky.
    # Falls back to pseudo-inverse for rank-deficient systems (RCS data).
    Z0tZ0_sp = (Z0.T @ Z0).tocsc()
    Z1t_wtr = Z1.T @ wtr_treated  # (p, n_wtr)
    if sp.issparse(Z1t_wtr):
        Z1t_wtr = Z1t_wtr.toarray() if hasattr(Z1t_wtr, 'toarray') else np.asarray(Z1t_wtr)

    _solve_bjs = _robust_factorized(Z0tZ0_sp)
    solved = np.column_stack([_solve_bjs(Z1t_wtr[:, j]) for j in range(Z1t_wtr.shape[1])])

    if sp.issparse(Z_w):
        v_star = -1.0 * np.asarray((Z_w @ solved))  # (n_obs, n_wtr)
    else:
        v_star = -1.0 * (Z_w @ solved)

    # Fix: v*[treated] = wtr[treated]
    v_star[treated] = wtr_mat[treated]

    # Residualized outcome
    adj = panel.Y - fs.y_hat

    # Eq 10: compute tau_bar_et per (cohort, event_time) group
    # Then recenter: adj_recentered = adj - tau_bar_et
    adj_recentered = _recenter_adj(
        adj, v_star, panel.cohort, panel.event_time, treated,
    )  # (n_obs, n_wtr)

    # Eq 8: clustered variance
    ses = np.empty(n_wtr, dtype=np.float64)
    for j in range(n_wtr):
        ses[j] = _cluster_se_bjs(
            v_star[:, j], adj_recentered[:, j], panel.cluster,
        )

    # -- Overall ATT SE using a single static weight vector ---------------
    # This matches R didimputation's static SE: one weight column where
    # every treated obs gets w_it / sum(w_treated).
    static_wtr = np.zeros((n_obs, 1), dtype=np.float64)
    static_col = np.where(treated, w, 0.0)
    static_total = static_col.sum()
    if static_total > 0:
        static_wtr[:, 0] = static_col / static_total

    # Run BJS SE formula on the static weight vector
    Z1t_static = Z1.T @ static_wtr[treated]
    if sp.issparse(Z1t_static):
        Z1t_static = Z1t_static.toarray() if hasattr(Z1t_static, 'toarray') else np.asarray(Z1t_static)
    solved_static = _solve_bjs(Z1t_static.ravel()).reshape(-1, 1)  # reuse factorization
    if sp.issparse(Z_w):
        v_star_static = -1.0 * np.asarray(Z_w @ solved_static)
    else:
        v_star_static = -1.0 * (Z_w @ solved_static)
    v_star_static[treated] = static_wtr[treated]

    adj_recentered_static = _recenter_adj(
        adj, v_star_static, panel.cohort, panel.event_time, treated,
    )
    att_avg_se = _cluster_se_bjs(
        v_star_static[:, 0], adj_recentered_static[:, 0], panel.cluster,
    )

    # For pre-treatment periods, the BJS imputation formula is not
    # applicable (it is defined only for treated observations).
    # R's didimputation computes pretrend SEs via a separate OLS
    # regression of Y on event-time indicators + FEs using only
    # control observations, with clustered SEs.  We replicate this.
    if eff.horizons is not None and len(eff.horizons) > 0:
        pre_mask = eff.horizons < 0
        if np.any(pre_mask):
            pre_ses = _pretrend_se_ols(panel, fs, eff)
            for j in np.where(pre_mask)[0]:
                ses[j] = pre_ses[j]

    # Build point estimates from effects
    if eff.horizons is not None and len(eff.horizons) > 0:
        beta = eff.att_by_horizon
    else:
        beta = np.array([eff.att_avg])

    vcov = np.diag(ses ** 2)
    return _build_inference(beta, vcov, eff, att_avg_se_override=att_avg_se)


# ===================================================================
# Bootstrap
# ===================================================================

def run_bootstrap(
    panel: PanelData,
    fs: FirstStageResult,
    eff: EffectsResult,
    *,
    n_bootstraps: int = 500,
    seed: int | None = None,
    n_jobs: int = 1,
) -> InferenceResult:
    """Cluster bootstrap standard errors.

    Resamples clusters with replacement, re-runs the full two-stage
    procedure, collects ATT estimates, computes SE from the bootstrap
    distribution.
    """
    from .first_stage import estimate_first_stage
    from .effects import compute_effects
    from .panel import prepare_panel

    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(panel.cluster)
    n_clusters = len(unique_clusters)

    # Pre-generate bootstrap cluster samples
    boot_samples = [
        rng.choice(unique_clusters, size=n_clusters, replace=True)
        for _ in range(n_bootstraps)
    ]

    def _boot_rep(cluster_sample):
        """Single bootstrap replicate."""
        # Build resampled index
        indices = []
        for c in cluster_sample:
            indices.append(np.where(panel.cluster == c)[0])
        idx = np.concatenate(indices)

        # Rebuild arrays for a "fake" PanelData
        boot_panel = PanelData(
            Y=panel.Y[idx],
            D=panel.D[idx],
            unit_ids=panel.unit_ids[idx],
            time_ids=panel.time_ids[idx],
            cohort=panel.cohort[idx],
            event_time=panel.event_time[idx],
            X=panel.X[idx] if panel.X is not None else None,
            W=panel.W[idx] if panel.W is not None else None,
            cluster=panel.cluster[idx],
            n_units=panel.n_units,
            n_periods=panel.n_periods,
            n_treated=int(panel.D[idx].sum()),
            n_control=int((panel.D[idx] == 0).sum()),
            n_obs=len(idx),
            cohort_sizes=panel.cohort_sizes,
            is_treated=panel.D[idx] == 1,
            is_control=panel.D[idx] == 0,
            unit_map=panel.unit_map,
            time_map=panel.time_map,
            cluster_map=panel.cluster_map,
        )

        try:
            boot_fs = estimate_first_stage(boot_panel)
            boot_eff = compute_effects(boot_panel, boot_fs)
            if boot_eff.horizons is not None and len(boot_eff.horizons) > 0:
                return boot_eff.att_by_horizon
            return np.array([boot_eff.att_avg])
        except Exception:
            return None

    # Run bootstrap (parallel or serial)
    if n_jobs == 1:
        results = [_boot_rep(s) for s in boot_samples]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_boot_rep)(s) for s in boot_samples
        )

    # Collect valid results
    valid = [r for r in results if r is not None]
    if not valid:
        raise RuntimeError("All bootstrap replications failed")

    boot_matrix = np.vstack(valid)  # (n_valid, K)

    # Point estimates
    if eff.horizons is not None and len(eff.horizons) > 0:
        theta_hat = eff.att_by_horizon
    else:
        theta_hat = np.array([eff.att_avg])

    # SE from bootstrap distribution
    boot_se = np.std(boot_matrix, axis=0, ddof=1)

    # Pivotal CI: 2*theta - quantile(1-alpha/2)
    alpha = 0.05
    q_lo = np.percentile(boot_matrix, 100 * (alpha / 2), axis=0)
    q_hi = np.percentile(boot_matrix, 100 * (1 - alpha / 2), axis=0)
    ci_lower = 2 * theta_hat - q_hi
    ci_upper = 2 * theta_hat - q_lo

    # p-values from bootstrap distribution (fraction of boot dist on opposite side of 0)
    pvals = np.array([
        float(2 * min(np.mean(boot_matrix[:, j] <= 0),
                       np.mean(boot_matrix[:, j] >= 0)))
        for j in range(boot_matrix.shape[1])
    ])

    vcov = np.cov(boot_matrix, rowvar=False)
    if vcov.ndim == 0:
        vcov = vcov.reshape(1, 1)

    return InferenceResult(
        att_avg_se=float(boot_se[0]) if len(boot_se) == 1 else float(
            np.sqrt(np.sum(vcov) / vcov.shape[0] ** 2)  # avg SE
        ),
        att_avg_ci=(float(ci_lower[0]), float(ci_upper[0])) if len(ci_lower) == 1 else (
            float(2 * eff.att_avg - np.percentile(boot_matrix.mean(axis=1), 97.5)),
            float(2 * eff.att_avg - np.percentile(boot_matrix.mean(axis=1), 2.5)),
        ),
        att_avg_pval=float(pvals[0]) if len(pvals) == 1 else float(
            2 * min(np.mean(boot_matrix.mean(axis=1) <= 0),
                    np.mean(boot_matrix.mean(axis=1) >= 0))
        ),
        horizon_se=boot_se,
        horizon_ci_lower=ci_lower,
        horizon_ci_upper=ci_upper,
        horizon_pval=pvals,
        vcov=vcov,
        boot_dist=boot_matrix,
    )


# ===================================================================
# Internal helpers
# ===================================================================

def _pretrend_se_ols(
    panel: PanelData,
    fs: FirstStageResult,
    eff: EffectsResult,
) -> np.ndarray:
    """Compute pre-treatment SEs via OLS on the control subsample.

    Replicates R didimputation's approach: regress Y on event-time
    indicators + unit/time FEs using only control observations
    (zz000treat == 0), with clustered standard errors.

    Returns array of SEs aligned with eff.horizons (pre-treatment
    entries filled, post-treatment entries set to 0).
    """
    n_h = len(eff.horizons)
    pre_ses = np.zeros(n_h, dtype=np.float64)
    pre_idx = np.where(eff.horizons < 0)[0]
    if len(pre_idx) == 0:
        return pre_ses

    # Control subsample
    ctrl = panel.is_control
    Y_ctrl = panel.Y[ctrl]
    n_ctrl = int(ctrl.sum())

    # Build design: unit FE + time FE + event-time indicators for pre-periods
    # Event-time indicators: for each pre-horizon h, indicator for
    # eventually-treated units at event_time == h
    pre_horizons = eff.horizons[pre_idx]
    n_pre = len(pre_horizons)

    # Event-time indicators on control subsample
    event_time_ctrl = panel.event_time[ctrl]
    cohort_ctrl = panel.cohort[ctrl]
    X_pre = np.zeros((n_ctrl, n_pre), dtype=np.float64)
    for j, h in enumerate(pre_horizons):
        X_pre[:, j] = (
            (cohort_ctrl > 0)
            & np.isfinite(event_time_ctrl)
            & (event_time_ctrl == h)
        ).astype(np.float64)

    # Full design: FE design (control rows) + event-time indicators
    Z_ctrl = fs.design_ctrl  # sparse (n_ctrl, p)
    if sp.issparse(Z_ctrl):
        X_full = sp.hstack([Z_ctrl, sp.csc_matrix(X_pre)]).tocsc()
    else:
        X_full = np.hstack([Z_ctrl, X_pre])

    # Weighted OLS
    if panel.W is not None:
        w_ctrl = panel.W[ctrl]
        w_sqrt = np.sqrt(w_ctrl)
    else:
        w_sqrt = np.ones(n_ctrl, dtype=np.float64)

    if sp.issparse(X_full):
        Xw = X_full.multiply(w_sqrt[:, None])
    else:
        Xw = X_full * w_sqrt[:, None]
    Yw = Y_ctrl * w_sqrt

    # Solve for all coefficients (FE + event-time indicators)
    from scipy.sparse.linalg import lsqr
    if sp.issparse(Xw):
        beta_all = lsqr(Xw, Yw)[0]
    else:
        beta_all, _, _, _ = np.linalg.lstsq(Xw, Yw, rcond=None)

    # Event-time coefficients are the last n_pre entries
    beta_pre = beta_all[-n_pre:]

    # Residuals
    if sp.issparse(X_full):
        resid = Y_ctrl - np.asarray(X_full @ beta_all).ravel()
    else:
        resid = Y_ctrl - X_full @ beta_all

    # Clustered SEs for the event-time coefficients
    # IF_i = (Xw'Xw)^{-1} Xw_i * resid_w_i  for the pre columns only
    resid_w = resid * w_sqrt
    if sp.issparse(Xw):
        Xw_csc = Xw.tocsc()
        Xw_pre = np.asarray(Xw_csc[:, -n_pre:].toarray())
    else:
        Xw_pre = Xw[:, -n_pre:]

    XwTXw_pre = Xw_pre.T @ Xw_pre
    XwTXw_pre_inv = robust_solve(XwTXw_pre, np.eye(n_pre))

    # Per-obs influence: (n_pre, n_ctrl)
    IF_pre = XwTXw_pre_inv @ (Xw_pre * resid_w[:, None]).T

    # Cluster
    cluster_ctrl = panel.cluster[ctrl]
    vcov_pre = _cluster_vcov(IF_pre, cluster_ctrl)
    se_pre = np.sqrt(np.diag(vcov_pre))

    # Place into output array
    for j_local, j_global in enumerate(pre_idx):
        pre_ses[j_global] = se_pre[j_local]

    return pre_ses


def _build_second_stage_design(
    panel: PanelData,
    eff: EffectsResult,
) -> np.ndarray:
    """Build the second-stage design matrix X2.

    For static: single column ``D_it``.
    For event-study: one indicator column per horizon (pre + post).
    Pre-treatment columns use not-yet-treated obs of eventually-treated units.
    Post-treatment columns use treated obs.
    """
    n = panel.n_obs
    if eff.horizons is not None and len(eff.horizons) > 0:
        K = len(eff.horizons)
        X2 = np.zeros((n, K), dtype=np.float64)
        for j, h in enumerate(eff.horizons):
            if h < 0:
                # Pre-treatment: not-yet-treated obs of eventually-treated units
                X2[:, j] = (
                    ~panel.is_treated
                    & (panel.cohort > 0)
                    & np.isfinite(panel.event_time)
                    & (panel.event_time == h)
                ).astype(np.float64)
            else:
                # Post-treatment: treated obs
                X2[:, j] = (
                    panel.is_treated & (panel.event_time == h)
                ).astype(np.float64)
        return X2
    # Static: single treatment column
    return panel.is_treated.astype(np.float64).reshape(-1, 1)


def _cluster_vcov(IF: np.ndarray, cluster: np.ndarray) -> np.ndarray:
    """Compute clustered variance-covariance from influence functions.

    Parameters
    ----------
    IF : (K, n_obs) array
        Per-observation influence functions.
    cluster : (n_obs,) array
        Cluster assignments.

    Returns
    -------
    (K, K) array
    """
    K = IF.shape[0]
    unique_clusters = np.unique(cluster)
    n_cl = len(unique_clusters)

    # Vectorized: sum IF within each cluster using bincount
    # cluster_sums[k, c] = sum of IF[k, i] for i in cluster c
    cluster_sums = np.zeros((K, n_cl), dtype=np.float64)
    # Map cluster IDs to 0..n_cl-1 (vectorized via searchsorted)
    cl_idx = np.searchsorted(unique_clusters, cluster)

    for k in range(K):
        cluster_sums[k] = np.bincount(cl_idx, weights=IF[k], minlength=n_cl)

    # vcov = cluster_sums @ cluster_sums.T
    return cluster_sums @ cluster_sums.T


def _recenter_adj(
    adj: np.ndarray,
    v_star: np.ndarray,
    cohort: np.ndarray,
    event_time: np.ndarray,
    treated: np.ndarray,
) -> np.ndarray:
    """Eq 10: compute tau_bar_et and recenter adj.

    For each (cohort g, event_time e) group of treated observations,
    compute weighted group-mean treatment effect and subtract it.

    Returns (n_obs, n_wtr) recentered adj matrix.
    """
    n_obs, n_wtr = v_star.shape
    adj_recentered = np.tile(adj[:, None], (1, n_wtr))  # (n_obs, n_wtr)

    # Unique (cohort, event_time) pairs among treated
    treated_idx = np.where(treated)[0]
    if len(treated_idx) == 0:
        return adj_recentered

    pairs = np.column_stack([cohort[treated_idx], event_time[treated_idx]])
    unique_pairs = np.unique(pairs, axis=0)

    for g_val, e_val in unique_pairs:
        if not np.isfinite(e_val):
            continue
        group_mask = treated & (cohort == g_val) & (event_time == e_val)
        if group_mask.sum() == 0:
            continue

        for j in range(n_wtr):
            v_grp = v_star[group_mask, j]
            adj_grp = adj[group_mask]
            v2_sum = np.sum(v_grp ** 2)
            if v2_sum > 0:
                tau_bar = np.sum(v_grp ** 2 * adj_grp) / v2_sum
            else:
                tau_bar = 0.0
            adj_recentered[group_mask, j] -= tau_bar

    return adj_recentered


def _cluster_se_bjs(
    v_star: np.ndarray,
    adj_recentered: np.ndarray,
    cluster: np.ndarray,
) -> float:
    """Eq 8: clustered SE for a single treatment weight.

    .. math::
        \\text{Var} = \\sum_c \\left[\\sum_{i \\in c} v^*_i \\cdot adj_i\\right]^2
    """
    var = 0.0
    for c in np.unique(cluster):
        mask = cluster == c
        score_c = np.sum(v_star[mask] * adj_recentered[mask])
        var += score_c ** 2
    return float(np.sqrt(var))


def _build_inference(
    beta: np.ndarray,
    vcov: np.ndarray,
    eff: EffectsResult,
    att_avg_se_override: float | None = None,
) -> InferenceResult:
    """Build InferenceResult from point estimates and vcov.

    Parameters
    ----------
    att_avg_se_override : float, optional
        If provided, use this as the overall ATT SE instead of
        deriving it from the event-study vcov. This is needed because
        the overall ATT SE should be computed from a single static
        weight vector, not aggregated from per-period estimates.
    """
    se = np.sqrt(np.diag(vcov))
    alpha = 0.05
    z = stats.norm.ppf(1 - alpha / 2)

    ci_lower = beta - z * se
    ci_upper = beta + z * se
    pval = 2 * stats.norm.sf(np.abs(beta / np.maximum(se, 1e-300)))

    # Overall ATT inference
    att_avg = eff.att_avg
    if att_avg_se_override is not None:
        att_se = att_avg_se_override
    elif len(beta) == 1:
        att_se = float(se[0])
    else:
        # Fallback: aggregate from event-study vcov using count weights
        post_mask = eff.horizons >= 0 if eff.horizons is not None and len(eff.horizons) > 0 else np.ones(len(beta), dtype=bool)
        post_idx = np.where(post_mask)[0]
        if len(post_idx) == 0:
            post_idx = np.arange(len(beta))
        post_counts = eff.counts[post_idx] if eff.counts is not None and len(eff.counts) > 0 else np.ones(len(post_idx))
        w_full = np.zeros(len(beta))
        w_full[post_idx] = post_counts / post_counts.sum()
        att_se = float(np.sqrt(w_full @ vcov @ w_full))

    att_ci = (att_avg - z * att_se, att_avg + z * att_se)
    att_pval = float(2 * stats.norm.sf(abs(att_avg) / max(att_se, 1e-300)))

    return InferenceResult(
        att_avg_se=att_se,
        att_avg_ci=att_ci,
        att_avg_pval=att_pval,
        horizon_se=se,
        horizon_ci_lower=ci_lower,
        horizon_ci_upper=ci_upper,
        horizon_pval=pval,
        vcov=vcov,
        boot_dist=None,
    )
