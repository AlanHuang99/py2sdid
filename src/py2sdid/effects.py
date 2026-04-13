"""
Treatment effect computation and aggregation.

Computes individual treatment effects tau_it = Y_it - Y_hat_it(0),
then aggregates into per-period (event-study) estimates for every
relative time period present in the data.
"""

from __future__ import annotations

import numpy as np

from .results import EffectsResult, FirstStageResult, PanelData


def compute_effects(
    panel: PanelData,
    fs: FirstStageResult,
) -> EffectsResult:
    """Compute treatment effects and aggregate by relative time.

    Always computes estimates for ALL relative time periods present in
    the data — both pre-treatment and post-treatment. Pre-treatment
    periods use not-yet-treated observations of eventually-treated
    units. Post-treatment periods use treated observations.

    Parameters
    ----------
    panel : PanelData
        Structured panel data.
    fs : FirstStageResult
        First-stage estimation output.
    """
    # -- Residualized outcome -----------------------------------------------
    y_tilde = panel.Y - fs.y_hat

    # Exclude singletons (unidentifiable FE groups) from ATT computation
    not_singleton = ~panel.is_singleton
    treated_mask = panel.is_treated & not_singleton
    effects_all = y_tilde[treated_mask]

    # Weights for treated obs
    if panel.W is not None:
        w_treated = panel.W[treated_mask]
    else:
        w_treated = np.ones(treated_mask.sum(), dtype=np.float64)

    # -- Overall ATT (post-treatment only) ----------------------------------
    att_avg = float(_weighted_mean(effects_all, w_treated))

    # -- Discover all relative time periods ---------------------------------
    # Post-treatment: from treated obs (D==1, not singleton)
    event_treated = panel.event_time[treated_mask]
    finite_post = event_treated[np.isfinite(event_treated)]
    unique_post = np.unique(finite_post).astype(int)

    # Pre-treatment: from not-yet-treated obs of eventually-treated units
    # (also exclude singletons)
    pre_mask_all = (
        ~panel.is_treated & not_singleton
        & (panel.cohort > 0) & np.isfinite(panel.event_time)
    )
    unique_pre = (
        np.unique(panel.event_time[pre_mask_all]).astype(int)
        if pre_mask_all.any()
        else np.array([], dtype=int)
    )
    unique_pre = unique_pre[unique_pre < 0]

    # Combine and sort all relative time periods
    all_horizons = np.sort(np.concatenate([unique_pre, unique_post]))

    # -- Compute per-period ATT for all relative times ----------------------
    w = panel.W if panel.W is not None else np.ones(panel.n_obs, dtype=np.float64)
    att_arr = np.empty(len(all_horizons), dtype=np.float64)
    count_arr = np.empty(len(all_horizons), dtype=int)

    for i, h in enumerate(all_horizons):
        if h < 0:
            # Pre-treatment: not-yet-treated obs of eventually-treated units
            mask = (
                ~panel.is_treated & not_singleton
                & (panel.cohort > 0)
                & np.isfinite(panel.event_time)
                & (panel.event_time == h)
            )
        else:
            # Post-treatment: treated obs (not singleton)
            mask = treated_mask & (panel.event_time == h)

        count_arr[i] = int(mask.sum())
        if count_arr[i] > 0:
            att_arr[i] = _weighted_mean(y_tilde[mask], w[mask])
        else:
            att_arr[i] = np.nan

    # -- Treatment weight matrix for SE computation -------------------------
    wtr_matrix = _build_weight_matrix(panel, all_horizons)

    return EffectsResult(
        att_avg=att_avg,
        att_by_horizon=att_arr,
        horizons=all_horizons,
        counts=count_arr,
        effects=effects_all,
        pretrend_att=att_arr[all_horizons < 0] if np.any(all_horizons < 0) else None,
        pretrend_horizons=all_horizons[all_horizons < 0] if np.any(all_horizons < 0) else None,
        weights_matrix=wtr_matrix,
    )


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _weighted_mean(vals: np.ndarray, weights: np.ndarray) -> float:
    """Weighted mean, handling zero total weight."""
    total_w = weights.sum()
    if total_w == 0:
        return 0.0
    return float(np.sum(vals * weights) / total_w)


def _build_weight_matrix(
    panel: PanelData,
    horizon_vals: np.ndarray,
) -> np.ndarray:
    """Build treatment-weight matrix for SE computation.

    Each column j is a weight vector of length n_obs. For pre-treatment
    horizons, weights are on not-yet-treated obs of eventually-treated
    units. For post-treatment horizons, weights are on treated obs.
    """
    n = panel.n_obs
    w = panel.W if panel.W is not None else np.ones(n, dtype=np.float64)
    not_singleton = ~panel.is_singleton

    if len(horizon_vals) == 0:
        wtr = np.zeros((n, 1), dtype=np.float64)
        col = np.where(panel.is_treated & not_singleton, w, 0.0)
        total = col.sum()
        if total > 0:
            wtr[:, 0] = col / total
        return wtr

    wtr = np.zeros((n, len(horizon_vals)), dtype=np.float64)
    for j, h in enumerate(horizon_vals):
        if h < 0:
            mask = (
                ~panel.is_treated & not_singleton
                & (panel.cohort > 0)
                & np.isfinite(panel.event_time)
                & (panel.event_time == h)
            )
        else:
            mask = panel.is_treated & not_singleton & (panel.event_time == h)
        col = np.where(mask, w, 0.0)
        total = col.sum()
        if total > 0:
            wtr[:, j] = col / total
    return wtr
