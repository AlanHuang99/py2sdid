"""
Matplotlib plotting for py2sdid results.

Six plot types via ``plot(result, kind=...)``, all returning
``matplotlib.figure.Figure``.

Style follows pyfector conventions: blue for estimates, shaded CIs,
red dashed treatment markers, clean grid, tight layout.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# -------------------------------------------------------------------
# Dispatch
# -------------------------------------------------------------------

def plot(
    result: Any,  # DiDResult
    kind: str = "event_study",
    *,
    figsize: tuple[float, float] = (10, 6),
    title: str | None = None,
    units: list | None = None,
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """Plot estimation results.

    Parameters
    ----------
    result : DiDResult
        Output from ``ts_did()`` or ``bjs_did()``.
    kind : str
        Plot type: ``"event_study"``, ``"pretrends"``,
        ``"treatment_status"``, ``"counterfactual"``,
        ``"honestdid"``, ``"calendar"``.
    figsize : tuple
        Figure size in inches.
    title : str, optional
        Override the default title.
    units : list, optional
        Unit IDs for counterfactual plot.
    ax : matplotlib Axes, optional
        Existing axes to draw on.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    _PLOT_REGISTRY[kind](result, ax, title=title, units=units, **kwargs)

    fig.tight_layout()
    return fig


# -------------------------------------------------------------------
# Plot implementations
# -------------------------------------------------------------------

def _plot_event_study(result: Any, ax: Any, *, title: str | None = None, **kw: Any) -> None:
    """Event study with ALL periods (pre + post) and CI bands."""
    df = result.event_study  # all relative time periods
    h = df["rel_time"].to_numpy()
    est = df["estimate"].to_numpy()

    has_se = "se" in df.columns and df["se"][0] is not None
    has_count = "count" in df.columns

    # Main line
    ax.plot(h, est, "o-", color="#1f77b4", linewidth=1.5, markersize=4, label="ATT", zorder=3)

    # CI band
    if has_se:
        ci_lo = df["ci_lower"].to_numpy().astype(float)
        ci_hi = df["ci_upper"].to_numpy().astype(float)
        ax.fill_between(h, ci_lo, ci_hi, alpha=0.15, color="#1f77b4", label="95% CI")

    # Reference lines
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, zorder=1)
    ax.axvline(-0.5, color="#d62728", linestyle="--", linewidth=1.2, label="Treatment onset", zorder=2)

    # Count bars (secondary axis)
    if has_count:
        counts = df["count"].to_numpy()
        ax2 = ax.twinx()
        ax2.bar(h, counts, alpha=0.12, color="gray", width=0.8, zorder=0)
        ax2.set_ylabel("Observations", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")

    ax.set_xlabel("Relative time")
    ax.set_ylabel("Estimate")
    ax.set_title(title or f"Event Study ({result.method})")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)


def _plot_pretrends(result: Any, ax: Any, *, title: str | None = None, **kw: Any) -> None:
    """Pre-treatment coefficients with equivalence bands."""
    pre = result.pretrend_tests
    if pre is None or len(pre) == 0:
        ax.text(0.5, 0.5, "No pre-trend estimates available",
                transform=ax.transAxes, ha="center")
        return

    h = pre["rel_time"].to_numpy()
    est = pre["estimate"].to_numpy()

    ax.plot(h, est, "o-", color="#1f77b4", linewidth=2, markersize=6)

    if "se" in pre.columns and pre["se"][0] is not None:
        se = pre["se"].to_numpy().astype(float)
        ax.errorbar(h, est, yerr=1.96 * se, fmt="none", color="#1f77b4", capsize=3)

    # Equivalence bands (if sigma2 available)
    delta = 0.36 * np.sqrt(max(result.sigma2, 1e-10))
    ax.axhspan(-delta, delta, alpha=0.15, color="green", label=f"Equiv. band (±{delta:.3f})")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Event time (pre-treatment)")
    ax.set_ylabel("Coefficient")
    ax.set_title(title or "Pre-trend Coefficients")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_treatment_status(result: Any, ax: Any, *, title: str | None = None, **kw: Any) -> None:
    """Treatment status heatmap across units/groups and time.

    For panel data, rows are individual units.  For RCS data, rows are
    groups (using ``fe_ids`` / ``n_fe_levels``), which avoids a sparse
    24,000-row matrix when individuals each appear only once.
    """
    panel = result.panel

    # Use fe_ids/n_fe_levels: equals unit_ids/n_units for panels,
    # group_ids/n_groups for RCS — one code path for all three modes.
    row_ids = panel.fe_ids
    n_rows = panel.n_fe_levels

    # Build status matrix (n_rows x n_periods)
    # For individual RCS, multiple individuals map to the same group row.
    # We take the max status per cell (treated > not-yet-treated > never).
    status = np.zeros((n_rows, panel.n_periods), dtype=np.float64)
    for obs_idx in range(panel.n_obs):
        r = row_ids[obs_idx]
        t = panel.time_ids[obs_idx]
        if panel.D[obs_idx] == 1:
            status[r, t] = 1.0  # treated
        elif panel.cohort[obs_idx] > 0:
            status[r, t] = max(status[r, t], 0.5)  # not-yet-treated
        # never-treated stays 0.0

    # Sort: never-treated first, then by cohort timing
    row_cohort = np.zeros(n_rows, dtype=np.float64)
    for obs_idx in range(panel.n_obs):
        r = row_ids[obs_idx]
        c = panel.cohort[obs_idx]
        if c > 0:
            row_cohort[r] = c
    sort_order = np.argsort(row_cohort)
    status = status[sort_order]

    import matplotlib.colors as mcolors
    cmap = mcolors.ListedColormap(["#d4e6f1", "#f9e79f", "#e74c3c"])
    bounds = [-0.25, 0.25, 0.75, 1.25]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    time_labels = [panel.time_map.get(t, t) for t in range(panel.n_periods)]
    ax.imshow(status, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xlabel("Time period")
    ylabel = "Group (sorted by cohort)" if panel.is_rcs else "Unit (sorted by cohort)"
    ax.set_ylabel(ylabel)
    ax.set_title(title or "Treatment Status")

    # Tick labels (subsample if many)
    n_t = panel.n_periods
    if n_t <= 20:
        ax.set_xticks(range(n_t))
        ax.set_xticklabels(time_labels, rotation=45, fontsize=7)


def _plot_counterfactual(
    result: Any, ax: Any, *, units: list | None = None, title: str | None = None, **kw: Any,
) -> None:
    """Observed vs counterfactual Y(0) for selected units or groups.

    For panel data, shows individual unit trajectories.  For RCS data,
    aggregates to group-period means (since individuals appear only
    once and have no time series).
    """
    panel = result.panel
    is_individual_rcs = panel.is_rcs and panel.n_units != panel.n_fe_levels

    if is_individual_rcs:
        _plot_counterfactual_rcs(result, ax, units=units, title=title)
    else:
        _plot_counterfactual_panel(result, ax, units=units, title=title)


def _plot_counterfactual_panel(
    result: Any, ax: Any, *, units: list | None = None, title: str | None = None,
) -> None:
    """Counterfactual for panel or aggregated RCS (one row per unit-period)."""
    panel = result.panel
    entity = "Group" if panel.is_rcs else "Unit"

    if units is None:
        treated_units = np.unique(panel.unit_ids[panel.is_treated])[:3]
        units = [panel.unit_map.get(u, u) for u in treated_units]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, uid in enumerate(units):
        code = None
        for k, v in panel.unit_map.items():
            if v == uid:
                code = k
                break
        if code is None:
            continue

        mask = panel.unit_ids == code
        t_idx = panel.time_ids[mask]
        t_vals = np.array([panel.time_map.get(t, t) for t in t_idx])
        y_actual = panel.Y[mask]
        y_cf = result.y_hat[mask]

        color = colors[i % len(colors)]
        ax.plot(t_vals, y_actual, "-", color=color, linewidth=1.5,
                label=f"{entity} {uid} (actual)")
        ax.plot(t_vals, y_cf, "--", color=color, linewidth=1.5, alpha=0.7,
                label=f"{entity} {uid} (Y(0))")

        cohort_val = panel.cohort[mask][0]
        if cohort_val > 0:
            ax.axvline(cohort_val, color=color, linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Time")
    ax.set_ylabel("Outcome")
    ax.set_title(title or f"Counterfactual Comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_counterfactual_rcs(
    result: Any, ax: Any, *, units: list | None = None, title: str | None = None,
) -> None:
    """Counterfactual for individual-level RCS.

    Aggregates Y and Y_hat to group-period means, then plots group
    trajectories (actual vs counterfactual).
    """
    panel = result.panel
    fe_map = panel.fe_map if panel.fe_map is not None else panel.unit_map

    if units is None:
        # Pick first 3 treated groups
        treated_groups = np.unique(panel.fe_ids[panel.is_treated])[:3]
        units = [fe_map.get(g, g) for g in treated_groups]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, gid in enumerate(units):
        # Find integer code for this group
        code = None
        for k, v in fe_map.items():
            if v == gid:
                code = k
                break
        if code is None:
            continue

        group_mask = panel.fe_ids == code
        group_time = panel.time_ids[group_mask]
        group_y = panel.Y[group_mask]
        group_yhat = result.y_hat[group_mask]

        # Aggregate to period means within this group
        unique_t = np.unique(group_time)
        t_vals = np.array([panel.time_map.get(t, t) for t in unique_t])
        y_mean = np.array([group_y[group_time == t].mean() for t in unique_t])
        yhat_mean = np.array([group_yhat[group_time == t].mean() for t in unique_t])

        color = colors[i % len(colors)]
        ax.plot(t_vals, y_mean, "-", color=color, linewidth=1.5,
                label=f"Group {gid} (actual)")
        ax.plot(t_vals, yhat_mean, "--", color=color, linewidth=1.5, alpha=0.7,
                label=f"Group {gid} (Y(0))")

        # Treatment onset for this group
        cohort_val = panel.cohort[group_mask][0]
        if cohort_val > 0:
            ax.axvline(cohort_val, color=color, linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Time")
    ax.set_ylabel("Outcome (group mean)")
    ax.set_title(title or "Counterfactual Comparison (group means)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_honestdid(result: Any, ax: Any, *, title: str | None = None, **kw: Any) -> None:
    """HonestDiD sensitivity — robust CIs as function of M."""
    # Need diagnostics to have been run
    diag = result.diagnose()
    if diag.honestdid_results is None:
        ax.text(0.5, 0.5, "Run diagnose() first", transform=ax.transAxes, ha="center")
        return

    hd = diag.honestdid_results
    M = hd["M"].to_numpy()
    ci_lo = hd["ci_lower"].to_numpy()
    ci_hi = hd["ci_upper"].to_numpy()

    ax.fill_between(M, ci_lo, ci_hi, alpha=0.3, color="#1f77b4", label="Robust CI")
    ax.plot(M, ci_lo, "-", color="#1f77b4", linewidth=1)
    ax.plot(M, ci_hi, "-", color="#1f77b4", linewidth=1)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    ax.set_xlabel("M (smoothness bound)")
    ax.set_ylabel("ATT")
    ax.set_title(title or "HonestDiD Sensitivity Analysis")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_calendar(result: Any, ax: Any, *, title: str | None = None, **kw: Any) -> None:
    """ATT by calendar time period."""
    panel = result.panel

    # Use first-stage prediction (includes covariates) for treatment effects
    # Exclude singletons (unidentifiable counterfactuals)
    treated = panel.is_treated & ~panel.is_singleton
    time_vals = panel.time_ids[treated]
    effects = (panel.Y - result.y_hat)[treated]

    unique_times = np.unique(time_vals)
    cal_att = np.empty(len(unique_times))
    for i, t in enumerate(unique_times):
        mask = time_vals == t
        cal_att[i] = np.mean(effects[mask])

    t_labels = [panel.time_map.get(t, t) for t in unique_times]

    ax.bar(t_labels, cal_att, color="#1f77b4", alpha=0.7, width=0.8)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Calendar time")
    ax.set_ylabel("ATT")
    ax.set_title(title or "ATT by Calendar Time")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3, axis="y")


# -------------------------------------------------------------------
# Registry
# -------------------------------------------------------------------

_PLOT_REGISTRY: dict[str, Any] = {
    "event_study": _plot_event_study,
    "pretrends": _plot_pretrends,
    "treatment_status": _plot_treatment_status,
    "counterfactual": _plot_counterfactual,
    "honestdid": _plot_honestdid,
    "calendar": _plot_calendar,
}
