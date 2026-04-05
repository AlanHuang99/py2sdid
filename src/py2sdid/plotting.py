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
    """Treatment status heatmap across units and time."""
    panel = result.panel

    # Build status matrix (n_units x n_periods)
    status = np.zeros((panel.n_units, panel.n_periods), dtype=np.float64)
    for obs_idx in range(panel.n_obs):
        u = panel.unit_ids[obs_idx]
        t = panel.time_ids[obs_idx]
        if panel.D[obs_idx] == 1:
            status[u, t] = 1.0  # treated
        elif panel.cohort[obs_idx] > 0:
            status[u, t] = 0.5  # not-yet-treated
        else:
            status[u, t] = 0.0  # never-treated

    # Sort: never-treated first, then by cohort timing
    unit_cohort = np.array([
        panel.cohort[panel.unit_ids == u][0]
        for u in range(panel.n_units)
    ])
    sort_order = np.argsort(unit_cohort)
    status = status[sort_order]

    import matplotlib.colors as mcolors
    cmap = mcolors.ListedColormap(["#d4e6f1", "#f9e79f", "#e74c3c"])
    bounds = [-0.25, 0.25, 0.75, 1.25]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    time_labels = [panel.time_map.get(t, t) for t in range(panel.n_periods)]
    ax.imshow(status, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xlabel("Time period")
    ax.set_ylabel("Unit (sorted by cohort)")
    ax.set_title(title or "Treatment Status")

    # Tick labels (subsample if many)
    n_t = panel.n_periods
    if n_t <= 20:
        ax.set_xticks(range(n_t))
        ax.set_xticklabels(time_labels, rotation=45, fontsize=7)


def _plot_counterfactual(
    result: Any, ax: Any, *, units: list | None = None, title: str | None = None, **kw: Any,
) -> None:
    """Observed vs counterfactual Y(0) for selected units."""
    panel = result.panel

    if units is None:
        # Pick first 3 treated units
        treated_units = np.unique(panel.unit_ids[panel.is_treated])[:3]
        units = [panel.unit_map.get(u, u) for u in treated_units]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, uid in enumerate(units):
        # Find integer code for this unit
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

        # Use first-stage prediction as counterfactual (includes covariates)
        y_cf = result.y_hat[mask]

        color = colors[i % len(colors)]
        ax.plot(t_vals, y_actual, "-", color=color, linewidth=1.5, label=f"Unit {uid} (actual)")
        ax.plot(t_vals, y_cf, "--", color=color, linewidth=1.5, alpha=0.7, label=f"Unit {uid} (Y(0))")

        # Treatment onset
        cohort_val = panel.cohort[mask][0]
        if cohort_val > 0:
            ax.axvline(cohort_val, color=color, linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Time")
    ax.set_ylabel("Outcome")
    ax.set_title(title or "Counterfactual Comparison")
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
    treated = panel.is_treated
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
