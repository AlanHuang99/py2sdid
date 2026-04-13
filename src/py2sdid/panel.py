"""
Panel data ingestion — Polars DataFrame to PanelData.

Converts long-format staggered-adoption panel data into the structured
arrays used by the estimation engine.  Handles treatment timing,
event-time computation, and integer-coding of identifiers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from .results import PanelData


def prepare_panel(
    data: pl.DataFrame,
    yname: str,
    idname: str,
    tname: str,
    gname: str,
    *,
    dataset_type: str = "panel",
    groupname: str | None = None,
    xformla: list[str] | None = None,
    wname: str | None = None,
    cluster_var: str | None = None,
) -> PanelData:
    """Convert long-format Polars DataFrame to structured PanelData.

    Supports three configurations:

    1. **Panel** (``dataset_type="panel"``): one row per unit-period,
       unit fixed effects from ``idname``.
    2. **Individual-level RCS** (``dataset_type="rcs"``,
       ``groupname`` provided): individuals sampled once, group FE
       from ``groupname``.
    3. **Aggregated RCS** (``dataset_type="rcs"``, no ``groupname``):
       each row is a group-period aggregate, group FE from ``idname``.

    Parameters
    ----------
    data : pl.DataFrame
        Data in long format.
    yname : str
        Outcome column.
    idname : str
        Unit identifier column (panel) or group identifier column
        (aggregated RCS).
    tname : str
        Time period column (must be integer-valued).
    gname : str
        Treatment cohort column.  A positive integer indicates the first
        period of treatment; ``0`` or ``null`` means never-treated.
        Must be constant within each unit (panel) or group (RCS).
    dataset_type : str
        ``"panel"`` or ``"rcs"``.
    groupname : str, optional
        Group identifier for individual-level RCS data.  When provided
        with ``dataset_type="rcs"``, group FE replace unit FE.
    xformla : list[str], optional
        Time-varying covariate column names.
    wname : str, optional
        Observation weight column.
    cluster_var : str, optional
        Clustering variable (defaults to *idname* for panels,
        *groupname* or *idname* for RCS).
    """
    is_rcs = dataset_type == "rcs"

    # -- Validate columns ------------------------------------------------
    required = [yname, idname, tname, gname]
    _check_columns(data, required)
    if groupname:
        _check_columns(data, [groupname])
    if xformla:
        _check_columns(data, xformla)
    if wname:
        _check_columns(data, [wname])
    if cluster_var:
        _check_columns(data, [cluster_var])

    # -- Validate tname is integer-like ----------------------------------
    _validate_integer_column(data, tname)

    # -- Sort for deterministic ordering ---------------------------------
    df = data.sort(idname, tname)

    # -- Determine the FE grouping variable ------------------------------
    # Panel:              gname constant within idname, FE from idname
    # Individual RCS:     gname constant within groupname, FE from groupname
    # Aggregated RCS:     gname constant within idname, FE from idname
    if is_rcs and groupname is not None:
        fe_col = groupname
    else:
        fe_col = idname
    _validate_gname_constant(df, fe_col, gname)

    # -- Integer-code identifiers ----------------------------------------
    unit_ids, unit_map = _factorize(df[idname])
    time_ids, time_map = _factorize(df[tname])

    # -- FE identifiers --------------------------------------------------
    if is_rcs and groupname is not None:
        # Individual-level RCS: FE from separate group column
        fe_ids, fe_map = _factorize(df[groupname])
        n_fe_levels = len(fe_map)
    elif is_rcs:
        # Aggregated RCS: FE from idname (idname IS the group)
        fe_ids = unit_ids
        fe_map = unit_map
        n_fe_levels = len(unit_map)
    else:
        # Panel: FE from idname (standard unit FE)
        fe_ids = unit_ids
        fe_map = unit_map
        n_fe_levels = None  # defaults to n_units in __post_init__

    # -- Cluster defaults ------------------------------------------------
    # Panel:           cluster by idname (unit)
    # Individual RCS:  cluster by groupname (group)
    # Aggregated RCS:  cluster by idname (group)
    if cluster_var is None:
        if is_rcs and groupname is not None:
            cluster_ids = fe_ids.copy()
            cluster_map = dict(fe_map)
        else:
            cluster_ids = unit_ids.copy()
            cluster_map = dict(unit_map)
    else:
        cluster_ids, cluster_map = _factorize(df[cluster_var])

    n_units = len(unit_map)
    n_periods = len(time_map)
    n_obs = len(df)

    # -- Cohort / treatment timing ---------------------------------------
    g_raw = df[gname].to_numpy().copy().astype(np.float64)
    # null → 0  (never-treated)
    g_raw = np.nan_to_num(g_raw, nan=0.0)
    cohort = g_raw.astype(np.int64)

    # Time values (original scale)
    t_vals = df[tname].to_numpy().astype(np.float64)

    # Event time: e = t - g  (inf for never-treated)
    event_time = np.where(cohort > 0, t_vals - cohort, np.inf)

    # -- Treatment indicator ---------------------------------------------
    D = np.where((cohort > 0) & (t_vals >= cohort), 1, 0).astype(np.int32)

    # -- Outcome ---------------------------------------------------------
    Y = df[yname].to_numpy().astype(np.float64)

    # -- Covariates ------------------------------------------------------
    if xformla:
        X = df.select(xformla).to_numpy().astype(np.float64)
    else:
        X = None

    # -- Weights ---------------------------------------------------------
    if wname:
        W = df[wname].to_numpy().astype(np.float64)
    else:
        W = None

    # -- Masks -----------------------------------------------------------
    is_treated = D == 1
    is_control = D == 0
    n_treated = int(is_treated.sum())
    n_control = int(is_control.sum())

    # -- Cohort sizes (number of *units* per cohort) ---------------------
    unit_cohort = np.column_stack([unit_ids, cohort])
    unique_uc = np.unique(unit_cohort, axis=0)
    cohort_sizes: dict[int, int] = {}
    for c_val in np.unique(unique_uc[:, 1]):
        cohort_sizes[int(c_val)] = int((unique_uc[:, 1] == c_val).sum())

    return PanelData(
        Y=Y,
        D=D,
        unit_ids=unit_ids,
        time_ids=time_ids,
        cohort=cohort,
        event_time=event_time,
        X=X,
        W=W,
        cluster=cluster_ids,
        n_units=n_units,
        n_periods=n_periods,
        n_treated=n_treated,
        n_control=n_control,
        n_obs=n_obs,
        cohort_sizes=cohort_sizes,
        is_treated=is_treated,
        is_control=is_control,
        unit_map=unit_map,
        time_map=time_map,
        cluster_map=cluster_map,
        fe_ids=fe_ids,
        n_fe_levels=n_fe_levels,
        fe_map=fe_map if is_rcs and groupname is not None else None,
        is_rcs=is_rcs,
    )


# -- Helpers -------------------------------------------------------------

def _check_columns(df: pl.DataFrame, cols: list[str]) -> None:
    """Raise ValueError if any column is missing."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def _validate_integer_column(df: pl.DataFrame, col: str) -> None:
    """Validate that a column contains integer-like values."""
    dtype = df[col].dtype
    if dtype.is_integer():
        return
    if dtype.is_float():
        vals = df[col].drop_nulls()
        if not (vals == vals.round(0)).all():
            raise ValueError(
                f"Column '{col}' must be integer-valued, "
                f"but contains non-integer floats"
            )
        return
    raise ValueError(
        f"Column '{col}' must be integer-valued, got dtype {dtype}"
    )


def _validate_gname_constant(df: pl.DataFrame, group_col: str, gname: str) -> None:
    """Validate that gname is constant within each unit/group."""
    g_filled = df.with_columns(pl.col(gname).fill_null(0).alias("_g_check"))
    varying = (
        g_filled.group_by(group_col)
        .agg(pl.col("_g_check").n_unique().alias("_n_g"))
        .filter(pl.col("_n_g") > 1)
    )
    if len(varying) > 0:
        bad_ids = varying[group_col].head(5).to_list()
        raise ValueError(
            f"Column '{gname}' must be constant within each '{group_col}'. "
            f"Values with varying '{gname}': {bad_ids}"
        )


def _factorize(series: pl.Series) -> tuple[np.ndarray, dict[int, Any]]:
    """Integer-code a Polars Series using native Polars operations.

    Returns
    -------
    codes : np.ndarray[int64]
        Integer codes starting at 0.
    mapping : dict[int, Any]
        Code -> original value.
    """
    # Use Polars rank-based approach for stable, type-safe factorization
    unique_vals = series.unique().sort()
    val_to_code = {v: i for i, v in enumerate(unique_vals.to_list())}
    codes = series.replace_strict(val_to_code, return_dtype=pl.Int64).to_numpy()
    mapping = {i: v for v, i in val_to_code.items()}
    return codes, mapping
