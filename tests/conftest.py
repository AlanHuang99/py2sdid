"""
Shared fixtures and DGP helpers for py2sdid tests.

``gen_data()`` is a faithful Python port of R ``did2s::gen_data`` — it
generates staggered-adoption panel data with unit FE, year FE, treatment
effects (level + dynamic slope), and N(0,1) noise.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest


# ---------------------------------------------------------------------------
# DGP: port of R did2s::gen_data
# ---------------------------------------------------------------------------

def gen_data(
    n: int = 1500,
    panel: tuple[int, int] = (1990, 2020),
    g1: int = 2000,
    g2: int = 2010,
    g3: int = 0,
    te1: float = 2.0,
    te2: float = 2.0,
    te3: float = 0.0,
    te_m1: float = 0.0,
    te_m2: float = 0.0,
    te_m3: float = 0.0,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate staggered DiD panel data (port of R did2s::gen_data).

    Parameters
    ----------
    n : int
        Number of units.
    panel : (int, int)
        Start and end years (inclusive).
    g1, g2, g3 : int
        Treatment onset year for groups 1, 2, 3.  Set to 0 for never-treated.
    te1, te2, te3 : float
        Level treatment effect for each group.
    te_m1, te_m2, te_m3 : float
        Treatment effect slope (additional effect per year post-treatment).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pl.DataFrame
        Columns: unit, year, state, g, dep_var, treat, rel_year
    """
    rng = np.random.default_rng(seed)

    years = list(range(panel[0], panel[1] + 1))
    n_years = len(years)

    # --- Unit-level attributes ---
    state = rng.integers(1, 51, size=n)
    group_draw = rng.uniform(size=n)

    # Assign groups (matching R: <0.33 -> G1, <0.66 -> G2, else G3)
    group = np.where(
        group_draw < 1 / 3,
        1,
        np.where(group_draw < 2 / 3, 2, 3),
    )

    # Treatment cohort (g column)
    g_map = {1: g1, 2: g2, 3: g3}
    g_val = np.array([g_map[gi] for gi in group], dtype=np.int64)

    # Unit fixed effects: N(state/5, 1)
    unit_fe = rng.normal(loc=state / 5.0, scale=1.0)

    # --- Year-level attributes ---
    # Year FE: one draw per year, N(0, 0.2)
    # (R does rnorm(1, 0, 0.2) grouped by year — one value per year)
    year_fe_vals = rng.normal(loc=0.0, scale=0.2, size=n_years)

    # --- Expand to panel (unit x year) ---
    # Arrays of length n * n_years
    total = n * n_years

    unit_arr = np.repeat(np.arange(1, n + 1), n_years)
    year_arr = np.tile(years, n)
    state_arr = np.repeat(state, n_years)
    g_arr = np.repeat(g_val, n_years)
    unit_fe_arr = np.repeat(unit_fe, n_years)

    # Map year -> year_fe
    year_idx = np.tile(np.arange(n_years), n)
    year_fe_arr = year_fe_vals[year_idx]

    # Treatment indicator: treat = (year >= g) & (g within panel range)
    valid_panel = np.array(years)
    g_in_panel = np.isin(g_arr, valid_panel)
    treat = ((year_arr >= g_arr) & g_in_panel).astype(np.int32)

    # Relative year (event time)
    rel_year = (year_arr - g_arr).astype(np.float64)
    rel_year[g_arr == 0] = np.inf

    # Error term: N(0, 1)
    error = rng.normal(loc=0.0, scale=1.0, size=total)

    # --- Treatment effects ---
    group_arr = np.repeat(group, n_years)

    # Level effect
    te = np.zeros(total)
    for grp, gi, tei in [(1, g1, te1), (2, g2, te2), (3, g3, te3)]:
        mask = (group_arr == grp) & (year_arr >= gi) & (gi != 0)
        te[mask] = tei

    # Dynamic (slope) effect
    te_dynamic = np.zeros(total)
    for grp, gi, temi in [(1, g1, te_m1), (2, g2, te_m2), (3, g3, te_m3)]:
        mask = (group_arr == grp) & (year_arr >= gi) & (gi != 0)
        te_dynamic[mask] = temi * (year_arr[mask] - gi)

    # Dependent variable
    dep_var = unit_fe_arr + year_fe_arr + te + te_dynamic + error

    return pl.DataFrame({
        "unit": unit_arr,
        "year": year_arr,
        "state": state_arr.astype(np.int64),
        "g": g_arr,
        "dep_var": dep_var,
        "treat": treat,
        "rel_year": rel_year,
    }).sort("unit", "year")


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_hom() -> pl.DataFrame:
    """Homogeneous treatment effect panel (default DGP).

    - 1500 units, panel 1990-2020
    - Group 1 treated at 2000, Group 2 at 2010, Group 3 never-treated
    - Constant TE = 2 for both treated groups
    """
    return gen_data(
        n=1500,
        panel=(1990, 2020),
        g1=2000,
        g2=2010,
        g3=0,
        te1=2,
        te2=2,
        te3=0,
        te_m1=0,
        te_m2=0,
        te_m3=0,
        seed=42,
    )


@pytest.fixture
def df_het() -> pl.DataFrame:
    """Heterogeneous treatment effect panel.

    - Same structure as df_hom but with different TE levels and dynamic slopes
    - Group 1: TE=2 + 0.05/year, Group 2: TE=1 + 0.15/year
    """
    return gen_data(
        n=1500,
        panel=(1990, 2020),
        g1=2000,
        g2=2010,
        g3=0,
        te1=2,
        te2=1,
        te3=0,
        te_m1=0.05,
        te_m2=0.15,
        te_m3=0,
        seed=42,
    )


@pytest.fixture
def df_small() -> pl.DataFrame:
    """Small panel for fast tests.

    - 100 units, panel 2000-2010
    - Single treated group at 2005, TE=3
    """
    return gen_data(
        n=100,
        panel=(2000, 2010),
        g1=2005,
        g2=0,
        g3=0,
        te1=3,
        te2=0,
        te3=0,
        te_m1=0,
        te_m2=0,
        te_m3=0,
        seed=123,
    )


# ---------------------------------------------------------------------------
# Repeated cross-section DGP
# ---------------------------------------------------------------------------

def gen_rcs_data(
    n_groups: int = 50,
    n_individuals_per_group_period: int = 20,
    panel: tuple[int, int] = (2000, 2010),
    g1: int = 2005,
    g2: int = 0,
    te: float = 3.0,
    te_m: float = 0.0,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate repeated cross-section data with group-level treatment.

    Each period draws a fresh sample of individuals from each group.
    Individuals do NOT persist across periods — this is the defining
    feature of RCS data.  Treatment is at the group level.

    Parameters
    ----------
    n_groups : int
        Number of groups (e.g. states, regions).
    n_individuals_per_group_period : int
        Individuals sampled per group per period.
    panel : (int, int)
        Start and end years (inclusive).
    g1 : int
        Treatment onset year for treated groups. Set to 0 for never-treated.
    g2 : int
        Second treatment cohort (0 = never-treated).
    te : float
        Level treatment effect.
    te_m : float
        Slope treatment effect per year post-treatment.
    seed : int
        Random seed.

    Returns
    -------
    pl.DataFrame
        Columns: individual_id, group, year, g, dep_var
    """
    rng = np.random.default_rng(seed)
    years = list(range(panel[0], panel[1] + 1))
    n_years = len(years)

    # Assign groups to treatment cohorts
    # ~half treated at g1, rest never-treated (or g2)
    group_cohort = np.zeros(n_groups, dtype=np.int64)
    group_cohort[: n_groups // 2] = g1
    if g2 > 0:
        quarter = n_groups // 4
        group_cohort[quarter : n_groups // 2] = g2

    # Group fixed effects
    group_fe = rng.normal(loc=0.0, scale=2.0, size=n_groups)

    # Year fixed effects
    year_fe_vals = rng.normal(loc=0.0, scale=0.5, size=n_years)

    rows_individual = []
    rows_group = []
    rows_year = []
    rows_g = []
    rows_dep = []

    ind_counter = 0
    for t_idx, yr in enumerate(years):
        for grp in range(n_groups):
            n_ind = n_individuals_per_group_period
            cohort = int(group_cohort[grp])

            # Treatment effect
            treat_effect = 0.0
            if cohort > 0 and yr >= cohort:
                treat_effect = te + te_m * (yr - cohort)

            for _ in range(n_ind):
                ind_counter += 1
                error = rng.normal(0.0, 1.0)
                y = group_fe[grp] + year_fe_vals[t_idx] + treat_effect + error

                rows_individual.append(ind_counter)
                rows_group.append(grp + 1)
                rows_year.append(yr)
                rows_g.append(cohort)
                rows_dep.append(y)

    return pl.DataFrame({
        "individual_id": rows_individual,
        "group": rows_group,
        "year": rows_year,
        "g": rows_g,
        "dep_var": rows_dep,
    })


@pytest.fixture
def df_rcs() -> pl.DataFrame:
    """Repeated cross-section with group-level treatment.

    - 50 groups, 20 individuals/group/period, panel 2000-2010
    - 25 groups treated at 2005, 25 never-treated
    - Constant TE = 3
    """
    return gen_rcs_data(
        n_groups=50,
        n_individuals_per_group_period=20,
        panel=(2000, 2010),
        g1=2005,
        g2=0,
        te=3.0,
        te_m=0.0,
        seed=42,
    )


@pytest.fixture
def df_rcs_small() -> pl.DataFrame:
    """Small RCS for fast tests.

    - 10 groups, 10 individuals/group/period, panel 2000-2005
    - 5 groups treated at 2003, 5 never-treated
    - TE = 2
    """
    return gen_rcs_data(
        n_groups=10,
        n_individuals_per_group_period=10,
        panel=(2000, 2005),
        g1=2003,
        g2=0,
        te=2.0,
        te_m=0.0,
        seed=123,
    )


def gen_agg_rcs_data(
    n_groups: int = 50,
    panel: tuple[int, int] = (2000, 2010),
    g1: int = 2005,
    g2: int = 0,
    te: float = 3.0,
    te_m: float = 0.0,
    n_per_cell: int = 30,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate aggregated repeated cross-section data.

    Each row is a (group, year) cell with the outcome averaged over
    ``n_per_cell`` individuals.  This is the format users get after
    collapsing individual-level RCS data to the group-period level.

    Returns
    -------
    pl.DataFrame
        Columns: group, year, g, dep_var
    """
    rng = np.random.default_rng(seed)
    years = list(range(panel[0], panel[1] + 1))
    n_years = len(years)

    group_cohort = np.zeros(n_groups, dtype=np.int64)
    group_cohort[: n_groups // 2] = g1
    if g2 > 0:
        quarter = n_groups // 4
        group_cohort[quarter : n_groups // 2] = g2

    group_fe = rng.normal(loc=0.0, scale=2.0, size=n_groups)
    year_fe_vals = rng.normal(loc=0.0, scale=0.5, size=n_years)

    rows_group = []
    rows_year = []
    rows_g = []
    rows_dep = []

    for t_idx, yr in enumerate(years):
        for grp in range(n_groups):
            cohort = int(group_cohort[grp])
            treat_effect = 0.0
            if cohort > 0 and yr >= cohort:
                treat_effect = te + te_m * (yr - cohort)

            # Average over n_per_cell individuals (reduces noise)
            errors = rng.normal(0.0, 1.0, size=n_per_cell)
            y = group_fe[grp] + year_fe_vals[t_idx] + treat_effect + errors.mean()

            rows_group.append(grp + 1)
            rows_year.append(yr)
            rows_g.append(cohort)
            rows_dep.append(y)

    return pl.DataFrame({
        "group": rows_group,
        "year": rows_year,
        "g": rows_g,
        "dep_var": rows_dep,
    })


@pytest.fixture
def df_agg_rcs() -> pl.DataFrame:
    """Aggregated RCS (one row per group-period).

    - 50 groups, panel 2000-2010
    - 25 groups treated at 2005, 25 never-treated
    - TE = 3, averaged over 30 individuals per cell
    """
    return gen_agg_rcs_data(
        n_groups=50,
        panel=(2000, 2010),
        g1=2005,
        te=3.0,
        seed=42,
    )
