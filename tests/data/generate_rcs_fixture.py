"""
Generate test fixture RCS datasets.

Creates deterministic repeated cross-section data with known treatment
effects, saved as parquet for fast loading in tests.  Produces both
individual-level and aggregated (group-period) datasets.

Usage:
    python tests/data/generate_rcs_fixture.py
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

SEED = 20260413
OUTPUT_DIR = Path(__file__).parent


def generate_individual_rcs(
    n_groups: int = 50,
    n_individuals_per_cell: int = 30,
    panel_start: int = 2000,
    panel_end: int = 2015,
    cohorts: dict[int, int] | None = None,
    te: dict[int, float] | None = None,
    te_m: dict[int, float] | None = None,
    seed: int = SEED,
) -> pl.DataFrame:
    """Generate individual-level repeated cross-section data.

    Each period draws a fresh random sample of individuals from each
    group.  Individuals do NOT persist across periods.

    Parameters
    ----------
    n_groups : int
        Number of groups (e.g. states).
    n_individuals_per_cell : int
        Individuals sampled per group per period.
    panel_start, panel_end : int
        Time period range (inclusive).
    cohorts : dict[int, int]
        Mapping of {cohort_year: n_groups_in_cohort}.
        Remaining groups are never-treated.
    te : dict[int, float]
        Static treatment effect by cohort year.
    te_m : dict[int, float]
        Dynamic treatment effect slope by cohort year.
    seed : int
        Random seed.
    """
    rng = np.random.default_rng(seed)
    years = list(range(panel_start, panel_end + 1))
    T = len(years)

    if cohorts is None:
        cohorts = {2005: 15, 2010: 10}
    if te is None:
        te = {2005: 2.0, 2010: 3.0}
    if te_m is None:
        te_m = {2005: 0.0, 2010: 0.0}

    # Assign groups to cohorts
    group_cohort = np.zeros(n_groups, dtype=np.int64)
    offset = 0
    for g_year, n_in_cohort in cohorts.items():
        group_cohort[offset : offset + n_in_cohort] = g_year
        offset += n_in_cohort

    # Group fixed effects
    group_fe = rng.normal(loc=0.0, scale=2.0, size=n_groups)

    # Year fixed effects
    year_fe = rng.normal(loc=0.0, scale=0.5, size=T)

    # Region (higher-level cluster): groups 0-24 in region 1, 25-49 in region 2, etc.
    group_region = (np.arange(n_groups) // 10) + 1

    rows_ind = []
    rows_group = []
    rows_region = []
    rows_year = []
    rows_g = []
    rows_dep = []

    ind_counter = 0
    for t_idx, yr in enumerate(years):
        for grp in range(n_groups):
            cohort_yr = int(group_cohort[grp])
            treat_effect = 0.0
            if cohort_yr > 0 and yr >= cohort_yr:
                treat_effect = te.get(cohort_yr, 0.0) + te_m.get(cohort_yr, 0.0) * (yr - cohort_yr)

            for _ in range(n_individuals_per_cell):
                ind_counter += 1
                error = rng.normal(0.0, 1.0)
                y = group_fe[grp] + year_fe[t_idx] + treat_effect + error

                rows_ind.append(ind_counter)
                rows_group.append(grp + 1)
                rows_region.append(int(group_region[grp]))
                rows_year.append(yr)
                rows_g.append(cohort_yr)
                rows_dep.append(round(y, 6))

    return pl.DataFrame({
        "individual_id": rows_ind,
        "group": rows_group,
        "region": rows_region,
        "year": rows_year,
        "g": rows_g,
        "dep_var": rows_dep,
    })


def aggregate_to_group_period(df: pl.DataFrame) -> pl.DataFrame:
    """Collapse individual-level RCS to group-period means."""
    return (
        df.group_by("group", "year", "g", "region")
        .agg(
            pl.col("dep_var").mean().alias("dep_var"),
            pl.col("individual_id").count().alias("n_obs"),
        )
        .sort("group", "year")
    )


def main():
    print("Generating RCS test fixtures...")

    # --- Individual-level RCS ---
    df_ind = generate_individual_rcs(
        n_groups=50,
        n_individuals_per_cell=30,
        panel_start=2000,
        panel_end=2015,
        cohorts={2005: 15, 2010: 10},
        te={2005: 2.0, 2010: 3.0},
        te_m={2005: 0.0, 2010: 0.0},
        seed=SEED,
    )

    path_ind = OUTPUT_DIR / "rcs_individual.parquet"
    df_ind.write_parquet(path_ind)
    print(f"  {path_ind}: {len(df_ind)} rows, {df_ind.estimated_size() / 1024:.0f} KB")

    # --- Aggregated RCS ---
    df_agg = aggregate_to_group_period(df_ind)

    path_agg = OUTPUT_DIR / "rcs_aggregated.parquet"
    df_agg.write_parquet(path_agg)
    print(f"  {path_agg}: {len(df_agg)} rows, {df_agg.estimated_size() / 1024:.0f} KB")

    # --- Metadata ---
    meta = {
        "seed": SEED,
        "n_groups": 50,
        "n_individuals_per_cell": 30,
        "panel": [2000, 2015],
        "cohorts": {"2005": 15, "2010": 10},
        "never_treated_groups": 25,
        "te": {"2005": 2.0, "2010": 3.0},
        "te_m": {"2005": 0.0, "2010": 0.0},
        "individual_shape": list(df_ind.shape),
        "aggregated_shape": list(df_agg.shape),
        "individual_columns": df_ind.columns,
        "aggregated_columns": df_agg.columns,
    }
    meta_path = OUTPUT_DIR / "rcs_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  {meta_path}")

    print("Done.")


if __name__ == "__main__":
    main()
