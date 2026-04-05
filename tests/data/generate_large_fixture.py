"""
Generate a large test fixture for stress testing and benchmarking.

Creates a deterministic staggered DiD panel with 50K units, 21 periods,
4 cohorts, and heterogeneous treatment effects.

Usage:
    python tests/data/generate_large_fixture.py
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

SEED = 20260406
OUTPUT_DIR = Path(__file__).parent


def generate_large_panel(
    n_units: int = 50000,
    panel_start: int = 2000,
    panel_end: int = 2020,
    cohorts: dict[int, int] | None = None,
    te: dict[int, float] | None = None,
    te_m: dict[int, float] | None = None,
    n_clusters: int = 200,
    seed: int = SEED,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    years = list(range(panel_start, panel_end + 1))
    T = len(years)

    if cohorts is None:
        cohorts = {2005: 10000, 2008: 10000, 2012: 10000, 2016: 10000}
    if te is None:
        te = {2005: 1.5, 2008: 2.0, 2012: 3.0, 2016: 4.0}
    if te_m is None:
        te_m = {2005: 0.0, 2008: 0.1, 2012: 0.0, 2016: -0.1}

    unit_g = []
    for g, cnt in cohorts.items():
        unit_g.extend([g] * cnt)
    n_never = n_units - len(unit_g)
    unit_g.extend([0] * n_never)
    unit_g = np.array(unit_g[:n_units])

    unit_cluster = rng.integers(1, n_clusters + 1, size=n_units)
    unit_fe = rng.normal(0, 1, size=n_units)
    year_fe = rng.normal(0, 0.5, size=T)

    # Pre-allocate arrays for speed
    total_rows = n_units * T
    units = np.empty(total_rows, dtype=np.int32)
    years_arr = np.empty(total_rows, dtype=np.int32)
    clusters = np.empty(total_rows, dtype=np.int32)
    gs = np.empty(total_rows, dtype=np.int32)
    dep_vars = np.empty(total_rows, dtype=np.float64)
    treats = np.empty(total_rows, dtype=np.int8)

    idx = 0
    for i in range(n_units):
        g = int(unit_g[i])
        for t_idx, yr in enumerate(years):
            treat = int(g > 0 and yr >= g)
            te_val = 0.0
            if treat and g in te:
                te_val = te[g] + te_m.get(g, 0.0) * (yr - g)

            units[idx] = i + 1
            years_arr[idx] = yr
            clusters[idx] = int(unit_cluster[i])
            gs[idx] = g
            dep_vars[idx] = unit_fe[i] + year_fe[t_idx] + te_val + rng.normal(0, 1)
            treats[idx] = treat
            idx += 1

    return pl.DataFrame({
        "unit": units,
        "year": years_arr,
        "cluster": clusters,
        "g": gs,
        "dep_var": dep_vars.round(6),
        "treat": treats,
    })


def main():
    print("Generating large test fixture (50K units x 21 periods)...")

    df = generate_large_panel()

    parquet_path = OUTPUT_DIR / "staggered_panel_large.parquet"
    df.write_parquet(parquet_path)
    print(f"  {parquet_path}: {len(df):,} rows, {df.estimated_size() / 1024 / 1024:.1f} MB")

    meta = {
        "seed": SEED,
        "n_units": 50000,
        "panel": [2000, 2020],
        "cohorts": {"2005": 10000, "2008": 10000, "2012": 10000, "2016": 10000},
        "never_treated": 10000,
        "te": {"2005": 1.5, "2008": 2.0, "2012": 3.0, "2016": 4.0},
        "te_m": {"2005": 0.0, "2008": 0.1, "2012": 0.0, "2016": -0.1},
        "n_clusters": 200,
        "columns": df.columns,
        "shape": list(df.shape),
    }
    meta_path = OUTPUT_DIR / "staggered_panel_large_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
