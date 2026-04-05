"""
Generate the test fixture panel dataset.

Creates a deterministic staggered DiD panel with known treatment effects,
saved as parquet for fast loading in tests and benchmarks.

Usage:
    python tests/data/generate_fixture.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

SEED = 20260405
OUTPUT_DIR = Path(__file__).parent


def generate_staggered_panel(
    n_units: int = 1500,
    panel_start: int = 1990,
    panel_end: int = 2020,
    cohorts: dict[int, int] | None = None,
    te: dict[int, float] | None = None,
    te_m: dict[int, float] | None = None,
    n_states: int = 50,
    seed: int = SEED,
) -> pl.DataFrame:
    """Generate a staggered DiD panel dataset.

    Parameters
    ----------
    n_units : int
        Total number of units (split across cohorts + never-treated).
    panel_start, panel_end : int
        Time period range (inclusive).
    cohorts : dict[int, int]
        Mapping of {cohort_year: n_units}. Remaining units are never-treated.
    te : dict[int, float]
        Static treatment effects by cohort year.
    te_m : dict[int, float]
        Dynamic treatment effect slopes by cohort year.
    n_states : int
        Number of cluster groups.
    seed : int
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    years = list(range(panel_start, panel_end + 1))
    T = len(years)

    if cohorts is None:
        cohorts = {2000: 500, 2010: 500}
    if te is None:
        te = {2000: 2.0, 2010: 3.0}
    if te_m is None:
        te_m = {2000: 0.0, 2010: 0.0}

    # Assign units to cohorts
    unit_g = []
    for g, n in cohorts.items():
        unit_g.extend([g] * n)
    n_never = n_units - len(unit_g)
    unit_g.extend([0] * n_never)
    unit_g = np.array(unit_g[:n_units])

    unit_state = rng.integers(1, n_states + 1, size=n_units)
    unit_fe = rng.normal(0, 1, size=n_units)
    year_fe = rng.normal(0, 0.5, size=T)

    rows = []
    for i in range(n_units):
        g = int(unit_g[i])
        for t_idx, yr in enumerate(years):
            treat = int(g > 0 and yr >= g)
            rel_year = (yr - g) if g > 0 else None

            te_val = 0.0
            if treat and g in te:
                te_val = te[g] + te_m.get(g, 0.0) * (yr - g)

            error = rng.normal(0, 1)
            dep_var = unit_fe[i] + year_fe[t_idx] + te_val + error

            rows.append({
                "unit": i + 1,
                "year": yr,
                "state": int(unit_state[i]),
                "g": g,
                "dep_var": round(dep_var, 6),
                "treat": treat,
            })

    return pl.DataFrame(rows)


def main():
    print("Generating test fixture...")

    df = generate_staggered_panel(
        n_units=1500,
        panel_start=1990,
        panel_end=2020,
        cohorts={2000: 500, 2010: 500},
        te={2000: 2.0, 2010: 3.0},
        te_m={2000: 0.0, 2010: 0.0},
        n_states=50,
        seed=SEED,
    )

    # Save parquet
    parquet_path = OUTPUT_DIR / "staggered_panel.parquet"
    df.write_parquet(parquet_path)
    print(f"  {parquet_path}: {len(df)} rows, {df.estimated_size() / 1024:.0f} KB")

    # Save metadata
    meta = {
        "seed": SEED,
        "n_units": 1500,
        "panel": [1990, 2020],
        "cohorts": {"2000": 500, "2010": 500},
        "never_treated": 500,
        "te": {"2000": 2.0, "2010": 3.0},
        "te_m": {"2000": 0.0, "2010": 0.0},
        "n_states": 50,
        "columns": df.columns,
        "shape": list(df.shape),
    }
    meta_path = OUTPUT_DIR / "staggered_panel_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  {meta_path}")

    print("Done.")


if __name__ == "__main__":
    main()
