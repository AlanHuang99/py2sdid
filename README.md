# py2sdid

Two-stage difference-in-differences (Gardner 2021) and BJS imputation estimator (Borusyak, Jaravel, Spiess 2024) for staggered treatment designs in Python, built on polars, scipy (sparse), and numpy.

## Installation

```bash
pip install py2sdid
```

**From source:**
```bash
git clone https://github.com/AlanHuang99/py2sdid.git
cd py2sdid
pip install -e ".[dev]"
```

## Quick Start

```python
import polars as pl
import py2sdid

data = pl.read_parquet("panel_data.parquet")

result = py2sdid.ts_did(
    data,
    yname="dep_var",
    idname="unit_id",
    tname="year",
    gname="cohort_year",
    cluster_var="cluster_id",
)

result.summary()
result.plot(kind="event_study")
result.diagnose()
```

## Estimators

### `ts_did` -- Two-Stage DiD (Gardner 2021)

Estimates unit and time fixed effects on untreated observations only, residualizes the outcome for the full sample, and regresses on treatment indicators. Standard errors use GMM influence functions that correct for the generated-regressor problem.

### `bjs_did` -- BJS Imputation (Borusyak, Jaravel, Spiess 2024)

Same first stage as `ts_did`. Directly averages residuals for treated observations rather than running a second-stage regression. Standard errors use the BJS imputation formula (Equations 6, 8, 10 of the paper).

Both produce identical point estimates. They differ in the analytic standard error formula. Both support cluster bootstrap as an alternative.

---

## Dataset Types

py2sdid supports three data configurations through the `dataset_type` and `groupname` parameters.

### 1. Panel Data (default)

Same units tracked over time. Each unit has one row per period. Unit fixed effects are estimated from `idname`.

```python
# unit_id | year | cohort | dep_var
# --------|------|--------|--------
# 1       | 2000 | 2005   | 4.23     <- unit 1 in 2000
# 1       | 2001 | 2005   | 4.51     <- same unit 1 in 2001
# ...

result = py2sdid.ts_did(
    data,
    yname="dep_var",
    idname="unit_id",        # unit identifier (used for unit FE)
    tname="year",
    gname="cohort",
)
```

### 2. Individual-Level Repeated Cross-Section

A fresh sample of individuals is drawn each period from the same groups (e.g., states, regions). Treatment is at the group level. Each individual typically appears only once. Group fixed effects replace unit fixed effects.

```python
# individual_id | state | year | cohort | dep_var
# --------------|-------|------|--------|--------
# 1001          | CA    | 2000 | 2005   | 4.23    <- individual 1001, sampled once
# 1002          | CA    | 2000 | 2005   | 3.87    <- different individual, same state
# 2001          | CA    | 2001 | 2005   | 4.60    <- new sample in 2001
# ...

result = py2sdid.ts_did(
    data,
    yname="dep_var",
    idname="individual_id",  # individual identifier (for observation tracking)
    tname="year",
    gname="cohort",
    dataset_type="rcs",      # repeated cross-section mode
    groupname="state",       # group FE estimated from this column
)
```

Key behavior:
- **Fixed effects** estimated from `groupname` (not `idname`)
- **Clustering** defaults to `groupname` (not `idname`)
- **Validation** checks `gname` is constant within each group

### 3. Aggregated Repeated Cross-Section

Each row is already a group-period aggregate (e.g., state-year means). `idname` IS the group identifier. No separate `groupname` needed.

```python
# state | year | cohort | dep_var
# ------|------|--------|--------
# CA    | 2000 | 2005   | 4.05     <- state-year average
# CA    | 2001 | 2005   | 4.15
# TX    | 2000 | 0      | 3.82     <- never-treated state
# ...

result = py2sdid.ts_did(
    data,
    yname="dep_var",
    idname="state",          # group identifier (used for group FE)
    tname="year",
    gname="cohort",
    dataset_type="rcs",      # group FE mode (no unit tracking needed)
)
```

Key behavior:
- **Fixed effects** estimated from `idname` (which IS the group)
- **Clustering** defaults to `idname`
- Functionally identical to panel mode but semantically distinct

---

## Parameters

### `ts_did()` / `bjs_did()`

```python
ts_did(
    data, yname, idname, tname, gname,
    *, dataset_type="panel", groupname=None,
    xformla=None, wname=None, cluster_var=None,
    se=True, bootstrap=False,
    n_bootstraps=500, seed=None, n_jobs=None, verbose=True,
) -> DiDResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pl.DataFrame` | required | Data in long format |
| `yname` | `str` | required | Outcome variable column |
| `idname` | `str` | required | Unit identifier (panel) or group identifier (aggregated RCS) |
| `tname` | `str` | required | Time period column (integer-valued) |
| `gname` | `str` | required | Treatment cohort column (see below) |
| `dataset_type` | `str` | `"panel"` | `"panel"` or `"rcs"` |
| `groupname` | `str` | `None` | Group column for individual-level RCS (must not be set when `dataset_type="panel"`) |
| `xformla` | `list[str]` | `None` | Time-varying covariate columns for the first stage |
| `wname` | `str` | `None` | Observation weight column |
| `cluster_var` | `str` | auto | Column to cluster SEs on. Defaults to `idname` (panel), `groupname` (individual RCS), or `idname` (aggregated RCS) |
| `se` | `bool` | `True` | Compute standard errors |
| `bootstrap` | `bool` | `False` | Use cluster bootstrap instead of analytic SEs |
| `n_bootstraps` | `int` | `500` | Number of bootstrap replications |
| `seed` | `int` | `None` | Random seed for bootstrap |
| `n_jobs` | `int` | CPU count | Parallel workers for bootstrap |
| `verbose` | `bool` | `True` | Print progress |

### The `gname` Column

The `gname` column encodes the treatment cohort — the time period when treatment begins.

- An integer value (e.g. `2000`) means the unit/group first receives treatment in that period.
- `0` or `null` means never treated.
- Must be constant within each unit (panel) or group (RCS).

The estimator uses this to determine:
- Which observations are untreated (used in the first stage): all observations where `tname < gname`, plus all observations from never-treated units/groups.
- Which observations are treated (used for treatment effect estimation): observations where `tname >= gname`.
- The relative time (event time) for each observation: `tname - gname`.

---

## Output

Both estimators return a `DiDResult` with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `att_avg` | `float` | Overall average treatment effect on treated |
| `att_avg_se` | `float` | Standard error of overall ATT |
| `att_avg_ci` | `tuple` | 95% confidence interval |
| `att_avg_pval` | `float` | p-value (H0: ATT = 0) |
| `event_study` | `pl.DataFrame` | Per-period estimates for all relative time periods |
| `unit_fe` | `np.ndarray` | Estimated unit fixed effects |
| `time_fe` | `np.ndarray` | Estimated time fixed effects |
| `vcov` | `np.ndarray` | Variance-covariance matrix |

The `event_study` DataFrame contains columns: `rel_time`, `estimate`, `se`, `ci_lower`, `ci_upper`, `pval`, `count`. It includes all relative time periods present in the data, both pre-treatment and post-treatment.

Convenience properties:
- `result.att_by_horizon` — post-treatment rows only (`rel_time >= 0`)
- `result.pretrend_tests` — pre-treatment rows only (`rel_time < 0`)

### Methods

| Method | Description |
|--------|-------------|
| `summary()` | Formatted text summary |
| `plot(kind)` | Matplotlib figure. Kinds: `event_study`, `pretrends`, `treatment_status`, `counterfactual`, `honestdid`, `calendar` |
| `diagnose()` | Pre-trend F-test, TOST equivalence, HonestDiD sensitivity |

---

## Data Format

Input: `polars.DataFrame` in long format. The required columns depend on the dataset type.

**Panel data:**

| Column | Type | Description |
|--------|------|-------------|
| outcome | float | Outcome variable |
| unit id | int/str | Unit identifier (appears in multiple periods) |
| time | int | Time period (e.g. year) |
| cohort | int | Treatment cohort (0 or null = never-treated) |

**Individual-level RCS** (additional column):

| Column | Type | Description |
|--------|------|-------------|
| group | int/str | Group identifier (e.g. state) for fixed effects |

**Aggregated RCS:** Same as panel, but unit id is the group identifier and each (group, time) pair has one row.

---

## Performance

Analytic SE computation uses sparse LU factorization and sparse matrix
operations throughout, avoiding dense intermediates that would blow up
memory and runtime for large panels.

| N units | Observations | ts_did | bjs_did |
|---------|-------------|--------|---------|
| 1,000 | 31,000 | 0.14s | 0.37s |
| 5,000 | 155,000 | 0.46s | 1.44s |
| 10,000 | 310,000 | 0.85s | 2.90s |
| 20,000 | 620,000 | 1.60s | 5.95s |
| 50,000 | 1,050,000 | 2.47s | 11.55s |

Timings are for the full pipeline (panel prep + first stage + effects +
analytic SEs) on a single core. For very large panels, `bootstrap=True`
with `n_jobs` scales linearly across cores.

---

## Testing

```bash
uv run pytest tests/                    # full suite (106 tests)
uv run pytest tests/test_rcs.py -v      # RCS-specific tests (38 tests)
uv run pytest tests/test_vs_r.py -v -s  # R validation (requires R + did2s + didimputation)
```

Test data fixture in `tests/data/` is generated deterministically via `tests/data/generate_fixture.py`.

---

## API Reference

- **[REFERENCE.md](REFERENCE.md)** -- parameter tables, return types
- **[API docs](https://alanhuang99.github.io/py2sdid/)** -- searchable HTML reference (pdoc)

---

## References

- Gardner, J. (2021). "Two-Stage Differences in Differences." Working paper.
- Borusyak, K., Jaravel, X., & Spiess, J. (2024). "Revisiting Event-Study Designs: Robust and Efficient Estimation." *Review of Economic Studies*, 91(6), 3253-3285.
- Rambachan, A. & Roth, J. (2023). "A More Credible Approach to Parallel Trends." *Review of Economic Studies*, 90(5), 2555-2591.
- Butts, K. & Gardner, J. (2022). "did2s: Two-Stage Difference-in-Differences." *The R Journal*, 14(1).

## License

MIT
