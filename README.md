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

## Parameters

### `ts_did()` / `bjs_did()`

```python
ts_did(
    data, yname, idname, tname, gname,
    *, xformla=None, wname=None, cluster_var=None,
    anticipation=0, se=True, bootstrap=False,
    n_bootstraps=500, seed=None, n_jobs=None, verbose=True,
) -> DiDResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pl.DataFrame` | required | Panel data in long format (one row per unit per period) |
| `yname` | `str` | required | Outcome variable column |
| `idname` | `str` | required | Unit identifier column |
| `tname` | `str` | required | Time period column (integer-valued) |
| `gname` | `str` | required | Treatment cohort column (see below) |
| `xformla` | `list[str]` | `None` | Time-varying covariate columns for the first stage |
| `wname` | `str` | `None` | Observation weight column |
| `cluster_var` | `str` | `idname` | Column to cluster standard errors on |
| `se` | `bool` | `True` | Compute standard errors |
| `bootstrap` | `bool` | `False` | Use cluster bootstrap instead of analytic SEs |
| `n_bootstraps` | `int` | `500` | Number of bootstrap replications |
| `seed` | `int` | `None` | Random seed for bootstrap |
| `n_jobs` | `int` | CPU count | Parallel workers for bootstrap |
| `verbose` | `bool` | `True` | Print progress |

### The `gname` Column

The `gname` column encodes the treatment cohort for each unit — the time period when treatment begins.

- An integer value (e.g. `2000`) means the unit first receives treatment in that period.
- `0` or `null` means the unit is never treated.
- All observations for a given unit should have the same `gname` value.

The estimator uses this to determine:
- Which observations are untreated (used in the first stage): all observations where `tname < gname`, plus all observations from never-treated units.
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

Input: `polars.DataFrame` in long panel format.

| Column | Type | Description |
|--------|------|-------------|
| outcome | float | Outcome variable |
| unit id | int/str | Unit identifier |
| time | int | Time period (e.g. year) |
| cohort | int | Treatment cohort (0 or null = never-treated) |

```
unit_id | year | cohort_year | dep_var
--------|------|-------------|--------
1       | 1990 | 2000        | 4.23
1       | 1991 | 2000        | 4.51
...
501     | 1990 | 0           | 3.82
501     | 1991 | 0           | 3.90
```

---

## Testing

```bash
uv run pytest tests/                    # all tests
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
