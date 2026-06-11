# py2sdid API Reference

**Version 0.1.7**

## Overview

py2sdid implements two-stage difference-in-differences (Gardner 2021) and BJS
imputation (Borusyak, Jaravel, Spiess 2024) for staggered treatment adoption.
Built on polars (data), scipy.sparse (fixed effects), and numpy (computation).

**Pipeline:** `prepare_panel()` -> `estimate_first_stage()` -> `compute_effects()` -> `compute_se_*()` -> `DiDResult`

---

## `ts_did()`

Two-stage DiD estimator. Analytic SEs via GMM influence functions.

```python
ts_did(
    data, yname, idname, tname, gname,
    *, dataset_type="panel", groupname=None, drop_singletons=True,
    xformla=None, wname=None, cluster_var=None,
    se=True, bootstrap=False,
    n_bootstraps=500, seed=None, n_jobs=None,
    diagnostics="none", diagnostics_options=None,
    verbose=True,
) -> DiDResult
```

## `bjs_did()`

BJS imputation estimator. Same signature as `ts_did()`. Identical point estimates, different analytic SE formula.

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pl.DataFrame` | required | Panel data in long format (unit x time) |
| `yname` | `str` | required | Outcome variable column |
| `idname` | `str` | required | Unit identifier column |
| `tname` | `str` | required | Time period column (integer) |
| `gname` | `str` | required | Treatment cohort column. Value = period when treatment begins. `0` or `null` = never-treated |
| `dataset_type` | `str` | `"panel"` | `"panel"` or `"rcs"` |
| `groupname` | `str` | `None` | Group column for individual-level RCS |
| `drop_singletons` | `bool` | `True` | Drop FE groups with no control-subsample observations |
| `xformla` | `list[str]` | `None` | Time-varying covariate columns for first stage |
| `wname` | `str` | `None` | Observation weight column |
| `cluster_var` | `str` | `idname` | Column to cluster standard errors on |
| `se` | `bool` | `True` | Compute standard errors |
| `bootstrap` | `bool` | `False` | Use cluster bootstrap |
| `n_bootstraps` | `int` | `500` | Bootstrap replications |
| `seed` | `int` | `None` | Random seed |
| `n_jobs` | `int` | CPU count | Parallel workers for bootstrap |
| `diagnostics` | `"none"`, `"full"`, or `list[str]` | `"none"` | Fit-time diagnostics to compute and store on `result.diagnostics` |
| `diagnostics_options` | `dict \| None` | `None` | Options such as `delta`, `placebo_period`, `alpha`, `honestdid_e`, `honestdid_Mvec` |
| `verbose` | `bool` | `True` | Print progress |

### The `gname` Column

Each value in `gname` indicates the time period when treatment begins for that unit.

- Integer value (e.g. `2000`): unit first receives treatment in that period
- `0` or `null`: unit is never treated
- Must be constant within each unit

The estimator derives from `gname`:
- Untreated observations: `tname < gname` plus all never-treated observations
- Treated observations: `tname >= gname`
- Relative time: `tname - gname`

---

## `DiDResult`

Returned by `ts_did()` and `bjs_did()`.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `method` | `str` | `"ts_did"` or `"bjs_did"` |
| `att_avg` | `float` | Overall average treatment effect on treated |
| `att_avg_se` | `float \| None` | SE of overall ATT |
| `att_avg_ci` | `tuple[float, float] \| None` | 95% confidence interval |
| `att_avg_pval` | `float \| None` | p-value (H0: ATT=0) |
| `event_study` | `pl.DataFrame` | Per-period estimates for all relative time periods |
| `unit_fe` | `np.ndarray` | (n_units,) unit fixed effects |
| `time_fe` | `np.ndarray` | (n_periods,) time fixed effects |
| `beta` | `np.ndarray \| None` | (K,) covariate coefficients |
| `effects` | `np.ndarray` | (n_treated_obs,) individual treatment effects |
| `vcov` | `np.ndarray \| None` | Variance-covariance matrix |
| `boot_dist` | `np.ndarray \| None` | Bootstrap distribution |
| `panel` | `PanelData` | Panel structure |
| `sigma2` | `float` | First-stage error variance |
| `n_clusters` | `int \| None` | Stored cluster count for slim-safe diagnostics |
| `diagnostics` | `DiagnosticResult \| None` | Fit-time or cached diagnostics |

### `event_study` DataFrame

Contains per-period estimates for every relative time period in the data. Negative relative times, including `rel_time = -1` when present, are pre-treatment/placebo estimates and are not omitted as a reference period. Post-treatment horizons (`rel_time >= 0`) are imputation/two-stage effects relative to the imputed untreated counterfactual.

| Column | Type | Description |
|--------|------|-------------|
| `rel_time` | `int` | Relative time (tname - gname). Negative = pre-treatment, 0+ = post-treatment |
| `estimate` | `float` | Per-period ATT estimate |
| `se` | `float` | Standard error |
| `ci_lower` | `float` | 95% CI lower bound |
| `ci_upper` | `float` | 95% CI upper bound |
| `pval` | `float` | p-value |
| `count` | `int` | Number of observations at this relative time |

### Properties

| Property | Returns | Description |
|----------|---------|-------------|
| `att_by_horizon` | `pl.DataFrame` | Post-treatment rows of `event_study` (rel_time >= 0) |
| `pretrend_tests` | `pl.DataFrame \| None` | Pre-treatment rows of `event_study` (rel_time < 0) |
| `bootstrap_atts` | `np.ndarray \| None` | Alias for `boot_dist` |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `str` | Formatted text summary |
| `plot(kind, **kwargs)` | `Figure` | Matplotlib figure |
| `diagnose(**kwargs)` | `DiagnosticResult` | Diagnostic tests; caches the result on `result.diagnostics` |

---

## `DiagnosticResult`

Returned by `result.diagnose()` and stored on `result.diagnostics` when
fit-time diagnostics are requested. The envelope is slim-safe: it stores
only primitives, tuples, dicts, and bounded `numpy` arrays. Legacy flat
attributes synthesize Polars tables on access for backward compatibility.

### Hierarchical fields

| Field | Type | Description |
|-------|------|-------------|
| `tests` | `dict[str, object]` | Registry of populated diagnostics |
| `available` | `tuple[str, ...]` | Populated test names |
| `pretrend_f` | `PretrendFResult \| None` | `f_stat`, `p_value`, `df1`, `df2` |
| `tost` | `TostResult \| None` | Per-period TOST `pvals`, `periods`, `threshold`, `max_pval`, `all_pass` |
| `placebo` | `PlaceboResult \| None` | Placebo-window `estimate`, `se`, `p_value`, `equiv_p_value`, `period` |
| `honestdid` | `HonestDiDResult \| None` | Sensitivity arrays `M`, `ci_lower`, `ci_upper` |
| `options` | `dict` | Resolved diagnostic options |

### Legacy compatibility attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `pretrend_f_stat` | `float` | Wald F-statistic |
| `pretrend_f_pval` | `float` | p-value of F-test |
| `pretrend_df` | `tuple[int, int]` | Degrees of freedom |
| `equiv_results` | `pl.DataFrame \| None` | TOST results: `rel_time`, `tost_pval`, `bound`, `reject` |
| `placebo_results` | `pl.DataFrame \| None` | Placebo result: `period`, `estimate`, `se`, `pval`, `equiv_pval` |
| `honestdid_results` | `pl.DataFrame \| None` | Sensitivity: `M`, `ci_lower`, `ci_upper` |

### Fit-time diagnostics

```python
result = py2sdid.bjs_did(
    data, "y", "id", "t", "g",
    diagnostics="full",
    diagnostics_options={"delta": 0.10, "placebo_period": (-3, -1)},
)

result.diagnostics.tost.max_pval
result.diagnostics.placebo.equiv_p_value
```

Accepted diagnostic names are `pretrend_f`, `tost`, `placebo`, and
`honestdid`. `diagnostics=["placebo"]` requires
`diagnostics_options["placebo_period"]`.

---

## SE Methods

### did2s Influence Functions (default for `ts_did`)

GMM correction for the generated-regressor problem:

```
IF_i = (X2'X2)^{-1} [X2_i v_i - X2'X1 (X10'X10)^{-1} X10_i u_i]
V(beta) = sum_c (sum_{i in c} IF_i)(sum_{i in c} IF_i)'
```

### BJS Imputation SEs (default for `bjs_did`)

Equations 6, 8, 10 of Borusyak et al. (2024):

```
v*_it = -Z (Z0'Z0)^{-1} Z1' w1
Var = sum_c [sum_{i in c} v*_it (adj_it - tau_et)]^2
```

### Cluster Bootstrap

Available for either method via `bootstrap=True`. Resamples clusters with replacement.

---

## Internal Modules

| Module | Purpose |
|--------|---------|
| `panel.py` | Polars DataFrame -> PanelData |
| `first_stage.py` | Sparse OLS on untreated subsample |
| `effects.py` | Treatment effect computation |
| `inference.py` | did2s IF SEs, BJS SEs, bootstrap |
| `diagnostics.py` | Pre-trend F-test, TOST, placebo, HonestDiD |
| `plotting.py` | Matplotlib plots |
| `results.py` | Dataclasses |
| `linalg.py` | Sparse FE, robust solve |
| `_types.py` | EstimatorProtocol |
