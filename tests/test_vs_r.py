"""Validation tests: compare py2sdid against R did2s and didimputation.

Uses subprocess to call R (avoids rpy2 compatibility issues).
Requires: R with did2s, didimputation, fixest, data.table, jsonlite.
Skip with: pytest -k "not test_vs_r"
"""
import json
import subprocess
import tempfile

import numpy as np
import polars as pl
import pytest

from conftest import gen_data

# ---------------------------------------------------------------------------
# Check R availability
# ---------------------------------------------------------------------------

def _find_r():
    """Find R binary that has did2s and didimputation installed."""
    for r_bin in ["/usr/bin/R", "/opt/R/current/bin/R", "R"]:
        try:
            result = subprocess.run(
                [r_bin, "-e", 'library(did2s); library(didimputation); cat("OK")'],
                capture_output=True, text=True, timeout=30,
            )
            if "OK" in result.stdout:
                return r_bin
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


R_BIN = _find_r()
pytestmark = pytest.mark.skipif(R_BIN is None, reason="R with did2s/didimputation not found")


def _run_r(script: str) -> dict:
    """Run R script, return parsed JSON output."""
    result = subprocess.run(
        [R_BIN, "--vanilla", "--slave", "-e", script],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"R failed:\n{result.stderr}")
    # Extract JSON from output (last line)
    for line in reversed(result.stdout.strip().split("\n")):
        line = line.strip()
        if line.startswith("{") or line.startswith("["):
            return json.loads(line)
    raise RuntimeError(f"No JSON in R output:\n{result.stdout}")


def _write_shared_csv(df: pl.DataFrame) -> str:
    """Write Polars DataFrame to temp CSV, return path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    df.write_csv(tmp.name)
    return tmp.name


# ---------------------------------------------------------------------------
# Shared data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def shared_data():
    return gen_data(n=900, te1=2.0, te2=3.0, te_m1=0.0, te_m2=0.0,
                    seed=42, g1=2000, g2=2010, panel=(1990, 2020))


@pytest.fixture(scope="module")
def csv_path(shared_data):
    path = _write_shared_csv(shared_data)
    yield path
    import os; os.unlink(path)


# ---------------------------------------------------------------------------
# R runners
# ---------------------------------------------------------------------------

def _r_did2s_static(csv_path: str) -> dict:
    return _run_r(f"""
        library(did2s); library(jsonlite)
        df <- read.csv("{csv_path}")
        df$treat <- as.integer(df$g > 0 & df$year >= df$g)
        est <- did2s(df,
            yname="dep_var",
            first_stage = ~ 0 | unit + year,
            second_stage = ~ i(treat, ref=FALSE),
            treatment="treat", cluster_var="state", verbose=FALSE)
        cat(toJSON(list(
            att=unbox(coef(est)[[1]]),
            se=unbox(sqrt(diag(est$cov.scaled))[[1]])
        )))
    """)


def _r_didimputation_static(csv_path: str) -> dict:
    return _run_r(f"""
        library(didimputation); library(jsonlite)
        df <- read.csv("{csv_path}")
        est <- did_imputation(data=df, yname="dep_var", gname="g",
                              tname="year", idname="unit", cluster_var="state")
        cat(toJSON(list(
            att=unbox(est$estimate),
            se=unbox(est$std.error)
        )))
    """)


def _r_didimputation_es(csv_path: str) -> dict:
    return _run_r(f"""
        library(didimputation); library(jsonlite)
        df <- read.csv("{csv_path}")
        est <- did_imputation(data=df, yname="dep_var", gname="g",
                              tname="year", idname="unit",
                              horizon=TRUE, cluster_var="state")
        cat(toJSON(list(
            terms=as.numeric(est$term),
            estimates=est$estimate,
            ses=est$std.error
        )))
    """)


# ---------------------------------------------------------------------------
# Point estimate tests
# ---------------------------------------------------------------------------

class TestPointEstimates:
    """Point estimates should match R packages closely."""

    def test_static_att_vs_did2s(self, shared_data, csv_path):
        from py2sdid import ts_did
        py = ts_did(shared_data, yname="dep_var", idname="unit", tname="year",
                    gname="g", cluster_var="state", se=False, verbose=False)
        r = _r_did2s_static(csv_path)
        print(f"\n  Python ATT: {py.att_avg:.6f}")
        print(f"  R did2s ATT: {r['att']:.6f}")
        print(f"  Diff: {abs(py.att_avg - r['att']):.2e}")
        np.testing.assert_allclose(py.att_avg, r["att"], atol=0.05)

    def test_static_att_vs_didimputation(self, shared_data, csv_path):
        from py2sdid import bjs_did
        py = bjs_did(shared_data, yname="dep_var", idname="unit", tname="year",
                     gname="g", cluster_var="state", se=False, verbose=False)
        r = _r_didimputation_static(csv_path)
        print(f"\n  Python ATT: {py.att_avg:.6f}")
        print(f"  R didimputation ATT: {r['att']:.6f}")
        print(f"  Diff: {abs(py.att_avg - r['att']):.2e}")
        np.testing.assert_allclose(py.att_avg, r["att"], atol=0.05)

    def test_python_ts_bjs_identical(self, shared_data):
        """ts_did and bjs_did must produce identical point estimates."""
        from py2sdid import ts_did, bjs_did
        r1 = ts_did(shared_data, yname="dep_var", idname="unit", tname="year",
                     gname="g", se=False, verbose=False)
        r2 = bjs_did(shared_data, yname="dep_var", idname="unit", tname="year",
                      gname="g", se=False, verbose=False)
        np.testing.assert_allclose(r1.att_avg, r2.att_avg, atol=1e-12)

    def test_event_study_vs_didimputation(self, shared_data, csv_path):
        from py2sdid import bjs_did
        py = bjs_did(shared_data, yname="dep_var", idname="unit", tname="year",
                     gname="g", cluster_var="state",
                     se=False, verbose=False)
        r = _r_didimputation_es(csv_path)

        py_h = py.att_by_horizon
        n_compared = 0
        max_diff = 0.0
        for i, term in enumerate(r["terms"]):
            h = int(term)
            py_row = py_h.filter(pl.col("rel_time") == h)
            if len(py_row) == 0:
                continue
            py_est = py_row["estimate"][0]
            r_est = r["estimates"][i]
            diff = abs(py_est - r_est)
            max_diff = max(max_diff, diff)
            n_compared += 1
            print(f"  h={h:3d}: py={py_est:+.4f}  R={r_est:+.4f}  diff={diff:.4f}")
            np.testing.assert_allclose(py_est, r_est, atol=0.15,
                                       err_msg=f"Event-study h={h} differs")
        print(f"\n  Compared {n_compared} horizons, max diff: {max_diff:.4f}")


# ---------------------------------------------------------------------------
# Standard error tests
# ---------------------------------------------------------------------------

class TestStandardErrors:
    """SEs should be in the same ballpark (different formulas, same asymptotics)."""

    def test_se_vs_did2s(self, shared_data, csv_path):
        from py2sdid import ts_did
        py = ts_did(shared_data, yname="dep_var", idname="unit", tname="year",
                    gname="g", cluster_var="state", se=True, verbose=False)
        r = _r_did2s_static(csv_path)
        ratio = py.att_avg_se / r["se"]
        print(f"\n  Python SE: {py.att_avg_se:.6f}")
        print(f"  R did2s SE: {r['se']:.6f}")
        print(f"  Ratio: {ratio:.4f}")
        assert 0.5 < ratio < 2.0, f"SE ratio {ratio:.4f} outside [0.5, 2.0]"

    def test_se_vs_didimputation(self, shared_data, csv_path):
        from py2sdid import bjs_did
        py = bjs_did(shared_data, yname="dep_var", idname="unit", tname="year",
                     gname="g", cluster_var="state", se=True, verbose=False)
        r = _r_didimputation_static(csv_path)
        ratio = py.att_avg_se / r["se"]
        print(f"\n  Python SE: {py.att_avg_se:.6f}")
        print(f"  R didimputation SE: {r['se']:.6f}")
        print(f"  Ratio: {ratio:.4f}")
        assert 0.5 < ratio < 2.0, f"SE ratio {ratio:.4f} outside [0.5, 2.0]"

    def test_event_study_se_vs_didimputation(self, shared_data, csv_path):
        from py2sdid import bjs_did
        py = bjs_did(shared_data, yname="dep_var", idname="unit", tname="year",
                     gname="g", cluster_var="state",
                     se=True, verbose=False)
        r = _r_didimputation_es(csv_path)

        py_h = py.att_by_horizon
        ratios = []
        for i, term in enumerate(r["terms"]):
            h = int(term)
            py_row = py_h.filter(pl.col("rel_time") == h)
            if len(py_row) == 0:
                continue
            py_se = py_row["se"][0]
            r_se = r["ses"][i]
            if r_se > 0 and py_se is not None and py_se > 0:
                ratio = py_se / r_se
                ratios.append(ratio)
                print(f"  h={h:3d}: py_se={py_se:.4f}  r_se={r_se:.4f}  ratio={ratio:.3f}")

        if ratios:
            median_ratio = np.median(ratios)
            print(f"\n  Median SE ratio: {median_ratio:.4f}")
            assert 0.3 < median_ratio < 3.0, f"Median SE ratio {median_ratio:.4f} outside [0.3, 3.0]"


# ---------------------------------------------------------------------------
# Sanity: R packages agree with each other
# ---------------------------------------------------------------------------

class TestRSanity:

    def test_r_packages_agree_on_att(self, csv_path):
        r1 = _r_did2s_static(csv_path)
        r2 = _r_didimputation_static(csv_path)
        print(f"\n  R did2s ATT: {r1['att']:.6f}")
        print(f"  R didimputation ATT: {r2['att']:.6f}")
        np.testing.assert_allclose(r1["att"], r2["att"], atol=0.05)
