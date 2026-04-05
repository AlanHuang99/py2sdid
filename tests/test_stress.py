"""Stress tests: large panels, many cohorts, performance bounds."""
import time

import numpy as np
import polars as pl
import pytest

from conftest import gen_data


class TestLargePanel:
    """Test on larger datasets to verify scalability and correctness."""

    def test_large_n(self):
        """N=5000 units, 31 periods."""
        from py2sdid import ts_did
        df = gen_data(n=5000, te1=2.0, te2=2.0, seed=1)
        t0 = time.perf_counter()
        result = ts_did(df, yname="dep_var", idname="unit", tname="year",
                        gname="g", cluster_var="state", verbose=False)
        elapsed = time.perf_counter() - t0
        assert abs(result.att_avg - 2.0) < 0.3
        assert result.att_avg_se is not None
        assert result.att_avg_se > 0
        assert len(result.event_study) > 30  # pre + post periods
        print(f"\n  N=5000: ATT={result.att_avg:.4f}, SE={result.att_avg_se:.4f}, "
              f"time={elapsed:.2f}s")

    def test_large_n_bjs(self):
        """N=5000 with BJS estimator."""
        from py2sdid import bjs_did
        df = gen_data(n=5000, te1=2.0, te2=2.0, seed=1)
        t0 = time.perf_counter()
        result = bjs_did(df, yname="dep_var", idname="unit", tname="year",
                         gname="g", cluster_var="state", verbose=False)
        elapsed = time.perf_counter() - t0
        assert abs(result.att_avg - 2.0) < 0.3
        print(f"\n  N=5000 BJS: ATT={result.att_avg:.4f}, SE={result.att_avg_se:.4f}, "
              f"time={elapsed:.2f}s")

    def test_point_estimates_match_large(self):
        """Point estimates must match on large panel."""
        from py2sdid import ts_did, bjs_did
        df = gen_data(n=3000, te1=2.5, te2=1.5, seed=7)
        r1 = ts_did(df, yname="dep_var", idname="unit", tname="year",
                     gname="g", se=False, verbose=False)
        r2 = bjs_did(df, yname="dep_var", idname="unit", tname="year",
                      gname="g", se=False, verbose=False)
        np.testing.assert_allclose(r1.att_avg, r2.att_avg, atol=1e-12)
        # Per-period too
        es1 = r1.event_study["estimate"].to_numpy()
        es2 = r2.event_study["estimate"].to_numpy()
        np.testing.assert_allclose(es1, es2, atol=1e-12)


class TestManyCohorts:
    """Test with more than 2 treatment cohorts."""

    def _gen_multi_cohort(self, n=1200, seed=42):
        """Generate panel with 4 cohorts + never-treated."""
        rng = np.random.default_rng(seed)
        years = list(range(1990, 2021))
        T = len(years)
        cohorts = {1995: 200, 2000: 200, 2005: 200, 2010: 200}
        te = {1995: 1.0, 2000: 2.0, 2005: 3.0, 2010: 4.0}
        unit_g = []
        for g, cnt in cohorts.items():
            unit_g.extend([g] * cnt)
        unit_g.extend([0] * (n - len(unit_g)))
        unit_g = np.array(unit_g[:n])
        unit_fe = rng.normal(0, 1, size=n)
        year_fe = rng.normal(0, 0.5, size=T)
        unit_state = rng.integers(1, 51, size=n)
        rows = []
        for i in range(n):
            g = int(unit_g[i])
            for t_idx, yr in enumerate(years):
                treat = int(g > 0 and yr >= g)
                te_val = te.get(g, 0.0) if treat else 0.0
                dep_var = unit_fe[i] + year_fe[t_idx] + te_val + rng.normal(0, 1)
                rows.append({"unit": i+1, "year": yr, "state": int(unit_state[i]),
                             "g": g, "dep_var": dep_var})
        return pl.DataFrame(rows)

    def test_four_cohorts(self):
        from py2sdid import ts_did, bjs_did
        df = self._gen_multi_cohort()
        r1 = ts_did(df, yname="dep_var", idname="unit", tname="year",
                     gname="g", cluster_var="state", verbose=False)
        r2 = bjs_did(df, yname="dep_var", idname="unit", tname="year",
                      gname="g", cluster_var="state", verbose=False)
        # Point estimates identical
        np.testing.assert_allclose(r1.att_avg, r2.att_avg, atol=1e-12)
        # ATT should be weighted average of 1,2,3,4 ~ 2.5
        assert 1.5 < r1.att_avg < 3.5
        assert r1.att_avg_se > 0
        print(f"\n  4 cohorts: ATT={r1.att_avg:.4f}, SE={r1.att_avg_se:.4f}")

    def test_four_cohorts_diagnostics(self):
        from py2sdid import ts_did
        df = self._gen_multi_cohort()
        result = ts_did(df, yname="dep_var", idname="unit", tname="year",
                        gname="g", cluster_var="state", verbose=False)
        diag = result.diagnose()
        # F-test should not reject (no pre-trends in DGP)
        assert diag.pretrend_f_pval > 0.01
        assert diag.equiv_all_pass is True


class TestBootstrap:
    """Test bootstrap SE computation."""

    def test_bootstrap_runs(self):
        from py2sdid import ts_did
        df = gen_data(n=300, te1=3.0, te2=3.0, seed=99)
        result = ts_did(df, yname="dep_var", idname="unit", tname="year",
                        gname="g", cluster_var="state", bootstrap=True,
                        n_bootstraps=30, seed=42, n_jobs=1, verbose=False)
        assert result.att_avg_se > 0
        assert result.boot_dist is not None
        assert result.boot_dist.shape[0] == 30

    def test_bootstrap_parallel(self):
        from py2sdid import ts_did
        df = gen_data(n=300, te1=3.0, te2=3.0, seed=99)
        result = ts_did(df, yname="dep_var", idname="unit", tname="year",
                        gname="g", cluster_var="state", bootstrap=True,
                        n_bootstraps=20, seed=42, n_jobs=2, verbose=False)
        assert result.att_avg_se > 0

    def test_bootstrap_se_reasonable(self):
        """Bootstrap SE should be in same ballpark as analytic."""
        from py2sdid import ts_did
        df = gen_data(n=600, te1=3.0, te2=3.0, seed=42)
        r_analytic = ts_did(df, yname="dep_var", idname="unit", tname="year",
                            gname="g", cluster_var="state", verbose=False)
        r_boot = ts_did(df, yname="dep_var", idname="unit", tname="year",
                        gname="g", cluster_var="state", bootstrap=True,
                        n_bootstraps=100, seed=42, n_jobs=1, verbose=False)
        ratio = r_boot.att_avg_se / r_analytic.att_avg_se
        print(f"\n  Boot/Analytic SE ratio: {ratio:.3f}")
        assert 0.3 < ratio < 3.0
