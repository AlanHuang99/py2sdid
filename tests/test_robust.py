"""Robustness tests: edge cases, unusual panel structures, input validation."""
import numpy as np
import polars as pl
import pytest

from conftest import gen_data


class TestEdgeCases:
    """Edge cases that should work correctly."""

    def test_single_cohort(self):
        """Only one treatment cohort + never-treated."""
        from py2sdid import ts_did
        df = gen_data(n=600, te1=3.0, te2=3.0, g1=2000, g2=0, seed=42)
        result = ts_did(df, yname="dep_var", idname="unit", tname="year",
                        gname="g", cluster_var="state", verbose=False)
        assert abs(result.att_avg - 3.0) < 0.5
        assert len(result.event_study) > 0

    def test_all_treated_same_time(self):
        """All treated units in the same cohort (no staggering)."""
        from py2sdid import ts_did
        df = gen_data(n=600, te1=2.0, te2=2.0, g1=2005, g2=2005, seed=42)
        result = ts_did(df, yname="dep_var", idname="unit", tname="year",
                        gname="g", se=False, verbose=False)
        assert abs(result.att_avg - 2.0) < 0.5

    def test_null_gname_for_never_treated(self):
        """Never-treated encoded as null (not 0)."""
        from py2sdid import ts_did
        df = gen_data(n=600, te1=3.0, te2=3.0)
        df = df.with_columns(
            pl.when(pl.col("g") == 0).then(None).otherwise(pl.col("g")).alias("g")
        )
        result = ts_did(df, yname="dep_var", idname="unit", tname="year",
                        gname="g", se=False, verbose=False)
        assert abs(result.att_avg - 3.0) < 0.5

    def test_no_se(self):
        """se=False should skip inference entirely."""
        from py2sdid import ts_did
        df = gen_data(n=300, te1=3.0, te2=3.0)
        result = ts_did(df, yname="dep_var", idname="unit", tname="year",
                        gname="g", se=False, verbose=False)
        assert result.att_avg_se is None
        assert result.vcov is None
        assert result.att_avg_ci is None

    def test_with_weights(self):
        """Uniform weights should give same result as no weights."""
        from py2sdid import ts_did
        df = gen_data(n=600, te1=3.0, te2=3.0, seed=42)
        r_no_w = ts_did(df, yname="dep_var", idname="unit", tname="year",
                        gname="g", se=False, verbose=False)
        df_w = df.with_columns(w=pl.lit(1.0))
        r_w = ts_did(df_w, yname="dep_var", idname="unit", tname="year",
                     gname="g", wname="w", se=False, verbose=False)
        np.testing.assert_allclose(r_no_w.att_avg, r_w.att_avg, atol=1e-10)

    def test_with_covariates_preserves_names(self):
        """Covariate names should appear in result."""
        from py2sdid import ts_did
        df = gen_data(n=600, te1=3.0, te2=3.0)
        df = df.with_columns(
            x1=pl.col("year").cast(pl.Float64) / 100,
            x2=pl.col("state").cast(pl.Float64) / 50,
        )
        result = ts_did(df, yname="dep_var", idname="unit", tname="year",
                        gname="g", xformla=["x1", "x2"], se=False, verbose=False)
        assert result.covariate_names == ["x1", "x2"]
        assert result.beta is not None
        assert len(result.beta) == 2

    def test_cluster_at_unit_level(self):
        """Default clustering (no cluster_var) should cluster at unit."""
        from py2sdid import ts_did
        df = gen_data(n=300, te1=3.0, te2=3.0)
        result = ts_did(df, yname="dep_var", idname="unit", tname="year",
                        gname="g", verbose=False)
        # 300 units = 300 clusters
        assert len(result.panel.cluster_map) == 300
        assert result.att_avg_se > 0


class TestInputValidation:
    """Tests for input validation and error messages."""

    def test_missing_column_raises(self):
        from py2sdid import ts_did
        df = gen_data(n=100)
        with pytest.raises(ValueError, match="Missing columns"):
            ts_did(df, yname="nonexistent", idname="unit", tname="year",
                   gname="g", verbose=False)

    def test_gname_varying_raises(self):
        from py2sdid import ts_did
        df = gen_data(n=300)
        bad = df.with_row_index("_idx").with_columns(
            pl.when((pl.col("unit") == 1) & (pl.col("year") > 2005))
            .then(2010).otherwise(pl.col("g")).alias("g")
        ).drop("_idx")
        with pytest.raises(ValueError, match="constant within each unit"):
            ts_did(bad, yname="dep_var", idname="unit", tname="year",
                   gname="g", verbose=False)

    def test_float_tname_with_integers_works(self):
        """Float tname that contains integer values should work."""
        from py2sdid import ts_did
        df = gen_data(n=300)
        df = df.with_columns(pl.col("year").cast(pl.Float64))
        result = ts_did(df, yname="dep_var", idname="unit", tname="year",
                        gname="g", se=False, verbose=False)
        assert result.att_avg is not None

    def test_float_tname_with_decimals_raises(self):
        """Float tname with non-integer values should raise."""
        from py2sdid import ts_did
        df = gen_data(n=300)
        df = df.with_columns((pl.col("year").cast(pl.Float64) + 0.5).alias("year"))
        with pytest.raises(ValueError, match="integer-valued"):
            ts_did(df, yname="dep_var", idname="unit", tname="year",
                   gname="g", verbose=False)


class TestResultConsistency:
    """Verify internal consistency of result objects."""

    def test_event_study_covers_all_periods(self):
        """event_study should have pre + post relative times."""
        from py2sdid import ts_did
        df = gen_data(n=600, te1=3.0, te2=3.0)
        result = ts_did(df, yname="dep_var", idname="unit", tname="year",
                        gname="g", se=False, verbose=False)
        es = result.event_study
        rel_times = es["rel_time"].to_numpy()
        assert (rel_times < 0).any(), "Should have pre-treatment periods"
        assert (rel_times >= 0).any(), "Should have post-treatment periods"
        # Should be sorted
        assert (np.diff(rel_times) > 0).all(), "rel_time should be sorted ascending"

    def test_att_by_horizon_is_post_only(self):
        from py2sdid import ts_did
        df = gen_data(n=600, te1=3.0, te2=3.0)
        result = ts_did(df, yname="dep_var", idname="unit", tname="year",
                        gname="g", se=False, verbose=False)
        post = result.att_by_horizon
        assert (post["rel_time"] >= 0).all()

    def test_pretrend_tests_is_pre_only(self):
        from py2sdid import ts_did
        df = gen_data(n=600, te1=3.0, te2=3.0)
        result = ts_did(df, yname="dep_var", idname="unit", tname="year",
                        gname="g", se=False, verbose=False)
        pre = result.pretrend_tests
        assert pre is not None
        assert (pre["rel_time"] < 0).all()

    def test_summary_is_string(self):
        from py2sdid import ts_did
        df = gen_data(n=300, te1=3.0, te2=3.0)
        result = ts_did(df, yname="dep_var", idname="unit", tname="year",
                        gname="g", verbose=False)
        s = result.summary()
        assert isinstance(s, str)
        assert "ATT" in s
        assert "py2sdid" in s
        assert "Treated:" in s
        assert "units" in s

    def test_vcov_shape_matches_event_study(self):
        from py2sdid import ts_did
        df = gen_data(n=600, te1=3.0, te2=3.0)
        result = ts_did(df, yname="dep_var", idname="unit", tname="year",
                        gname="g", cluster_var="state", verbose=False)
        n_periods = len(result.event_study)
        assert result.vcov.shape == (n_periods, n_periods)

    def test_y_hat_shape(self):
        from py2sdid import ts_did
        df = gen_data(n=300, te1=3.0, te2=3.0)
        result = ts_did(df, yname="dep_var", idname="unit", tname="year",
                        gname="g", se=False, verbose=False)
        assert result.y_hat.shape == (result.panel.n_obs,)

    def test_diagnostics_fields(self):
        from py2sdid import ts_did
        df = gen_data(n=600, te1=3.0, te2=3.0)
        result = ts_did(df, yname="dep_var", idname="unit", tname="year",
                        gname="g", cluster_var="state", verbose=False)
        diag = result.diagnose()
        assert diag.pretrend_f_stat >= 0
        assert 0 <= diag.pretrend_f_pval <= 1
        assert diag.equiv_results is not None
        assert diag.equiv_max_pval is not None
        assert isinstance(diag.equiv_all_pass, bool)
        assert diag.honestdid_results is not None
        s = diag.summary()
        assert "Pre-trend" in s
        assert "Equivalence" in s
        assert "PASS" in s or "FAIL" in s
