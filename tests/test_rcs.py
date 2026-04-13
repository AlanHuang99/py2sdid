"""Comprehensive tests for repeated cross-section (RCS) support.

Organized in three tiers:
- TestSmoke: basic correctness (both individual-level and aggregated RCS)
- TestRobust: edge cases, validation, options
- TestStress: scale, inference, dynamic effects, diagnostics
"""
import numpy as np
import polars as pl
import pytest

from conftest import gen_rcs_data, gen_agg_rcs_data, gen_data


# ===================================================================
# Smoke Tests — basic correctness
# ===================================================================

class TestSmoke:
    """Core functionality for both individual-level and aggregated RCS."""

    # --- Individual-level RCS ---

    def test_ts_did_individual_rcs(self):
        df = gen_rcs_data(n_groups=20, n_individuals_per_group_period=15,
                          panel=(2000, 2006), g1=2003, te=3.0, seed=42)
        from py2sdid import ts_did
        result = ts_did(df, yname="dep_var", idname="individual_id", tname="year",
                        gname="g", dataset_type="rcs", groupname="group",
                        se=False, verbose=False)
        assert result.method == "ts_did"
        assert result.panel.is_rcs
        assert abs(result.att_avg - 3.0) < 0.5

    def test_bjs_did_individual_rcs(self):
        df = gen_rcs_data(n_groups=20, n_individuals_per_group_period=15,
                          panel=(2000, 2006), g1=2003, te=3.0, seed=42)
        from py2sdid import bjs_did
        result = bjs_did(df, yname="dep_var", idname="individual_id", tname="year",
                         gname="g", dataset_type="rcs", groupname="group",
                         se=False, verbose=False)
        assert result.method == "bjs_did"
        assert result.panel.is_rcs
        assert abs(result.att_avg - 3.0) < 0.5

    # --- Aggregated RCS ---

    def test_ts_did_aggregated_rcs(self):
        df = gen_agg_rcs_data(n_groups=30, panel=(2000, 2008), g1=2004,
                              te=2.0, seed=42)
        from py2sdid import ts_did
        result = ts_did(df, yname="dep_var", idname="group", tname="year",
                        gname="g", dataset_type="rcs",
                        se=False, verbose=False)
        assert result.panel.is_rcs
        assert abs(result.att_avg - 2.0) < 0.5

    def test_bjs_did_aggregated_rcs(self):
        df = gen_agg_rcs_data(n_groups=30, panel=(2000, 2008), g1=2004,
                              te=2.0, seed=42)
        from py2sdid import bjs_did
        result = bjs_did(df, yname="dep_var", idname="group", tname="year",
                         gname="g", dataset_type="rcs",
                         se=False, verbose=False)
        assert result.panel.is_rcs
        assert abs(result.att_avg - 2.0) < 0.5

    # --- Point estimate agreement ---

    def test_individual_rcs_ts_bjs_match(self):
        df = gen_rcs_data(n_groups=30, n_individuals_per_group_period=20,
                          panel=(2000, 2008), g1=2004, te=2.0, seed=99)
        from py2sdid import ts_did, bjs_did
        r1 = ts_did(df, yname="dep_var", idname="individual_id", tname="year",
                    gname="g", dataset_type="rcs", groupname="group",
                    se=False, verbose=False)
        r2 = bjs_did(df, yname="dep_var", idname="individual_id", tname="year",
                     gname="g", dataset_type="rcs", groupname="group",
                     se=False, verbose=False)
        np.testing.assert_allclose(r1.att_avg, r2.att_avg, atol=1e-12)

    def test_aggregated_rcs_ts_bjs_match(self):
        df = gen_agg_rcs_data(n_groups=30, panel=(2000, 2008), g1=2004,
                              te=2.0, seed=99)
        from py2sdid import ts_did, bjs_did
        r1 = ts_did(df, yname="dep_var", idname="group", tname="year",
                    gname="g", dataset_type="rcs", se=False, verbose=False)
        r2 = bjs_did(df, yname="dep_var", idname="group", tname="year",
                     gname="g", dataset_type="rcs", se=False, verbose=False)
        np.testing.assert_allclose(r1.att_avg, r2.att_avg, atol=1e-12)

    # --- Event study ---

    def test_individual_rcs_event_study(self):
        df = gen_rcs_data(n_groups=50, n_individuals_per_group_period=30,
                          panel=(2000, 2010), g1=2005, te=3.0, seed=42)
        from py2sdid import ts_did
        result = ts_did(df, yname="dep_var", idname="individual_id", tname="year",
                        gname="g", dataset_type="rcs", groupname="group",
                        se=False, verbose=False)
        es = result.event_study
        assert len(es) > 0
        assert (es["rel_time"] < 0).any()
        assert (es["rel_time"] >= 0).any()
        pre = es.filter(pl.col("rel_time") < 0)
        assert abs(pre["estimate"].mean()) < 0.5
        post = es.filter(pl.col("rel_time") >= 0)
        assert abs(post["estimate"].mean() - 3.0) < 0.5

    def test_aggregated_rcs_event_study(self):
        df = gen_agg_rcs_data(n_groups=50, panel=(2000, 2010), g1=2005,
                              te=3.0, seed=42)
        from py2sdid import ts_did
        result = ts_did(df, yname="dep_var", idname="group", tname="year",
                        gname="g", dataset_type="rcs", se=False, verbose=False)
        es = result.event_study
        assert (es["rel_time"] < 0).any()
        assert (es["rel_time"] >= 0).any()
        post = es.filter(pl.col("rel_time") >= 0)
        assert abs(post["estimate"].mean() - 3.0) < 0.5


# ===================================================================
# Robust Tests — edge cases, validation, options
# ===================================================================

class TestRobust:
    """Edge cases, validation rules, and optional parameters."""

    # --- Validation ---

    def test_invalid_dataset_type_raises(self):
        from py2sdid import ts_did
        df = gen_data(n=100, te1=2.0, te2=2.0, seed=42)
        with pytest.raises(ValueError, match="dataset_type must be"):
            ts_did(df, yname="dep_var", idname="unit", tname="year",
                   gname="g", dataset_type="invalid", se=False, verbose=False)

    def test_panel_with_groupname_raises(self):
        from py2sdid import ts_did
        df = gen_data(n=100, te1=2.0, te2=2.0, seed=42)
        with pytest.raises(ValueError, match="groupname must not be provided"):
            ts_did(df, yname="dep_var", idname="unit", tname="year",
                   gname="g", dataset_type="panel", groupname="state",
                   se=False, verbose=False)

    def test_rcs_gname_varying_within_group_raises(self):
        from py2sdid import ts_did
        df = gen_rcs_data(n_groups=10, n_individuals_per_group_period=10,
                          panel=(2000, 2005), g1=2003, te=2.0, seed=42)
        # Corrupt one group's cohort for some individuals
        bad = df.with_columns(
            pl.when((pl.col("group") == 1) & (pl.col("year") > 2003))
            .then(2004)
            .otherwise(pl.col("g"))
            .alias("g")
        )
        with pytest.raises(ValueError, match="constant within each"):
            ts_did(bad, yname="dep_var", idname="individual_id", tname="year",
                   gname="g", dataset_type="rcs", groupname="group",
                   se=False, verbose=False)

    def test_missing_groupname_column_raises(self):
        from py2sdid import ts_did
        df = gen_rcs_data(n_groups=10, n_individuals_per_group_period=10,
                          panel=(2000, 2005), g1=2003, te=2.0, seed=42)
        with pytest.raises(ValueError, match="Missing columns"):
            ts_did(df, yname="dep_var", idname="individual_id", tname="year",
                   gname="g", dataset_type="rcs", groupname="nonexistent",
                   se=False, verbose=False)

    # --- Covariates ---

    def test_individual_rcs_with_covariates(self):
        from py2sdid import ts_did
        df = gen_rcs_data(n_groups=30, n_individuals_per_group_period=20,
                          panel=(2000, 2008), g1=2004, te=2.0, seed=42)
        df = df.with_columns(x1=pl.col("year").cast(pl.Float64) / 100)
        result = ts_did(df, yname="dep_var", idname="individual_id", tname="year",
                        gname="g", dataset_type="rcs", groupname="group",
                        xformla=["x1"], se=False, verbose=False)
        assert result.beta is not None
        assert abs(result.att_avg - 2.0) < 0.5

    def test_aggregated_rcs_with_covariates(self):
        from py2sdid import ts_did
        df = gen_agg_rcs_data(n_groups=30, panel=(2000, 2008), g1=2004,
                              te=2.0, seed=42)
        df = df.with_columns(x1=pl.col("year").cast(pl.Float64) / 100)
        result = ts_did(df, yname="dep_var", idname="group", tname="year",
                        gname="g", dataset_type="rcs", xformla=["x1"],
                        se=False, verbose=False)
        assert result.beta is not None

    # --- Weights ---

    def test_individual_rcs_with_weights(self):
        from py2sdid import ts_did
        df = gen_rcs_data(n_groups=30, n_individuals_per_group_period=20,
                          panel=(2000, 2008), g1=2004, te=2.0, seed=42)
        df = df.with_columns(w=pl.lit(1.0))
        result = ts_did(df, yname="dep_var", idname="individual_id", tname="year",
                        gname="g", dataset_type="rcs", groupname="group",
                        wname="w", se=False, verbose=False)
        assert abs(result.att_avg - 2.0) < 0.5

    # --- Cluster override ---

    def test_individual_rcs_cluster_override(self):
        from py2sdid import ts_did
        df = gen_rcs_data(n_groups=20, n_individuals_per_group_period=15,
                          panel=(2000, 2006), g1=2003, te=2.0, seed=42)
        # Add a higher-level cluster (e.g. region = group mod 4)
        df = df.with_columns(
            region=(pl.col("group") % 4).alias("region")
        )
        result = ts_did(df, yname="dep_var", idname="individual_id", tname="year",
                        gname="g", dataset_type="rcs", groupname="group",
                        cluster_var="region", se=True, verbose=False)
        n_clusters = len(result.panel.cluster_map)
        assert n_clusters == 4

    # --- Cluster defaults ---

    def test_individual_rcs_default_cluster_is_group(self):
        from py2sdid import ts_did
        df = gen_rcs_data(n_groups=20, n_individuals_per_group_period=15,
                          panel=(2000, 2006), g1=2003, te=2.0, seed=42)
        result = ts_did(df, yname="dep_var", idname="individual_id", tname="year",
                        gname="g", dataset_type="rcs", groupname="group",
                        se=True, verbose=False)
        n_clusters = len(result.panel.cluster_map)
        assert n_clusters == 20  # = n_groups, not n_individuals

    def test_aggregated_rcs_default_cluster_is_idname(self):
        from py2sdid import ts_did
        df = gen_agg_rcs_data(n_groups=30, panel=(2000, 2008), g1=2004,
                              te=2.0, seed=42)
        result = ts_did(df, yname="dep_var", idname="group", tname="year",
                        gname="g", dataset_type="rcs", se=True, verbose=False)
        n_clusters = len(result.panel.cluster_map)
        assert n_clusters == 30  # = n_groups

    # --- Null gname ---

    def test_rcs_null_gname_never_treated(self):
        from py2sdid import ts_did
        df = gen_rcs_data(n_groups=20, n_individuals_per_group_period=15,
                          panel=(2000, 2006), g1=2003, te=2.0, seed=42)
        df = df.with_columns(
            pl.when(pl.col("g") == 0).then(None).otherwise(pl.col("g")).alias("g")
        )
        result = ts_did(df, yname="dep_var", idname="individual_id", tname="year",
                        gname="g", dataset_type="rcs", groupname="group",
                        se=False, verbose=False)
        assert abs(result.att_avg - 2.0) < 0.5

    # --- Multiple cohorts ---

    def test_individual_rcs_two_cohorts(self):
        from py2sdid import ts_did
        df = gen_rcs_data(n_groups=40, n_individuals_per_group_period=20,
                          panel=(2000, 2010), g1=2004, g2=2007, te=3.0, seed=42)
        result = ts_did(df, yname="dep_var", idname="individual_id", tname="year",
                        gname="g", dataset_type="rcs", groupname="group",
                        se=False, verbose=False)
        assert abs(result.att_avg - 3.0) < 0.5
        n_cohorts = len([c for c in result.panel.cohort_sizes if c != 0])
        assert n_cohorts == 2

    # --- Summary output ---

    def test_individual_rcs_summary(self):
        from py2sdid import ts_did
        df = gen_rcs_data(n_groups=20, n_individuals_per_group_period=15,
                          panel=(2000, 2006), g1=2003, te=2.0, seed=42)
        result = ts_did(df, yname="dep_var", idname="individual_id", tname="year",
                        gname="g", dataset_type="rcs", groupname="group",
                        se=False, verbose=False)
        s = result.summary()
        assert "repeated cross-section" in s
        assert "G=" in s

    def test_aggregated_rcs_summary(self):
        from py2sdid import ts_did
        df = gen_agg_rcs_data(n_groups=20, panel=(2000, 2006), g1=2003,
                              te=2.0, seed=42)
        result = ts_did(df, yname="dep_var", idname="group", tname="year",
                        gname="g", dataset_type="rcs", se=False, verbose=False)
        s = result.summary()
        assert "repeated cross-section" in s

    # --- FE level counts ---

    def test_individual_rcs_fe_levels_equal_groups(self):
        from py2sdid import ts_did
        df = gen_rcs_data(n_groups=25, n_individuals_per_group_period=10,
                          panel=(2000, 2005), g1=2003, te=2.0, seed=42)
        result = ts_did(df, yname="dep_var", idname="individual_id", tname="year",
                        gname="g", dataset_type="rcs", groupname="group",
                        se=False, verbose=False)
        assert result.panel.n_fe_levels == 25
        assert result.panel.n_units > 25  # many more individuals than groups

    def test_aggregated_rcs_fe_levels_equal_groups(self):
        from py2sdid import ts_did
        df = gen_agg_rcs_data(n_groups=25, panel=(2000, 2005), g1=2003,
                              te=2.0, seed=42)
        result = ts_did(df, yname="dep_var", idname="group", tname="year",
                        gname="g", dataset_type="rcs", se=False, verbose=False)
        assert result.panel.n_fe_levels == 25
        assert result.panel.n_units == 25  # idname IS the group

    # --- Backward compatibility ---

    def test_panel_mode_unchanged(self):
        from py2sdid import ts_did
        df = gen_data(n=600, te1=3.0, te2=3.0, seed=42)
        result = ts_did(df, yname="dep_var", idname="unit", tname="year",
                        gname="g", cluster_var="state", se=False, verbose=False)
        assert not result.panel.is_rcs
        assert result.panel.n_fe_levels == result.panel.n_units
        assert abs(result.att_avg - 3.0) < 0.5

    def test_panel_default_dataset_type(self):
        """dataset_type defaults to 'panel' — existing code unchanged."""
        from py2sdid import ts_did
        df = gen_data(n=300, te1=3.0, te2=3.0, seed=42)
        result = ts_did(df, yname="dep_var", idname="unit", tname="year",
                        gname="g", se=False, verbose=False)
        assert not result.panel.is_rcs


# ===================================================================
# Stress Tests — scale, inference, dynamic effects
# ===================================================================

class TestStress:
    """Larger-scale tests and inference verification."""

    # --- Standard errors ---

    def test_individual_rcs_ts_did_se(self):
        df = gen_rcs_data(n_groups=50, n_individuals_per_group_period=30,
                          panel=(2000, 2010), g1=2005, te=3.0, seed=42)
        from py2sdid import ts_did
        result = ts_did(df, yname="dep_var", idname="individual_id", tname="year",
                        gname="g", dataset_type="rcs", groupname="group",
                        se=True, verbose=False)
        assert result.att_avg_se is not None
        assert result.att_avg_se > 0
        assert result.att_avg_ci is not None
        lo, hi = result.att_avg_ci
        assert lo < result.att_avg < hi

    def test_individual_rcs_bjs_did_se(self):
        df = gen_rcs_data(n_groups=50, n_individuals_per_group_period=30,
                          panel=(2000, 2010), g1=2005, te=3.0, seed=42)
        from py2sdid import bjs_did
        result = bjs_did(df, yname="dep_var", idname="individual_id", tname="year",
                         gname="g", dataset_type="rcs", groupname="group",
                         se=True, verbose=False)
        assert result.att_avg_se is not None
        assert result.att_avg_se > 0

    def test_aggregated_rcs_ts_did_se(self):
        df = gen_agg_rcs_data(n_groups=50, panel=(2000, 2010), g1=2005,
                              te=3.0, seed=42)
        from py2sdid import ts_did
        result = ts_did(df, yname="dep_var", idname="group", tname="year",
                        gname="g", dataset_type="rcs", se=True, verbose=False)
        assert result.att_avg_se is not None
        assert result.att_avg_se > 0

    def test_aggregated_rcs_bjs_did_se(self):
        df = gen_agg_rcs_data(n_groups=50, panel=(2000, 2010), g1=2005,
                              te=3.0, seed=42)
        from py2sdid import bjs_did
        result = bjs_did(df, yname="dep_var", idname="group", tname="year",
                         gname="g", dataset_type="rcs", se=True, verbose=False)
        assert result.att_avg_se is not None
        assert result.att_avg_se > 0

    # --- CI coverage ---

    def test_individual_rcs_ci_covers_true(self):
        df = gen_rcs_data(n_groups=50, n_individuals_per_group_period=40,
                          panel=(2000, 2010), g1=2005, te=3.0, seed=99)
        from py2sdid import ts_did
        result = ts_did(df, yname="dep_var", idname="individual_id", tname="year",
                        gname="g", dataset_type="rcs", groupname="group",
                        se=True, verbose=False)
        lo, hi = result.att_avg_ci
        assert lo < 3.0 < hi

    def test_aggregated_rcs_ci_covers_true(self):
        df = gen_agg_rcs_data(n_groups=50, panel=(2000, 2010), g1=2005,
                              te=3.0, seed=99)
        from py2sdid import ts_did
        result = ts_did(df, yname="dep_var", idname="group", tname="year",
                        gname="g", dataset_type="rcs", se=True, verbose=False)
        lo, hi = result.att_avg_ci
        assert lo < 3.0 < hi

    # --- Dynamic effects ---

    def test_individual_rcs_dynamic_effects(self):
        df = gen_rcs_data(n_groups=50, n_individuals_per_group_period=30,
                          panel=(2000, 2010), g1=2005, te=1.0, te_m=0.5,
                          seed=42)
        from py2sdid import ts_did
        result = ts_did(df, yname="dep_var", idname="individual_id", tname="year",
                        gname="g", dataset_type="rcs", groupname="group",
                        se=False, verbose=False)
        es = result.event_study
        att_0 = es.filter(pl.col("rel_time") == 0)["estimate"][0]
        att_5 = es.filter(pl.col("rel_time") == 5)["estimate"][0]
        assert att_5 > att_0 + 1.0  # slope = 0.5/yr, so +2.5 over 5 years

    # --- Large-scale ---

    def test_large_individual_rcs(self):
        df = gen_rcs_data(n_groups=100, n_individuals_per_group_period=50,
                          panel=(2000, 2015), g1=2008, te=2.0, seed=42)
        from py2sdid import ts_did
        result = ts_did(df, yname="dep_var", idname="individual_id", tname="year",
                        gname="g", dataset_type="rcs", groupname="group",
                        se=True, verbose=False)
        assert abs(result.att_avg - 2.0) < 0.3
        assert result.att_avg_se > 0

    # --- Bootstrap ---

    def test_individual_rcs_bootstrap(self):
        df = gen_rcs_data(n_groups=20, n_individuals_per_group_period=15,
                          panel=(2000, 2006), g1=2003, te=2.0, seed=42)
        from py2sdid import ts_did
        result = ts_did(df, yname="dep_var", idname="individual_id", tname="year",
                        gname="g", dataset_type="rcs", groupname="group",
                        bootstrap=True, n_bootstraps=50, seed=42, verbose=False)
        assert result.att_avg_se is not None
        assert result.att_avg_se > 0
        assert result.boot_dist is not None

    # --- Diagnostics ---

    def test_individual_rcs_diagnostics(self):
        df = gen_rcs_data(n_groups=50, n_individuals_per_group_period=30,
                          panel=(2000, 2010), g1=2005, te=3.0, seed=42)
        from py2sdid import ts_did
        result = ts_did(df, yname="dep_var", idname="individual_id", tname="year",
                        gname="g", dataset_type="rcs", groupname="group",
                        se=True, verbose=False)
        diag = result.diagnose()
        assert diag.pretrend_f_stat is not None
        assert diag.pretrend_f_pval > 0
        s = diag.summary()
        assert "Pre-trend" in s

    # --- Vcov shape ---

    def test_individual_rcs_vcov_shape(self):
        df = gen_rcs_data(n_groups=30, n_individuals_per_group_period=20,
                          panel=(2000, 2008), g1=2004, te=2.0, seed=42)
        from py2sdid import ts_did
        result = ts_did(df, yname="dep_var", idname="individual_id", tname="year",
                        gname="g", dataset_type="rcs", groupname="group",
                        se=True, verbose=False)
        n_h = len(result.event_study)
        assert result.vcov.shape == (n_h, n_h)

    # --- att_by_horizon and pretrend_tests properties ---

    def test_individual_rcs_properties(self):
        df = gen_rcs_data(n_groups=30, n_individuals_per_group_period=20,
                          panel=(2000, 2008), g1=2004, te=2.0, seed=42)
        from py2sdid import ts_did
        result = ts_did(df, yname="dep_var", idname="individual_id", tname="year",
                        gname="g", dataset_type="rcs", groupname="group",
                        se=False, verbose=False)
        post = result.att_by_horizon
        assert (post["rel_time"] >= 0).all()
        pre = result.pretrend_tests
        assert pre is not None
        assert (pre["rel_time"] < 0).all()
