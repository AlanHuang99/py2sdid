"""Basic smoke tests for py2sdid estimation pipeline."""
import numpy as np
import polars as pl
import pytest

from conftest import gen_data


def test_ts_did_static():
    from py2sdid import ts_did
    df = gen_data(n=600, te1=3.0, te2=3.0)
    result = ts_did(df, yname="dep_var", idname="unit", tname="year", gname="g",
                    cluster_var="state", se=False, verbose=False)
    assert abs(result.att_avg - 3.0) < 0.5
    assert result.method == "ts_did"


def test_bjs_did_static():
    from py2sdid import bjs_did
    df = gen_data(n=600, te1=3.0, te2=3.0)
    result = bjs_did(df, yname="dep_var", idname="unit", tname="year", gname="g",
                     cluster_var="state", se=False, verbose=False)
    assert abs(result.att_avg - 3.0) < 0.5
    assert result.method == "bjs_did"


def test_point_estimates_match():
    """ts_did and bjs_did must give identical point estimates."""
    from py2sdid import ts_did, bjs_did
    df = gen_data(n=600, te1=3.0, te2=3.0, seed=123)
    r1 = ts_did(df, yname="dep_var", idname="unit", tname="year", gname="g",
                se=False, verbose=False)
    r2 = bjs_did(df, yname="dep_var", idname="unit", tname="year", gname="g",
                 se=False, verbose=False)
    np.testing.assert_allclose(r1.att_avg, r2.att_avg, atol=1e-12)


def test_event_study_all_periods():
    """Should automatically compute all per-period event-study coefficients."""
    from py2sdid import ts_did
    df = gen_data(n=600, te1=3.0, te2=3.0)
    result = ts_did(df, yname="dep_var", idname="unit", tname="year", gname="g",
                    cluster_var="state", se=False, verbose=False)
    es = result.event_study
    assert len(es) > 0
    # Should have both pre-treatment (<0) and post-treatment (>=0) periods
    assert (es["rel_time"] < 0).any()
    assert (es["rel_time"] >= 0).any()
    # Pre-treatment estimates should be near 0
    pre = es.filter(pl.col("rel_time") < 0)
    assert abs(pre["estimate"].mean()) < 0.5
    # Post-treatment estimates should be near 3.0
    post = es.filter(pl.col("rel_time") >= 0)
    assert abs(post["estimate"].mean() - 3.0) < 0.5


def test_ts_did_with_se():
    from py2sdid import ts_did
    df = gen_data(n=600, te1=3.0, te2=3.0)
    result = ts_did(df, yname="dep_var", idname="unit", tname="year", gname="g",
                    cluster_var="state", se=True, verbose=False)
    assert result.att_avg_se is not None
    assert result.att_avg_se > 0
    assert result.att_avg_ci is not None
    lo, hi = result.att_avg_ci
    assert lo < result.att_avg < hi
    # Event study should have SEs too
    es = result.event_study
    assert "se" in es.columns
    assert es["se"].is_not_null().all()


def test_bjs_did_with_se():
    from py2sdid import bjs_did
    df = gen_data(n=600, te1=3.0, te2=3.0)
    result = bjs_did(df, yname="dep_var", idname="unit", tname="year", gname="g",
                     cluster_var="state", se=True, verbose=False)
    assert result.att_avg_se is not None
    assert result.att_avg_se > 0


def test_ts_did_with_covariates():
    from py2sdid import ts_did
    df = gen_data(n=600, te1=3.0, te2=3.0)
    df = df.with_columns(x1=pl.col("year").cast(pl.Float64) / 100)
    result = ts_did(df, yname="dep_var", idname="unit", tname="year", gname="g",
                    xformla=["x1"], se=False, verbose=False)
    assert result.beta is not None


def test_ts_did_weighted():
    from py2sdid import ts_did
    df = gen_data(n=600, te1=3.0, te2=3.0)
    df = df.with_columns(w=pl.lit(1.0))
    result = ts_did(df, yname="dep_var", idname="unit", tname="year", gname="g",
                    wname="w", se=False, verbose=False)
    assert abs(result.att_avg - 3.0) < 0.5


def test_summary_output():
    from py2sdid import ts_did
    df = gen_data(n=600, te1=3.0, te2=3.0)
    result = ts_did(df, yname="dep_var", idname="unit", tname="year", gname="g",
                    se=False, verbose=False)
    s = result.summary()
    assert "py2sdid" in s
    assert "ATT" in s


def test_ci_coverage():
    """95% CI should contain true ATT."""
    from py2sdid import ts_did
    df = gen_data(n=1500, te1=3.0, te2=3.0, seed=99)
    result = ts_did(df, yname="dep_var", idname="unit", tname="year", gname="g",
                    cluster_var="state", se=True, verbose=False)
    lo, hi = result.att_avg_ci
    assert lo < 3.0 < hi


def test_never_treated_null():
    """gname=null should be treated as never-treated."""
    from py2sdid import ts_did
    df = gen_data(n=600, te1=3.0, te2=3.0)
    df = df.with_columns(
        pl.when(pl.col("g") == 0).then(None).otherwise(pl.col("g")).alias("g")
    )
    result = ts_did(df, yname="dep_var", idname="unit", tname="year", gname="g",
                    se=False, verbose=False)
    assert abs(result.att_avg - 3.0) < 0.5


def test_heterogeneous_effects():
    from py2sdid import ts_did
    df = gen_data(n=1500, te1=2.0, te2=4.0, te_m1=0.1, te_m2=0.2, seed=42)
    result = ts_did(df, yname="dep_var", idname="unit", tname="year", gname="g",
                    cluster_var="state", se=True, verbose=False)
    assert 1.5 < result.att_avg < 6.0


def test_dynamic_effects_grow():
    """With positive te_m, effects should grow over time."""
    from py2sdid import ts_did
    df = gen_data(n=1500, te1=1.0, te2=1.0, te_m1=0.5, te_m2=0.5, seed=42)
    result = ts_did(df, yname="dep_var", idname="unit", tname="year", gname="g",
                    cluster_var="state", se=False, verbose=False)
    es = result.event_study
    att_0 = es.filter(pl.col("rel_time") == 0)["estimate"][0]
    att_10 = es.filter(pl.col("rel_time") == 10)["estimate"][0]
    assert att_10 > att_0 + 1.0


def test_gname_varying_within_unit_raises():
    """gname must be constant within each unit."""
    from py2sdid import ts_did
    df = gen_data(n=300, te1=3.0, te2=3.0)
    # Corrupt one unit's gname
    bad = df.with_row_index("_idx").with_columns(
        pl.when((pl.col("unit") == 1) & (pl.col("year") > 2005))
        .then(2010)
        .otherwise(pl.col("g"))
        .alias("g")
    ).drop("_idx")
    with pytest.raises(ValueError, match="constant within each unit"):
        ts_did(bad, yname="dep_var", idname="unit", tname="year", gname="g",
               se=False, verbose=False)


def test_att_by_horizon_property():
    """att_by_horizon property should return only post-treatment."""
    from py2sdid import ts_did
    df = gen_data(n=600, te1=3.0, te2=3.0)
    result = ts_did(df, yname="dep_var", idname="unit", tname="year", gname="g",
                    se=False, verbose=False)
    post = result.att_by_horizon
    assert (post["rel_time"] >= 0).all()


def test_pretrend_tests_property():
    """pretrend_tests property should return only pre-treatment."""
    from py2sdid import ts_did
    df = gen_data(n=600, te1=3.0, te2=3.0)
    result = ts_did(df, yname="dep_var", idname="unit", tname="year", gname="g",
                    se=False, verbose=False)
    pre = result.pretrend_tests
    assert pre is not None
    assert (pre["rel_time"] < 0).all()
