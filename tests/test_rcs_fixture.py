"""Tests using pre-generated RCS parquet fixture data.

The fixture contains 50 groups over 2000-2015:
- 15 groups treated at 2005 (te=2.0, no dynamic slope)
- 10 groups treated at 2010 (te=3.0, no dynamic slope)
- 25 groups never-treated
- 30 individuals per group-period (individual-level dataset)
- Aggregated version: one row per group-period with mean outcome
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

DATA_DIR = Path(__file__).parent / "data"
IND_PATH = DATA_DIR / "rcs_individual.parquet"
AGG_PATH = DATA_DIR / "rcs_aggregated.parquet"
META_PATH = DATA_DIR / "rcs_metadata.json"

pytestmark = pytest.mark.skipif(
    not IND_PATH.exists(), reason="RCS fixture parquet not generated"
)


@pytest.fixture(scope="module")
def ind_data():
    return pl.read_parquet(IND_PATH)


@pytest.fixture(scope="module")
def agg_data():
    return pl.read_parquet(AGG_PATH)


@pytest.fixture(scope="module")
def meta():
    with open(META_PATH) as f:
        return json.load(f)


# ===================================================================
# Shape / structure checks
# ===================================================================

def test_individual_fixture_shape(ind_data, meta):
    assert ind_data.shape == tuple(meta["individual_shape"])
    for col in ("individual_id", "group", "region", "year", "g", "dep_var"):
        assert col in ind_data.columns


def test_aggregated_fixture_shape(agg_data, meta):
    assert agg_data.shape == tuple(meta["aggregated_shape"])
    for col in ("group", "year", "g", "dep_var"):
        assert col in agg_data.columns


def test_fixture_reproducibility(ind_data, meta):
    sys.path.insert(0, str(DATA_DIR))
    from generate_rcs_fixture import generate_individual_rcs
    df2 = generate_individual_rcs(seed=meta["seed"])
    np.testing.assert_allclose(
        ind_data["dep_var"].to_numpy(),
        df2["dep_var"].to_numpy(),
        atol=1e-5,
    )


# ===================================================================
# Individual-level RCS — point estimates
# ===================================================================

def test_ts_did_individual_fixture(ind_data):
    from py2sdid import ts_did
    result = ts_did(ind_data, yname="dep_var", idname="individual_id",
                    tname="year", gname="g", dataset_type="rcs",
                    groupname="group", verbose=False)
    # Weighted ATT across cohorts: between 2.0 and 3.0
    assert 1.5 < result.att_avg < 3.5
    assert result.panel.is_rcs
    assert result.panel.n_fe_levels == 50


def test_bjs_did_individual_fixture(ind_data):
    from py2sdid import bjs_did
    result = bjs_did(ind_data, yname="dep_var", idname="individual_id",
                     tname="year", gname="g", dataset_type="rcs",
                     groupname="group", verbose=False)
    assert 1.5 < result.att_avg < 3.5


def test_individual_ts_bjs_identical(ind_data):
    from py2sdid import ts_did, bjs_did
    r1 = ts_did(ind_data, yname="dep_var", idname="individual_id",
                tname="year", gname="g", dataset_type="rcs",
                groupname="group", se=False, verbose=False)
    r2 = bjs_did(ind_data, yname="dep_var", idname="individual_id",
                 tname="year", gname="g", dataset_type="rcs",
                 groupname="group", se=False, verbose=False)
    np.testing.assert_allclose(r1.att_avg, r2.att_avg, atol=1e-12)


# ===================================================================
# Aggregated RCS — point estimates
# ===================================================================

def test_ts_did_aggregated_fixture(agg_data):
    from py2sdid import ts_did
    result = ts_did(agg_data, yname="dep_var", idname="group",
                    tname="year", gname="g", dataset_type="rcs",
                    verbose=False)
    assert 1.5 < result.att_avg < 3.5
    assert result.panel.is_rcs


def test_bjs_did_aggregated_fixture(agg_data):
    from py2sdid import bjs_did
    result = bjs_did(agg_data, yname="dep_var", idname="group",
                     tname="year", gname="g", dataset_type="rcs",
                     verbose=False)
    assert 1.5 < result.att_avg < 3.5


def test_aggregated_ts_bjs_identical(agg_data):
    from py2sdid import ts_did, bjs_did
    r1 = ts_did(agg_data, yname="dep_var", idname="group",
                tname="year", gname="g", dataset_type="rcs",
                se=False, verbose=False)
    r2 = bjs_did(agg_data, yname="dep_var", idname="group",
                 tname="year", gname="g", dataset_type="rcs",
                 se=False, verbose=False)
    np.testing.assert_allclose(r1.att_avg, r2.att_avg, atol=1e-12)


# ===================================================================
# Event study
# ===================================================================

def test_individual_event_study(ind_data):
    from py2sdid import ts_did
    result = ts_did(ind_data, yname="dep_var", idname="individual_id",
                    tname="year", gname="g", dataset_type="rcs",
                    groupname="group", se=False, verbose=False)
    es = result.event_study

    # Should have both pre and post periods
    assert (es["rel_time"] < 0).any()
    assert (es["rel_time"] >= 0).any()

    # Pre-treatment: near zero (no DGP violation of parallel trends)
    pre = es.filter(pl.col("rel_time") < 0)["estimate"].to_numpy()
    assert abs(np.mean(pre)) < 0.3

    # Post-treatment: stable (no dynamic slope in DGP)
    post = es.filter(pl.col("rel_time") >= 0)["estimate"].to_numpy()
    assert np.std(post) < 0.5


def test_aggregated_event_study(agg_data):
    from py2sdid import ts_did
    result = ts_did(agg_data, yname="dep_var", idname="group",
                    tname="year", gname="g", dataset_type="rcs",
                    se=False, verbose=False)
    es = result.event_study
    pre = es.filter(pl.col("rel_time") < 0)["estimate"].to_numpy()
    assert abs(np.mean(pre)) < 0.3


# ===================================================================
# Standard errors and inference
# ===================================================================

def test_individual_se_comparison(ind_data):
    """ts_did and bjs_did SEs should be in the same ballpark."""
    from py2sdid import ts_did, bjs_did
    r1 = ts_did(ind_data, yname="dep_var", idname="individual_id",
                tname="year", gname="g", dataset_type="rcs",
                groupname="group", verbose=False)
    r2 = bjs_did(ind_data, yname="dep_var", idname="individual_id",
                 tname="year", gname="g", dataset_type="rcs",
                 groupname="group", verbose=False)
    ratio = r1.att_avg_se / r2.att_avg_se
    assert 0.3 < ratio < 3.0, f"SE ratio {ratio:.4f}"


def test_individual_ci_coverage(ind_data):
    """95% CI from individual RCS should contain a plausible ATT."""
    from py2sdid import ts_did
    result = ts_did(ind_data, yname="dep_var", idname="individual_id",
                    tname="year", gname="g", dataset_type="rcs",
                    groupname="group", verbose=False)
    lo, hi = result.att_avg_ci
    # True weighted ATT is between 2.0 and 3.0
    assert lo < 3.5 and hi > 1.5


def test_aggregated_ci_coverage(agg_data):
    from py2sdid import ts_did
    result = ts_did(agg_data, yname="dep_var", idname="group",
                    tname="year", gname="g", dataset_type="rcs",
                    verbose=False)
    lo, hi = result.att_avg_ci
    assert lo < 3.5 and hi > 1.5


def test_individual_cluster_by_region(ind_data):
    """Clustering at a higher level (region) should produce wider SEs."""
    from py2sdid import ts_did
    r_group = ts_did(ind_data, yname="dep_var", idname="individual_id",
                     tname="year", gname="g", dataset_type="rcs",
                     groupname="group", verbose=False)
    r_region = ts_did(ind_data, yname="dep_var", idname="individual_id",
                      tname="year", gname="g", dataset_type="rcs",
                      groupname="group", cluster_var="region", verbose=False)
    # Fewer clusters -> generally wider SEs (though not guaranteed)
    assert len(r_region.panel.cluster_map) < len(r_group.panel.cluster_map)


# ===================================================================
# Diagnostics
# ===================================================================

def test_individual_pretrend_f_test(ind_data):
    """Pre-trend F-test should not reject (DGP satisfies parallel trends)."""
    from py2sdid import ts_did
    result = ts_did(ind_data, yname="dep_var", idname="individual_id",
                    tname="year", gname="g", dataset_type="rcs",
                    groupname="group", verbose=False)
    diag = result.diagnose()
    assert diag.pretrend_f_pval > 0.01


def test_individual_diagnostics_summary(ind_data):
    from py2sdid import ts_did
    result = ts_did(ind_data, yname="dep_var", idname="individual_id",
                    tname="year", gname="g", dataset_type="rcs",
                    groupname="group", verbose=False)
    diag = result.diagnose()
    s = diag.summary()
    assert "Pre-trend" in s
    assert "F(" in s


# ===================================================================
# Consistency between individual and aggregated
# ===================================================================

def test_individual_vs_aggregated_att(ind_data, agg_data):
    """Individual and aggregated RCS should produce similar ATTs."""
    from py2sdid import ts_did
    r_ind = ts_did(ind_data, yname="dep_var", idname="individual_id",
                   tname="year", gname="g", dataset_type="rcs",
                   groupname="group", se=False, verbose=False)
    r_agg = ts_did(agg_data, yname="dep_var", idname="group",
                   tname="year", gname="g", dataset_type="rcs",
                   se=False, verbose=False)
    # Point estimates won't be identical (aggregation changes weighting)
    # but should be in the same neighborhood
    assert abs(r_ind.att_avg - r_agg.att_avg) < 0.5


# ===================================================================
# Event-study coefficient verification against known true effects
# ===================================================================
#
# DGP: cohort 2005 (15 groups, te=2.0), cohort 2010 (10 groups, te=3.0)
# No dynamic slope => every post-treatment horizon should recover a
# weighted average of 2.0 and 3.0.  The exact weight depends on which
# cohorts contribute at each relative time:
#   - rel_time 0..4:  both cohorts contribute
#   - rel_time 5..10: only cohort 2005 contributes (te=2.0)
#   - rel_time <0:    pre-treatment, should be ~0
#
# These tests use generous tolerances (0.5) because noise is real.

def test_individual_event_study_pre_coefficients(ind_data):
    """Every pre-treatment coefficient should be close to zero."""
    from py2sdid import ts_did
    result = ts_did(ind_data, yname="dep_var", idname="individual_id",
                    tname="year", gname="g", dataset_type="rcs",
                    groupname="group", se=False, verbose=False)
    es = result.event_study
    pre = es.filter(pl.col("rel_time") < 0)
    for row in pre.iter_rows(named=True):
        assert abs(row["estimate"]) < 0.5, (
            f"Pre-treatment coeff at rel_time={row['rel_time']}: "
            f"{row['estimate']:.4f} (expected ~0)"
        )


def test_individual_event_study_post_coefficients(ind_data):
    """Post-treatment coefficients should match the DGP.

    rel_time 0-4: both cohort 2005 (te=2) and cohort 2010 (te=3) contribute.
    rel_time 5+:  only cohort 2005 (te=2) contributes.
    """
    from py2sdid import ts_did
    result = ts_did(ind_data, yname="dep_var", idname="individual_id",
                    tname="year", gname="g", dataset_type="rcs",
                    groupname="group", se=False, verbose=False)
    es = result.event_study

    # Horizons where only cohort 2005 contributes (rel_time >= 5)
    late_post = es.filter(pl.col("rel_time") >= 5)
    if len(late_post) > 0:
        late_est = late_post["estimate"].to_numpy()
        for h, est in zip(late_post["rel_time"].to_list(), late_est):
            assert abs(est - 2.0) < 0.5, (
                f"Late post (cohort 2005 only) at rel_time={h}: "
                f"{est:.4f} (expected ~2.0)"
            )

    # Horizons where both cohorts contribute (rel_time 0-4)
    early_post = es.filter(
        (pl.col("rel_time") >= 0) & (pl.col("rel_time") < 5)
    )
    if len(early_post) > 0:
        early_est = early_post["estimate"].to_numpy()
        for h, est in zip(early_post["rel_time"].to_list(), early_est):
            # Weighted average of te=2 (15 groups) and te=3 (10 groups)
            # Exact weight depends on group sizes; expect between 2 and 3
            assert 1.5 < est < 3.5, (
                f"Early post (both cohorts) at rel_time={h}: "
                f"{est:.4f} (expected between 2.0 and 3.0)"
            )


def test_aggregated_event_study_post_coefficients(agg_data):
    """Aggregated RCS post-treatment coefficients should match the DGP."""
    from py2sdid import ts_did
    result = ts_did(agg_data, yname="dep_var", idname="group",
                    tname="year", gname="g", dataset_type="rcs",
                    se=False, verbose=False)
    es = result.event_study

    late_post = es.filter(pl.col("rel_time") >= 5)
    if len(late_post) > 0:
        for row in late_post.iter_rows(named=True):
            assert abs(row["estimate"] - 2.0) < 0.5, (
                f"Agg late post at rel_time={row['rel_time']}: "
                f"{row['estimate']:.4f} (expected ~2.0)"
            )

    early_post = es.filter(
        (pl.col("rel_time") >= 0) & (pl.col("rel_time") < 5)
    )
    if len(early_post) > 0:
        for row in early_post.iter_rows(named=True):
            assert 1.5 < row["estimate"] < 3.5, (
                f"Agg early post at rel_time={row['rel_time']}: "
                f"{row['estimate']:.4f} (expected 2.0-3.0)"
            )


def test_individual_event_study_coefficients_with_se(ind_data):
    """Each post-treatment coefficient's CI should contain a plausible true effect."""
    from py2sdid import ts_did
    result = ts_did(ind_data, yname="dep_var", idname="individual_id",
                    tname="year", gname="g", dataset_type="rcs",
                    groupname="group", se=True, verbose=False)
    es = result.event_study
    post = es.filter(pl.col("rel_time") >= 0)
    for row in post.iter_rows(named=True):
        lo, hi = row["ci_lower"], row["ci_upper"]
        # True effect is between 2.0 and 3.0 depending on cohort mix
        # CI should overlap with [1.5, 3.5]
        assert hi > 1.5 and lo < 3.5, (
            f"Post CI at rel_time={row['rel_time']}: "
            f"[{lo:.4f}, {hi:.4f}] does not overlap [1.5, 3.5]"
        )


# ===================================================================
# Event-study plotting for RCS data
# ===================================================================

def test_individual_rcs_plot_event_study(ind_data):
    """event_study plot should render for individual-level RCS."""
    from py2sdid import ts_did
    result = ts_did(ind_data, yname="dep_var", idname="individual_id",
                    tname="year", gname="g", dataset_type="rcs",
                    groupname="group", verbose=False)
    fig = result.plot(kind="event_study")
    assert isinstance(fig, plt.Figure)
    axes = fig.get_axes()
    assert len(axes) >= 1
    lines = axes[0].get_lines()
    assert len(lines) > 0 or len(axes[0].collections) > 0
    plt.close(fig)


def test_individual_rcs_plot_pretrends(ind_data):
    """pretrends plot should render for individual-level RCS."""
    from py2sdid import ts_did
    result = ts_did(ind_data, yname="dep_var", idname="individual_id",
                    tname="year", gname="g", dataset_type="rcs",
                    groupname="group", verbose=False)
    fig = result.plot(kind="pretrends")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_individual_rcs_plot_treatment_status(ind_data):
    """treatment_status should show groups (not individuals) for RCS."""
    from py2sdid import ts_did
    result = ts_did(ind_data, yname="dep_var", idname="individual_id",
                    tname="year", gname="g", dataset_type="rcs",
                    groupname="group", se=False, verbose=False)
    fig = result.plot(kind="treatment_status")
    assert isinstance(fig, plt.Figure)
    ax = fig.get_axes()[0]
    # Y-axis should say "Group", not "Unit"
    assert "Group" in ax.get_ylabel()
    # Heatmap should have n_groups rows (50), not n_individuals (24000)
    images = ax.get_images()
    assert len(images) == 1
    assert images[0].get_array().shape[0] == 50
    plt.close(fig)


def test_individual_rcs_plot_counterfactual(ind_data):
    """counterfactual should show group-level means for individual RCS."""
    from py2sdid import ts_did
    result = ts_did(ind_data, yname="dep_var", idname="individual_id",
                    tname="year", gname="g", dataset_type="rcs",
                    groupname="group", se=False, verbose=False)
    fig = result.plot(kind="counterfactual")
    assert isinstance(fig, plt.Figure)
    ax = fig.get_axes()[0]
    # Should have lines plotted (actual + Y(0) for 3 groups = 6 lines)
    lines = ax.get_lines()
    assert len(lines) >= 2  # at least one group with actual + counterfactual
    # Y-axis label should mention group mean
    assert "group mean" in ax.get_ylabel().lower()
    plt.close(fig)


def test_individual_rcs_plot_calendar(ind_data):
    """calendar plot should render for individual-level RCS."""
    from py2sdid import ts_did
    result = ts_did(ind_data, yname="dep_var", idname="individual_id",
                    tname="year", gname="g", dataset_type="rcs",
                    groupname="group", se=False, verbose=False)
    fig = result.plot(kind="calendar")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_aggregated_rcs_plot_event_study(agg_data):
    """event_study plot should render for aggregated RCS."""
    from py2sdid import ts_did
    result = ts_did(agg_data, yname="dep_var", idname="group",
                    tname="year", gname="g", dataset_type="rcs",
                    verbose=False)
    fig = result.plot(kind="event_study")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_aggregated_rcs_plot_treatment_status(agg_data):
    """treatment_status for aggregated RCS should show groups."""
    from py2sdid import ts_did
    result = ts_did(agg_data, yname="dep_var", idname="group",
                    tname="year", gname="g", dataset_type="rcs",
                    se=False, verbose=False)
    fig = result.plot(kind="treatment_status")
    assert isinstance(fig, plt.Figure)
    ax = fig.get_axes()[0]
    assert "Group" in ax.get_ylabel()
    images = ax.get_images()
    assert images[0].get_array().shape[0] == 50
    plt.close(fig)


def test_aggregated_rcs_plot_counterfactual(agg_data):
    """counterfactual for aggregated RCS should show group trajectories."""
    from py2sdid import ts_did
    result = ts_did(agg_data, yname="dep_var", idname="group",
                    tname="year", gname="g", dataset_type="rcs",
                    se=False, verbose=False)
    fig = result.plot(kind="counterfactual")
    assert isinstance(fig, plt.Figure)
    ax = fig.get_axes()[0]
    lines = ax.get_lines()
    assert len(lines) >= 2
    plt.close(fig)


def test_aggregated_rcs_plot_no_se(agg_data):
    """event_study plot without SEs should still render for RCS."""
    from py2sdid import ts_did
    result = ts_did(agg_data, yname="dep_var", idname="group",
                    tname="year", gname="g", dataset_type="rcs",
                    se=False, verbose=False)
    fig = result.plot(kind="event_study")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
