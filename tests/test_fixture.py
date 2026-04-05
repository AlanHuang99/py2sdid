"""Tests using the pre-generated parquet fixture data.

The fixture contains 1500 units (500 in cohort 2000, 500 in cohort 2010,
500 never-treated) over 1990-2020. True treatment effects: te=2.0 for
cohort 2000, te=3.0 for cohort 2010, no dynamic slope.
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

DATA_DIR = Path(__file__).parent / "data"
PARQUET_PATH = DATA_DIR / "staggered_panel.parquet"
META_PATH = DATA_DIR / "staggered_panel_metadata.json"

pytestmark = pytest.mark.skipif(
    not PARQUET_PATH.exists(), reason="Fixture parquet not generated"
)


@pytest.fixture(scope="module")
def fixture_data():
    return pl.read_parquet(PARQUET_PATH)


@pytest.fixture(scope="module")
def fixture_meta():
    with open(META_PATH) as f:
        return json.load(f)


def test_fixture_shape(fixture_data, fixture_meta):
    """Fixture should have expected dimensions."""
    assert fixture_data.shape == tuple(fixture_meta["shape"])
    assert "unit" in fixture_data.columns
    assert "year" in fixture_data.columns
    assert "g" in fixture_data.columns
    assert "dep_var" in fixture_data.columns


def test_ts_did_on_fixture(fixture_data):
    """ts_did should recover known treatment effects from fixture."""
    from py2sdid import ts_did
    result = ts_did(fixture_data, yname="dep_var", idname="unit",
                    tname="year", gname="g", cluster_var="state",
                    verbose=False)
    # Weighted ATT should be between 2.0 and 3.0
    # (weighted average of te=2 for cohort 2000 and te=3 for cohort 2010)
    assert 1.5 < result.att_avg < 3.5


def test_bjs_did_on_fixture(fixture_data):
    """bjs_did should produce identical point estimates as ts_did."""
    from py2sdid import ts_did, bjs_did
    r1 = ts_did(fixture_data, yname="dep_var", idname="unit",
                tname="year", gname="g", se=False, verbose=False)
    r2 = bjs_did(fixture_data, yname="dep_var", idname="unit",
                 tname="year", gname="g", se=False, verbose=False)
    np.testing.assert_allclose(r1.att_avg, r2.att_avg, atol=1e-12)


def test_event_study_on_fixture(fixture_data):
    """Event study should show flat effects (no dynamic slope in DGP)."""
    from py2sdid import ts_did
    result = ts_did(fixture_data, yname="dep_var", idname="unit",
                    tname="year", gname="g", cluster_var="state",
                    verbose=False)
    es = result.event_study
    post = es.filter(pl.col("rel_time") >= 0)
    pre = es.filter(pl.col("rel_time") < 0)

    # Post-treatment estimates should be relatively stable (no slope)
    post_estimates = post["estimate"].to_numpy()
    assert np.std(post_estimates) < 0.5

    # Pre-treatment estimates should be near zero
    pre_estimates = pre["estimate"].to_numpy()
    assert abs(np.mean(pre_estimates)) < 0.2


def test_se_comparison_on_fixture(fixture_data):
    """ts_did and bjs_did SEs should be similar on fixture data."""
    from py2sdid import ts_did, bjs_did
    r1 = ts_did(fixture_data, yname="dep_var", idname="unit",
                tname="year", gname="g", cluster_var="state",
                verbose=False)
    r2 = bjs_did(fixture_data, yname="dep_var", idname="unit",
                 tname="year", gname="g", cluster_var="state",
                 verbose=False)
    ratio = r1.att_avg_se / r2.att_avg_se
    # SEs should be within 50% of each other
    assert 0.5 < ratio < 2.0, f"SE ratio {ratio:.4f}"


def test_pretrend_f_test_on_fixture(fixture_data):
    """Pre-trend F-test should not reject on well-specified fixture."""
    from py2sdid import ts_did
    result = ts_did(fixture_data, yname="dep_var", idname="unit",
                    tname="year", gname="g", cluster_var="state",
                    verbose=False)
    diag = result.diagnose()
    assert diag.pretrend_f_pval > 0.01


def test_fixture_reproducibility(fixture_data, fixture_meta):
    """Re-generating fixture with same seed should give same data."""
    sys.path.insert(0, str(DATA_DIR))
    from generate_fixture import generate_staggered_panel
    df2 = generate_staggered_panel(seed=fixture_meta["seed"])
    np.testing.assert_allclose(
        fixture_data["dep_var"].to_numpy(),
        df2["dep_var"].to_numpy(),
        atol=1e-5,
    )
