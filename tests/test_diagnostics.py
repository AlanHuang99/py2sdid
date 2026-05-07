"""Tests for diagnostic module."""
import pickle

import numpy as np
import pytest

from conftest import gen_data


def test_pretrend_f_test_passes():
    """Under correct specification, F-test should NOT reject."""
    from py2sdid import ts_did
    df = gen_data(n=1500, te1=3.0, te2=3.0, te_m1=0.0, te_m2=0.0)
    result = ts_did(df, yname="dep_var", idname="unit", tname="year", gname="g",
                    cluster_var="state", verbose=False)
    diag = result.diagnose()
    assert diag.pretrend_f_pval > 0.05


def test_equivalence_test():
    from py2sdid import ts_did
    df = gen_data(n=1500, te1=3.0, te2=3.0)
    result = ts_did(df, yname="dep_var", idname="unit", tname="year", gname="g",
                    cluster_var="state", verbose=False)
    diag = result.diagnose()
    assert diag.equiv_results is not None
    assert "tost_pval" in diag.equiv_results.columns
    assert diag.equiv_results["reject"].all()


def test_honestdid_sensitivity():
    from py2sdid import ts_did
    df = gen_data(n=1500, te1=3.0, te2=3.0)
    result = ts_did(df, yname="dep_var", idname="unit", tname="year", gname="g",
                    cluster_var="state", verbose=False)
    diag = result.diagnose()
    assert diag.honestdid_results is not None
    hd = diag.honestdid_results
    widths = (hd["ci_upper"] - hd["ci_lower"]).to_numpy()
    assert widths[-1] > widths[0]


def test_diagnostic_summary():
    from py2sdid import ts_did
    df = gen_data(n=1500, te1=3.0, te2=3.0)
    result = ts_did(df, yname="dep_var", idname="unit", tname="year", gname="g",
                    cluster_var="state", verbose=False)
    diag = result.diagnose()
    s = diag.summary()
    assert "Pre-trend" in s
    assert "Equivalence" in s


def test_hierarchical_diagnostics_and_placebo():
    from py2sdid import ts_did

    df = gen_data(n=500, te1=3.0, te2=3.0)
    result = ts_did(df, yname="dep_var", idname="unit", tname="year", gname="g",
                    cluster_var="state", verbose=False)
    diag = result.diagnose(delta=0.10, placebo_period=(-3, -1))

    assert diag.pretrend_f is not None
    assert diag.tost is not None
    assert diag.placebo is not None
    assert diag.pretrend_f.f_stat == diag.pretrend_f_stat
    assert diag.tost.max_pval == diag.equiv_max_pval
    assert "pretrend_f" in diag.available
    assert "tost" in diag.available
    assert "placebo" in diag.available

    placebo = diag.placebo_results
    assert placebo is not None
    assert placebo["period"][0] == (-3, -1)
    assert np.isfinite(placebo["estimate"][0])
    assert np.isfinite(placebo["se"][0])
    assert "equiv_pval" in placebo.columns


def test_fit_time_diagnostics_survive_slim_pickle():
    from py2sdid import ts_did

    df = gen_data(n=500, te1=3.0, te2=3.0)
    result = ts_did(
        df,
        yname="dep_var",
        idname="unit",
        tname="year",
        gname="g",
        cluster_var="state",
        diagnostics="full",
        diagnostics_options={"delta": 0.10, "placebo_period": (-3, -1)},
        verbose=False,
    )

    assert result.diagnostics is not None
    assert result.diagnostics.placebo is not None

    result.panel = None
    result.y_hat = None
    result.effects = None
    loaded = pickle.loads(pickle.dumps(result))

    assert loaded.diagnostics is not None
    assert loaded.diagnostics.tost is not None
    assert loaded.diagnostics.placebo is not None
    assert loaded.diagnostics.placebo.period == (-3, -1)
    assert loaded.diagnose() is loaded.diagnostics


def test_fit_time_diagnostics_validation_before_estimation():
    from py2sdid import ts_did

    df = gen_data(n=50, panel=(2000, 2004), g1=2002, g2=0, g3=0)

    with pytest.raises(ValueError, match="placebo_period"):
        ts_did(df, yname="dep_var", idname="unit", tname="year", gname="g",
               diagnostics=["placebo"], verbose=False)

    with pytest.raises(ValueError, match="unknown diagnostics_options"):
        ts_did(
            df,
            yname="dep_var",
            idname="unit",
            tname="year",
            gname="g",
            diagnostics="full",
            diagnostics_options={"unknown": 1},
            verbose=False,
        )


def test_bootstrap_placebo_uses_bootstrap_distribution():
    from py2sdid import ts_did

    df = gen_data(n=100, panel=(2000, 2010), g1=2005, g2=0, g3=0)
    result = ts_did(
        df,
        yname="dep_var",
        idname="unit",
        tname="year",
        gname="g",
        cluster_var="state",
        bootstrap=True,
        n_bootstraps=20,
        n_jobs=1,
        seed=123,
        verbose=False,
    )
    diag = result.diagnose(delta=0.50, placebo_period=(-3, -1))

    assert result.bootstrap_atts is result.boot_dist
    assert result.boot_dist is not None
    assert diag.placebo is not None
    assert np.isfinite(diag.placebo.se)
    assert 0 <= diag.placebo.p_value <= 1
    assert 0 <= diag.placebo.equiv_p_value <= 1
