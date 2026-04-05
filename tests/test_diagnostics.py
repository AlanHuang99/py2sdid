"""Tests for diagnostic module."""
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
