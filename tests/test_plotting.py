"""Smoke tests for all plot types."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from conftest import gen_data


@pytest.fixture
def result_with_se():
    from py2sdid import ts_did
    df = gen_data(n=600, te1=3.0, te2=3.0, seed=42)
    return ts_did(df, yname="dep_var", idname="unit", tname="year", gname="g",
                  cluster_var="state", verbose=False)


@pytest.fixture
def result_no_se():
    from py2sdid import ts_did
    df = gen_data(n=300, te1=3.0, te2=3.0)
    return ts_did(df, yname="dep_var", idname="unit", tname="year", gname="g",
                  se=False, verbose=False)


@pytest.mark.parametrize("kind", [
    "event_study", "pretrends", "treatment_status", "counterfactual", "calendar",
])
def test_plot_renders(result_with_se, kind):
    fig = result_with_se.plot(kind=kind)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_no_se_event_study(result_no_se):
    fig = result_no_se.plot(kind="event_study")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_no_se_treatment_status(result_no_se):
    fig = result_no_se.plot(kind="treatment_status")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
