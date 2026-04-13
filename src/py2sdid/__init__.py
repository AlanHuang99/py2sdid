"""
py2sdid -- Two-stage DiD and BJS imputation estimator for Python.

Implementation of two-stage difference-in-differences (Gardner 2021)
and BJS imputation (Borusyak, Jaravel, Spiess 2024) estimators for
causal inference under staggered treatment adoption.

Usage::

    import py2sdid

    result = py2sdid.ts_did(
        data=df,
        yname="dep_var",
        idname="unit",
        tname="year",
        gname="g",
        cluster_var="cluster_id",
    )
    result.summary()
    result.plot()

"""

__version__ = "0.1.6"

from .core import ts_did, bjs_did
from .results import DiDResult, DiagnosticResult

__all__ = ["ts_did", "bjs_did", "DiDResult", "DiagnosticResult"]
