"""
Protocols and type aliases for py2sdid extensibility.

The EstimatorProtocol defines the interface that any DiD estimator must
implement. This enables future estimators (e.g., Callaway-Sant'Anna,
Sun-Abraham) to plug into the shared panel, plotting, and diagnostics
infrastructure without modification.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EstimatorProtocol(Protocol):
    """Interface for DiD estimators.

    Any estimator that implements ``estimate()`` and ``compute_se()`` can be
    used with py2sdid's shared infrastructure (panel preparation, plotting,
    diagnostics).
    """

    def estimate(
        self,
        panel: Any,  # PanelData — forward ref to avoid circular import
        **kwargs: Any,
    ) -> Any:  # EffectsResult
        """Compute treatment effects from prepared panel data.

        Parameters
        ----------
        panel : PanelData
            Structured panel data from ``panel.prepare_panel()``.
        **kwargs
            Estimator-specific options.

        Returns
        -------
        EffectsResult
            Treatment effect estimates and aggregations.
        """
        ...

    def compute_se(
        self,
        panel: Any,  # PanelData
        effects: Any,  # EffectsResult
        first_stage: Any,  # FirstStageResult
        **kwargs: Any,
    ) -> Any:  # InferenceResult
        """Compute standard errors for the estimated treatment effects.

        Parameters
        ----------
        panel : PanelData
            Structured panel data.
        effects : EffectsResult
            Output from ``estimate()``.
        first_stage : FirstStageResult
            First-stage estimation results (FE coefficients, design matrices).
        **kwargs
            Inference-specific options (e.g., ``n_bootstraps``, ``cluster``).

        Returns
        -------
        InferenceResult
            Standard errors, confidence intervals, p-values, and optionally
            the bootstrap distribution.
        """
        ...
