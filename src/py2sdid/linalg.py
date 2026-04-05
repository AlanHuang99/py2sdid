"""
Sparse linear algebra utilities for py2sdid.

Provides sparse fixed-effect matrix construction, robust least-squares
solving (QR with SVD fallback), and related matrix operations needed by
the first-stage estimator and inference modules.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# -------------------------------------------------------------------
# Fixed-effect indicator matrices
# -------------------------------------------------------------------

def build_fe_matrix(ids: np.ndarray, n_levels: int) -> sp.csc_matrix:
    """Build a sparse one-hot indicator matrix for fixed effects.

    Parameters
    ----------
    ids : ndarray of int
        Integer-coded identifiers (0-based).
    n_levels : int
        Number of unique levels.

    Returns
    -------
    scipy.sparse.csc_matrix, shape (len(ids), n_levels)
    """
    n = len(ids)
    return sp.csc_matrix(
        (np.ones(n, dtype=np.float64), (np.arange(n), ids)),
        shape=(n, n_levels),
    )


def build_design_matrix(
    unit_ids: np.ndarray,
    n_units: int,
    time_ids: np.ndarray,
    n_periods: int,
    X: np.ndarray | None = None,
) -> sp.csc_matrix:
    """Build the full design matrix: [unit_FE | time_FE | covariates].

    Parameters
    ----------
    unit_ids, time_ids : ndarray of int
        Integer-coded unit and time identifiers (0-based).
    n_units, n_periods : int
        Number of unique units and time periods.
    X : ndarray, shape (n_obs, K), optional
        Dense covariate matrix.

    Returns
    -------
    scipy.sparse.csc_matrix, shape (n_obs, n_units + n_periods + K)
    """
    fe_unit = build_fe_matrix(unit_ids, n_units)
    fe_time = build_fe_matrix(time_ids, n_periods)

    parts = [fe_unit, fe_time]
    if X is not None:
        parts.append(sp.csc_matrix(X))

    return sp.hstack(parts, format="csc")


# -------------------------------------------------------------------
# Robust linear-system solver
# -------------------------------------------------------------------

def robust_solve(A, b: np.ndarray) -> np.ndarray:
    """Solve ``A x = b`` robustly with QR / SVD fallback.

    Works for both dense and sparse ``A``.  For sparse systems uses
    ``scipy.sparse.linalg.lsqr``; for dense tries ``np.linalg.solve``
    and falls back to the SVD-based pseudo-inverse on rank deficiency.

    Parameters
    ----------
    A : ndarray or sparse matrix
        Coefficient matrix.
    b : ndarray
        Right-hand-side vector (1-D) or matrix (2-D).

    Returns
    -------
    ndarray
        Least-squares solution *x*.
    """
    if sp.issparse(A):
        return _solve_sparse(A, b)
    return _solve_dense(np.asarray(A, dtype=np.float64), np.asarray(b, dtype=np.float64))


def _solve_sparse(A: sp.spmatrix, b: np.ndarray) -> np.ndarray:
    """Solve via scipy lsqr (handles rank deficiency natively)."""
    b = np.asarray(b, dtype=np.float64)
    if b.ndim == 1:
        return spla.lsqr(A.astype(np.float64), b, atol=1e-14, btol=1e-14)[0]
    # Multi-RHS: solve column by column
    out = np.empty((A.shape[1], b.shape[1]), dtype=np.float64)
    for j in range(b.shape[1]):
        out[:, j] = spla.lsqr(A.astype(np.float64), b[:, j], atol=1e-14, btol=1e-14)[0]
    return out


def _solve_dense(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve via direct then SVD fallback."""
    try:
        if A.shape[0] == A.shape[1]:
            return np.linalg.solve(A, b)
        return np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return _pinv_solve(A, b)


def _pinv_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """SVD-based pseudo-inverse solve."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    tol = max(A.shape) * np.finfo(np.float64).eps * s[0]
    mask = s > tol
    s_inv = np.where(mask, 1.0 / s, 0.0)
    return Vt.T @ np.diag(s_inv) @ U.T @ b


# -------------------------------------------------------------------
# Convenience wrappers
# -------------------------------------------------------------------

def sparse_xtx_inv(X: sp.spmatrix) -> np.ndarray:
    """Compute ``(X'X)^{-1}`` for a sparse *X*.

    Falls back to pseudo-inverse when rank-deficient.

    Returns
    -------
    ndarray, shape (p, p)
    """
    XtX = (X.T @ X).toarray()
    return robust_solve(XtX, np.eye(XtX.shape[0]))


def sparse_crossprod(X: sp.spmatrix, y: np.ndarray) -> np.ndarray:
    """Compute ``X' y`` for sparse *X* and dense *y*.

    Returns
    -------
    ndarray
    """
    if y.ndim == 1:
        return np.asarray(X.T @ y).ravel()
    return np.asarray(X.T @ y)
