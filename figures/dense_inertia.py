"""Exact half-signature of a dense Hermitian matrix from an LDL^T factorisation."""
import numpy as np
from scipy import linalg


def inertia(M):
    """Number of positive minus number of negative eigenvalues of the Hermitian matrix M."""
    A = M.toarray() if hasattr(M, "toarray") else np.array(M)
    lu, d, perm = linalg.ldl(A, hermitian=True, lower=True, overwrite_a=True, check_finite=False)
    dg = np.real(np.diag(d)); off = np.diag(d, -1)
    ev = linalg.eigvalsh_tridiagonal(dg, np.abs(off), check_finite=False) if off.size else dg
    return int((ev > 0).sum() - (ev < 0).sum())
