"""Zero-dimensional invariants of a gapped spectral localizer.

Three invariants are used in the paper:

* class A / AI / AII : ``half_signature`` = Sig(H')/2, an integer;
* class BDI          : ``sgn det A`` where ``A`` is the off-diagonal block of
  ``H'`` in the chiral basis, real up to an overall phase;
* class D            : ``sgn Pf A`` where ``H' = i A`` with ``A`` real
  antisymmetric in a Majorana basis.

The signature is obtained from a Hermitian LDL* factorisation with 1x1 and 2x2
pivots, which only needs the inertia and never a full diagonalisation.  The
sparse implementation is kept from the original HOTI script.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

__all__ = [
    "half_signature",
    "signature_dense",
    "chiral_block",
    "determinant_sign",
    "majorana_form",
    "pfaffian_sign_log",
    "sparse_ldl_inertia",
]


# --------------------------------------------------------------------------
# Signature
# --------------------------------------------------------------------------

def signature_dense(matrix, atol: float = 1.0e-9):
    """Return ``(signature, n_zero)`` from a dense diagonalisation."""
    dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    eigenvalues = np.linalg.eigvalsh(dense)
    positive = int(np.count_nonzero(eigenvalues > atol))
    negative = int(np.count_nonzero(eigenvalues < -atol))
    return positive - negative, dense.shape[0] - positive - negative


def half_signature(matrix, atol: float = 1.0e-10, sparse_limit: int = 4000):
    """Return ``(Sig/2, Sig, n_zero)``.

    Uses the sparse LDL* inertia for large matrices and a dense
    diagonalisation for small ones.
    """
    n = matrix.shape[0]
    if n <= sparse_limit or not sparse.issparse(matrix):
        signature, zero = signature_dense(matrix, atol=max(atol, 1.0e-9))
        return signature // 2, signature, zero

    positive, negative, zero = sparse_ldl_inertia(matrix, atol=atol, pivot_tol=atol)
    signature = positive - negative
    return signature // 2, signature, zero


# --------------------------------------------------------------------------
# Class BDI: sign of the determinant of the chiral off-diagonal block
# --------------------------------------------------------------------------

def chiral_block(localizer, chiral, atol: float = 1.0e-8):
    """Off-diagonal block ``A`` of ``localizer`` in the eigenbasis of ``chiral``.

    ``chiral`` must be a Hermitian unitary anticommuting with ``localizer``.
    """
    dense_l = localizer.toarray() if sparse.issparse(localizer) else np.asarray(localizer)
    dense_c = chiral.toarray() if sparse.issparse(chiral) else np.asarray(chiral)

    residual = np.linalg.norm(dense_c @ dense_l + dense_l @ dense_c)
    if residual > atol * max(1.0, np.linalg.norm(dense_l)):
        raise ValueError(f"chiral operator does not anticommute with the localizer "
                         f"(residual {residual:.3e})")

    values, vectors = np.linalg.eigh(dense_c)
    plus = vectors[:, values > 0]
    minus = vectors[:, values < 0]
    if plus.shape[1] != minus.shape[1]:
        raise ValueError("the chiral operator has unbalanced eigenspaces")
    return plus.conj().T @ dense_l @ minus


def determinant_sign(block, reference_phase: complex | None = None):
    """Return ``(sign, phase, log|det|)`` for the BDI determinant invariant.

    ``block`` is real only in a particular basis; in any other basis ``det``
    carries a constant overall phase.  Passing ``reference_phase`` (the phase
    obtained in a known-trivial configuration) divides it out, which is what
    makes the sign meaningful along a parameter sweep.
    """
    phase, log_abs = np.linalg.slogdet(np.asarray(block))
    if reference_phase is None:
        reference_phase = phase / abs(phase) if abs(phase) else 1.0
    ratio = phase / reference_phase
    if abs(ratio.imag) > 1.0e-6:
        raise RuntimeError(f"det A is not real relative to the reference phase "
                           f"(imaginary part {ratio.imag:.2e})")
    return int(np.sign(ratio.real)), phase, float(log_abs)


# --------------------------------------------------------------------------
# Class D: sign of the Pfaffian in a Majorana basis
# --------------------------------------------------------------------------

def majorana_form(localizer, rotation, atol: float = 1.0e-8):
    """Return ``A`` real antisymmetric such that ``rotation^dag L rotation = i A``."""
    rotated = rotation.conj().T @ localizer @ rotation
    dense = rotated.toarray() if sparse.issparse(rotated) else np.asarray(rotated)
    real_part = np.linalg.norm(dense.real)
    if real_part > atol * max(1.0, np.linalg.norm(dense)):
        raise ValueError(f"the rotated localizer is not purely imaginary "
                         f"(real part {real_part:.3e})")
    a_matrix = dense.imag
    return 0.5 * (a_matrix - a_matrix.T)


def pfaffian_sign_log(matrix, atol: float = 1.0e-12):
    """Return ``(sign(Pf A), log|Pf A|)`` for a real antisymmetric ``A``.

    Uses skew-symmetric Gaussian elimination with partial pivoting; the
    logarithm avoids overflow for the large matrices used here.
    """
    a = np.array(matrix, dtype=float, copy=True)
    n = a.shape[0]
    if a.shape != (n, n) or n % 2:
        raise ValueError("the Pfaffian needs an even-dimensional square matrix")

    pf_sign = 1
    log_abs = 0.0
    for k in range(0, n - 1, 2):
        pivot = k + 1 + int(np.argmax(np.abs(a[k, k + 1:])))
        if abs(a[k, pivot]) < atol:
            return 0, -np.inf
        if pivot != k + 1:
            a[[k + 1, pivot], :] = a[[pivot, k + 1], :]
            a[:, [k + 1, pivot]] = a[:, [pivot, k + 1]]
            pf_sign *= -1

        pivot_value = a[k, k + 1]
        pf_sign *= 1 if pivot_value > 0 else -1
        log_abs += np.log(abs(pivot_value))

        if k + 2 < n:
            row = a[k + 1, k + 2:].copy()
            tau = a[k, k + 2:].copy() / pivot_value
            a[k + 2:, k + 2:] += np.outer(row, tau) - np.outer(tau, row)

    return pf_sign, log_abs


# --------------------------------------------------------------------------
# Sparse Hermitian LDL* inertia
# --------------------------------------------------------------------------

def sparse_ldl_inertia(matrix, atol: float = 1.0e-10, pivot_tol: float = 1.0e-10,
                       ordering: str | None = "rcm"):
    """Inertia ``(n_positive, n_negative, n_zero)`` from a sparse LDL*.

    Symmetric 1x1 and 2x2 pivots are used, so the signature follows directly
    from the eigenvalues of the block-diagonal ``D``.
    """
    a = _as_hermitian_sparse(matrix, atol)
    active = _initial_symmetric_order(a, ordering)
    active_set = set(active)
    work = _to_hermitian_row_dicts(a, atol)
    positive = negative = zero = 0
    alpha = 0.64

    while active:
        i = active[0]
        diag_i = _real_diag(work, i)
        row_i = _active_offdiagonal_values(work, i, active_set, atol)
        max_off = max((abs(value) for value in row_i.values()), default=0.0)

        if abs(diag_i) > pivot_tol and (max_off <= pivot_tol or abs(diag_i) >= alpha * max_off):
            positive += diag_i > atol
            negative += diag_i < -atol
            zero += abs(diag_i) <= atol
            rest = active[1:]
            _ldl_1x1_update(work, i, rest, diag_i, atol)
            _drop_hermitian_indices(work, (i,))
            active_set.remove(i)
            active = rest
            continue

        if not row_i:
            if abs(diag_i) <= pivot_tol:
                zero += 1
            else:
                positive += diag_i > atol
                negative += diag_i < -atol
            _drop_hermitian_indices(work, (i,))
            active_set.remove(i)
            active = active[1:]
            continue

        j = max(row_i, key=lambda col: abs(row_i[col]))
        pivot = (_real_diag(work, i), work[i].get(j, 0.0), _real_diag(work, j))
        pivot_eigs = _hermitian_2x2_eigenvalues(*pivot)
        if np.min(np.abs(pivot_eigs)) <= pivot_tol:
            raise np.linalg.LinAlgError("sparse LDL* found a singular 2x2 pivot")

        for eigenvalue in pivot_eigs:
            positive += eigenvalue > atol
            negative += eigenvalue < -atol
            zero += abs(eigenvalue) <= atol

        rest = [node for node in active[1:] if node != j]
        _ldl_2x2_update(work, i, j, rest, pivot, atol)
        _drop_hermitian_indices(work, (i, j))
        active_set.remove(i)
        active_set.remove(j)
        active = rest

    return positive, negative, zero


def _as_hermitian_sparse(matrix, atol):
    a = matrix.asformat("csr") if sparse.issparse(matrix) else sparse.csr_matrix(matrix)
    residual = (a - a.getH()).tocoo()
    if residual.nnz and np.max(np.abs(residual.data)) > atol:
        raise ValueError("matrix must be Hermitian")
    return (0.5 * (a + a.getH())).tocsr()


def _initial_symmetric_order(a, ordering):
    if ordering is None:
        return list(range(a.shape[0]))
    if ordering != "rcm":
        raise ValueError("ordering must be 'rcm' or None")
    pattern = (a != 0).astype(np.int8)
    pattern = pattern + pattern.T
    return list(map(int, sparse.csgraph.reverse_cuthill_mckee(pattern, symmetric_mode=True)))


def _to_hermitian_row_dicts(a, atol):
    rows = [dict() for _ in range(a.shape[0])]
    coo = a.tocoo()
    for row, col, value in zip(coo.row, coo.col, coo.data):
        if abs(value) > atol:
            rows[int(row)][int(col)] = complex(value)
    return rows


def _real_diag(work, index):
    return float(work[index].get(index, 0.0).real)


def _active_offdiagonal_values(work, row, active_set, atol):
    return {col: value for col, value in work[row].items()
            if col != row and col in active_set and abs(value) > atol}


def _ldl_1x1_update(work, i, rest, pivot, atol):
    neighbors = {col: work[i].get(col, 0.0) for col in rest
                 if abs(work[i].get(col, 0.0)) > atol}
    affected = sorted(neighbors)
    for left_pos, r in enumerate(affected):
        vr = neighbors[r]
        for s in affected[left_pos:]:
            value = work[r].get(s, 0.0) - np.conj(vr) * neighbors[s] / pivot
            _set_hermitian_value(work, r, s, value, atol)


def _hermitian_2x2_eigenvalues(a, b, c):
    center = 0.5 * (a + c)
    radius = np.sqrt((0.5 * (a - c)) ** 2 + abs(b) ** 2)
    return center - radius, center + radius


def _ldl_2x2_update(work, i, j, rest, pivot, atol):
    a, b, c = pivot
    det = a * c - abs(b) ** 2
    vectors = {}
    for col in rest:
        first = work[i].get(col, 0.0)
        second = work[j].get(col, 0.0)
        if abs(first) > atol or abs(second) > atol:
            vectors[col] = (first, second)
    affected = sorted(vectors)
    for left_pos, r in enumerate(affected):
        xr, yr = vectors[r]
        conj_xr = np.conj(xr)
        conj_yr = np.conj(yr)
        for s in affected[left_pos:]:
            xs, ys = vectors[s]
            delta = (conj_xr * (c * xs - b * ys)
                     + conj_yr * (-np.conj(b) * xs + a * ys)) / det
            _set_hermitian_value(work, r, s, work[r].get(s, 0.0) - delta, atol)


def _set_hermitian_value(work, row, col, value, atol):
    if row == col:
        value = complex(value.real, 0.0)
        if abs(value.real) <= atol:
            work[row].pop(row, None)
        else:
            work[row][row] = value
        return
    if abs(value) <= atol:
        work[row].pop(col, None)
        work[col].pop(row, None)
        return
    work[row][col] = complex(value)
    work[col][row] = complex(np.conj(value))


def _drop_hermitian_indices(work, indices):
    for index in indices:
        for col in tuple(work[index]):
            work[col].pop(index, None)
        work[index].clear()
