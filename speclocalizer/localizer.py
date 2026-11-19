"""Construction of spectral localizers.

The paper writes the fully reduced localizer as

    H' = H Gamma_0 + kappa sum_{j=1}^{d+1} G_j Gamma_j          (no chiral symmetry)
    H'  = H        + kappa sum_{j=1}^{d+1} (G_j C) Gamma_j     (chiral symmetry C)

The chiral form is the one that reproduces Eq. (11) of the paper:

    H'^2 = H^2 + kappa^2 sum_j G_j^2
           + kappa sum_j [H, G_j] C Gamma_j
           + kappa^2 sum_{i<j} [G_i, G_j] Gamma_i Gamma_j,

because {H, G_j C} = [H, G_j] C when {H, C} = 0 and [C, G_j] = 0.  Writing the
Hamiltonian term as "i H C" instead, as in Eq. (8b) of the current draft, gives
an anticommutator {H, G_j} in the cross term and does not close.

where ``d`` is the dimension of the gapless set of ``H`` and the ``Gamma``'s
anticommute.  Equivalently, the localizer is built recursively by

    H_{d-(j+1)} = H_{d-j} + kappa C G_{j+1}                     (chiral step)
    H_{d-(j+1)} = H_{d-j} (x) sigma_z + kappa G_{j+1} (x) sigma_y   (non-chiral step)

The chiral step consumes the chiral symmetry, the non-chiral step creates a new
one (``sigma_x``), so the two alternate along the recursion.  Only the
non-chiral step doubles the Hilbert space, hence the size of the Clifford
representation:

    dim = 2 ** ceil((d+1)/2)   if H has no chiral symmetry,
    dim = 2 ** floor((d+1)/2)  if it has one.

When the total number of anticommuting matrices (``d+2`` without chiral
symmetry, ``d+1`` with) is odd, one of them is fixed by the product of the
others; this is the case for every example in the paper except the
time-reversal Dirac semimetal.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from .lattice import embed, pauli

__all__ = [
    "clifford_generators",
    "chirality_element",
    "spectral_localizer",
    "reduction_step",
    "localizer_gap",
    "smallest_eigenvalues",
]


def clifford_generators(count: int) -> list[np.ndarray]:
    """Return ``count`` anticommuting Hermitian matrices of minimal size.

    They satisfy ``{G_i, G_j} = 2 delta_ij`` and act on ``2**(count//2)``
    dimensions, which is the minimal irreducible representation.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    _, sigma_x, sigma_y, sigma_z = pauli()

    if count % 2:
        generators = [np.ones((1, 1), dtype=complex)]
    else:
        generators = []

    while len(generators) < count:
        dimension = generators[0].shape[0] if generators else 1
        identity = np.eye(dimension, dtype=complex)
        generators = [np.kron(g, sigma_z) for g in generators]
        generators.append(np.kron(identity, sigma_x))
        generators.append(np.kron(identity, sigma_y))

    return generators[:count]


def chirality_element(generators: list[np.ndarray]) -> np.ndarray:
    """Hermitian product ``i**(n(n-1)/2) G_1 ... G_n`` of the generators.

    It squares to the identity, and anticommutes with every generator when
    ``n`` is even (it is proportional to the identity when ``n`` is odd).
    """
    n = len(generators)
    product = np.eye(generators[0].shape[0], dtype=complex)
    for generator in generators:
        product = product @ generator
    return (1j ** (n * (n - 1) // 2)) * product


def spectral_localizer(hamiltonian, confining, kappa: float, chiral=None):
    """Build the fully reduced spectral localizer.

    Parameters
    ----------
    hamiltonian : sparse or dense matrix
        The Hamiltonian ``H``.
    confining : sequence of matrices
        The ``d+1`` confining operators ``G_j``, in the order they appear in
        the paper.
    kappa : float
        Localizer strength.
    chiral : matrix or None
        The chiral operator ``C`` of ``H`` if it has one, otherwise ``None``.

    Returns
    -------
    sparse matrix
        ``H'``, Hermitian and (for a suitable ``kappa``) invertible.
    """
    hamiltonian = sparse.csr_matrix(hamiltonian)
    confining = [sparse.csr_matrix(g) for g in confining]
    n_confining = len(confining)

    if chiral is None:
        generators = clifford_generators(n_confining + 1)
        localizer = embed(hamiltonian, generators[0])
        generators = generators[1:]
    else:
        chiral = sparse.csr_matrix(chiral)
        generators = clifford_generators(n_confining)
        dimension = generators[0].shape[0] if generators else 1
        localizer = embed(hamiltonian, np.eye(dimension, dtype=complex))
        confining = [(g @ chiral).tocsr() for g in confining]

    for operator, generator in zip(confining, generators):
        localizer = localizer + kappa * embed(operator, generator)

    return localizer.tocsr()


def reduction_step(hamiltonian, confining, kappa: float, chiral=None):
    """One step of the recursion, returning ``(H_next, chiral_next)``.

    Useful to build the *partial* localizers ``H_0``, ``H_1``, ... shown in the
    figures, which are the intermediate Hamiltonians of the recursion.
    """
    hamiltonian = sparse.csr_matrix(hamiltonian)
    confining = sparse.csr_matrix(confining)
    _, sigma_x, sigma_y, sigma_z = pauli()

    if chiral is not None:
        chiral = sparse.csr_matrix(chiral)
        return (hamiltonian + kappa * chiral @ confining).tocsr(), None

    nxt = embed(hamiltonian, sigma_z) + kappa * embed(confining, sigma_y)
    dimension = hamiltonian.shape[0]
    next_chiral = embed(sparse.identity(dimension, dtype=complex, format="csr"), sigma_x)
    return nxt.tocsr(), next_chiral.tocsr()


def smallest_eigenvalues(matrix, k: int = 8, tol: float = 1.0e-9, dense_limit: int = 2000):
    """Eigenvalues closest to zero, dense for small matrices, ARPACK otherwise."""
    n = matrix.shape[0]
    if n <= dense_limit or not sparse.issparse(matrix):
        dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
        values = np.linalg.eigvalsh(dense)
        order = np.argsort(np.abs(values))
        return values[order[: min(k, n)]]

    k = min(k, n - 2)
    try:
        return spla.eigsh(matrix, k=k, sigma=0.0, which="LM",
                          return_eigenvectors=False, tol=tol)
    except RuntimeError:
        return spla.eigsh(matrix, k=k, which="SM", return_eigenvectors=False, tol=tol)


def localizer_gap(matrix, k: int = 8, tol: float = 1.0e-9) -> float:
    """Smallest ``|eigenvalue|`` of a Hermitian localizer."""
    return float(np.min(np.abs(smallest_eigenvalues(matrix, k=k, tol=tol))))
