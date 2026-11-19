"""Lattice operators: Pauli matrices, translations and position operators.

Index convention
----------------
For a lattice of shape ``(L_0, L_1, ..., L_{n-1})`` the site index is built as

    kron(A_0, A_1, ..., A_{n-1})

so that **axis 0 is the slowest index and axis n-1 the fastest**.  Internal
degrees of freedom are always appended last, i.e. an operator on the full
Hilbert space is ``kron(site_operator, internal_matrix)``.  Every routine in
this package follows this convention; do not mix it with the opposite one.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

__all__ = [
    "pauli",
    "shift_open",
    "shift_periodic",
    "position_values",
    "LatticeOperators",
    "build_lattice",
    "embed",
]


def pauli() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(sigma_0, sigma_x, sigma_y, sigma_z)`` as dense complex arrays."""
    sigma_0 = np.eye(2, dtype=complex)
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    return sigma_0, sigma_x, sigma_y, sigma_z


def shift_open(length: int) -> sparse.csr_matrix:
    """Open-boundary translation ``T |x> = |x+1>`` (no wrap-around)."""
    return sparse.diags(
        [np.ones(length - 1, dtype=complex)],
        offsets=[-1],
        shape=(length, length),
        format="csr",
    )


def shift_periodic(length: int) -> sparse.csr_matrix:
    """Periodic translation ``T |x> = |x+1>`` including the wrap-around."""
    rows = [(x + 1) % length for x in range(length)]
    cols = list(range(length))
    data = np.ones(length, dtype=complex)
    return sparse.csr_matrix((data, (rows, cols)), shape=(length, length))


def position_values(length: int, centered: bool = True) -> np.ndarray:
    """Site coordinates, centred on the lattice by default."""
    values = np.arange(length, dtype=float)
    if centered:
        values -= (length - 1) / 2
    return values


def _kron(*matrices):
    result = matrices[0]
    for matrix in matrices[1:]:
        result = sparse.kron(result, matrix, format="csr")
    return result


class LatticeOperators:
    """Translations and position operators of a hypercubic lattice.

    Attributes
    ----------
    shifts : list of sparse matrices
        ``shifts[i]`` translates by one site along axis ``i``.
    positions : list of sparse matrices
        ``positions[i]`` is the (centred) coordinate operator along axis ``i``.
    coordinates : list of 1D arrays
        The coordinate values along each axis, in the same convention.
    """

    def __init__(self, shape: tuple[int, ...], boundary: str = "open"):
        if boundary not in ("open", "periodic"):
            raise ValueError("boundary must be 'open' or 'periodic'")
        self.shape = tuple(shape)
        self.boundary = boundary
        self.n_sites = int(np.prod(self.shape))

        maker = shift_open if boundary == "open" else shift_periodic
        identities = [sparse.identity(n, dtype=complex, format="csr") for n in self.shape]

        self.shifts = []
        self.positions = []
        self.coordinates = []
        for axis, length in enumerate(self.shape):
            factors_shift = list(identities)
            factors_shift[axis] = maker(length)
            self.shifts.append(_kron(*factors_shift))

            values = position_values(length)
            self.coordinates.append(values)
            factors_position = list(identities)
            factors_position[axis] = sparse.diags(values.astype(complex), format="csr")
            self.positions.append(_kron(*factors_position))

        self.identity = sparse.identity(self.n_sites, dtype=complex, format="csr")

    def cos(self, axis: int) -> sparse.csr_matrix:
        """``(T + T^dagger)/2``, whose symbol is ``cos(k_axis)``."""
        shift = self.shifts[axis]
        return 0.5 * (shift + shift.conj().T)

    def sin(self, axis: int) -> sparse.csr_matrix:
        """``(T - T^dagger)/(2i)``.  With ``T|x> = |x+1>`` its symbol is ``-sin(k_axis)``."""
        shift = self.shifts[axis]
        return (shift - shift.conj().T) / 2j

    def site_grid(self) -> np.ndarray:
        """Return an ``(n_sites, ndim)`` array of coordinates in site order."""
        meshes = np.meshgrid(*self.coordinates, indexing="ij")
        return np.stack([mesh.ravel() for mesh in meshes], axis=1)


def build_lattice(shape, boundary: str = "open") -> LatticeOperators:
    """Convenience wrapper around :class:`LatticeOperators`."""
    return LatticeOperators(shape, boundary)


def embed(site_operator, internal_matrix) -> sparse.csr_matrix:
    """Tensor a site-space operator with an internal matrix (internal last)."""
    return sparse.kron(site_operator, sparse.csr_matrix(internal_matrix), format="csr")
