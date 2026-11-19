"""Wigner--Weyl transform of a Gaussian-filtered low-energy projector.

Convention used everywhere in this code base
--------------------------------------------
Instead of hand-picking a few eigenstates, every phase-space picture is built
from the Gaussian spectral filter

    rho_beta = exp(-beta H^2),        beta = 1 / (2 sigma^2),

i.e. each eigenstate is weighted by ``exp(-E^2 / 2 sigma^2)``.  This is the
convention with a theoretical locality guarantee: ``exp(-beta H^2)`` of a local
Hamiltonian has matrix elements decaying like ``exp(-d(x,y)^2 / 4 beta v^2)``
for a Lieb--Robinson velocity ``v``, so its Wigner transform really is a
phase-space density and not an artefact of which states were kept.  A hard
energy cut-off has no such guarantee: it is a discontinuous function of ``H``
and its kernel decays only algebraically.

Choosing sigma
--------------
The filter must keep the whole zero-mode multiplet and drop everything else:

    |E_0| << sigma << E_1,

where ``E_0`` are the (nearly) zero modes and ``E_1`` the first excited state.
Taking sigma too large mixes in the first excited level and smears the blob in
position; taking it too small does not lift the degeneracy of the zero modes
(all of them keep weight 1), but it does amplify the numerical noise of the
states that should have been discarded.  Every figure script prints the ratios
``|E_0|/sigma`` and ``sigma/E_1`` so the choice can be checked.
"""

from __future__ import annotations

import numpy as np

__all__ = ["gaussian_weights", "sigma_from_beta", "beta_from_sigma",
           "low_energy_density_matrix", "low_energy_density_matrix_full",
           "wigner_transform", "wigner_transform_open", "wigner_transform_onesided",
           "wigner_trace_norm"]


def beta_from_sigma(sigma_energy: float) -> float:
    """``beta = 1 / (2 sigma^2)`` in ``exp(-beta H^2)``."""
    return 0.5 / float(sigma_energy) ** 2


def sigma_from_beta(beta: float) -> float:
    """Inverse of :func:`beta_from_sigma`."""
    return float(np.sqrt(0.5 / float(beta)))


def gaussian_weights(energies, sigma_energy=None, beta=None) -> np.ndarray:
    """Diagonal of ``exp(-beta H^2)`` in the eigenbasis, given the energies."""
    if (sigma_energy is None) == (beta is None):
        raise ValueError("give exactly one of sigma_energy and beta")
    if beta is None:
        beta = beta_from_sigma(sigma_energy)
    return np.exp(-beta * np.asarray(energies, dtype=float) ** 2)


def low_energy_density_matrix(hamiltonian, n_sites: int, sigma_energy: float,
                              internal_dim: int, beta: float | None = None,
                              accumulator: np.ndarray | None = None) -> np.ndarray:
    """Position-space kernel of ``exp(-beta H^2)``, traced over internal indices.

    ``sigma_energy`` and ``beta`` are two names for the same filter, related by
    ``beta = 1 / (2 sigma^2)``; give one of them.  Pass ``accumulator`` to add
    several blocks (for instance several transverse momenta) into one kernel;
    in that case the caller normalises at the end.
    """
    energies, states = np.linalg.eigh(hamiltonian)
    weights = gaussian_weights(energies, None if beta is not None else sigma_energy,
                               beta)

    density = (np.zeros((n_sites, n_sites), dtype=complex)
               if accumulator is None else accumulator)
    for weight, state in zip(weights, states.T):
        if weight < 1.0e-12:
            continue
        site_state = state.reshape(n_sites, internal_dim)
        density += weight * (site_state @ site_state.conj().T)

    if accumulator is not None:
        return density
    trace = density.trace().real
    if trace <= 0.0:
        raise RuntimeError("the filtered projector has zero weight")
    return density / trace



def _deinterlace(wigner: np.ndarray) -> np.ndarray:
    """Average neighbouring midpoints to cancel the ``k -> k + pi`` image.

    On the doubled grid the integer midpoints only see even separations and the
    half-integer midpoints only odd ones, so each row taken alone is either
    pi-periodic or pi-antiperiodic in ``k``: a blob at ``k_0`` is copied at
    ``k_0 + pi`` with, respectively, the same and the opposite sign.  Averaging
    two neighbouring rows -- a smoothing over half a lattice site -- cancels the
    copy coherently and leaves the physical blob untouched.  Without it a mode
    at the graphene valley ``k = 2 pi / 3`` is drawn twice, once at ``-pi/3``.
    """
    return 0.5 * (wigner + np.roll(wigner, -1, axis=0))


def wigner_transform(density: np.ndarray, n_sites: int):
    """Periodic Wigner--Weyl transform on the doubled midpoint grid.

    The doubled grid is required on an even lattice: restricting the midpoint
    to integer sites aliases a state at ``k = pi`` onto ``k = 0``.

    Returns ``(x_values, k_values, W)`` with ``W`` real of shape
    ``(2 n_sites, 2 n_sites)``.
    """
    x_values = np.arange(2 * n_sites, dtype=float) / 2.0 - n_sites / 2.0
    k_values = np.linspace(-np.pi, np.pi, 2 * n_sites, endpoint=False)
    separations = np.arange(-n_sites + 1, n_sites + 1)

    wigner = np.zeros((2 * n_sites, 2 * n_sites), dtype=complex)
    phases = np.exp(-1j * np.outer(separations, k_values))   # (r, k)
    for midpoint in range(2 * n_sites):
        allowed = (midpoint + separations) % 2 == 0
        r_allowed = separations[allowed]
        left = ((midpoint + r_allowed) // 2) % n_sites
        right = ((midpoint - r_allowed) // 2) % n_sites
        wigner[midpoint] = density[left, right] @ phases[allowed]
    return x_values, k_values, _deinterlace(wigner.real)


def wigner_transform_open(density: np.ndarray, n_sites: int):
    """Wigner--Weyl transform on an open lattice, on the same doubled grid.

    Identical to :func:`wigner_transform` except that the separation is not
    wrapped around: pairs ``(x + r/2, x - r/2)`` that leave the lattice are
    dropped instead of being folded back.  On an open system the periodic
    formula produces spurious copies of the blobs at the two ends, because a
    midpoint near ``+-L/2`` with ``r ~ L`` reconnects the two edges; those
    copies are an artefact of the wrapping, not density.

    Returns ``(x_values, k_values, W)`` with ``W`` real of shape
    ``(2 n_sites, 2 n_sites)``.
    """
    x_values = np.arange(2 * n_sites, dtype=float) / 2.0 - n_sites / 2.0
    k_values = np.linspace(-np.pi, np.pi, 2 * n_sites, endpoint=False)
    separations = np.arange(-n_sites + 1, n_sites + 1)

    wigner = np.zeros((2 * n_sites, 2 * n_sites), dtype=complex)
    phases = np.exp(-1j * np.outer(separations, k_values))   # (r, k)
    for midpoint in range(2 * n_sites):
        left = (midpoint + separations) // 2
        right = (midpoint - separations) // 2
        allowed = (((midpoint + separations) % 2 == 0)
                   & (left >= 0) & (left < n_sites)
                   & (right >= 0) & (right < n_sites))
        wigner[midpoint] = density[left[allowed], right[allowed]] @ phases[allowed]
    return x_values, k_values, _deinterlace(wigner.real)


def wigner_transform_onesided(density: np.ndarray, n_sites: int, periodic: bool = False):
    """One-sided (standard-ordered) phase-space transform on a lattice.

        W(x, k) = sum_r rho(x + r, x) e^{-i k r}

    The midpoint is a lattice site, not a half-integer one, which removes the
    doubled grid of the symmetric convention along with its ``k -> k + pi``
    images, and generalises to higher dimensions without any bookkeeping.  The
    price is that ``W`` is complex; the marginals are still exact,
    ``sum_k W(x,k)/N = rho(x,x)`` and ``sum_x W(x,k)`` the occupation at ``k``,
    and the figures show ``|W|``.

    With ``periodic=False`` the separation is not wrapped around, which is what
    an open system requires.
    """
    x_values = np.arange(n_sites, dtype=float) - n_sites / 2.0
    k_values = np.linspace(-np.pi, np.pi, n_sites, endpoint=False)
    separations = np.arange(-(n_sites // 2), n_sites - n_sites // 2)
    phases = np.exp(-1j * np.outer(separations, k_values))

    wigner = np.empty((n_sites, n_sites), dtype=complex)
    for site in range(n_sites):
        shifted = site + separations
        if periodic:
            column = density[shifted % n_sites, site]
        else:
            column = np.zeros(separations.size, dtype=complex)
            inside = (shifted >= 0) & (shifted < n_sites)
            column[inside] = density[shifted[inside], site]
        wigner[site] = column @ phases
    return x_values, k_values, wigner


def low_energy_density_matrix_full(hamiltonian, n_sites: int, sigma_energy: float,
                                   internal_dim: int, accumulator=None):
    """Kernel of ``exp(-H^2 / 2 sigma^2)`` keeping the internal indices.

    Returns the ``(n_sites*d, n_sites*d)`` matrix rho_{alpha beta}(x, x') with the
    site index slowest (index = site*d + alpha).  Normalised to unit trace
    unless an accumulator is given.
    """
    energies, states = np.linalg.eigh(hamiltonian)
    weights = gaussian_weights(energies, sigma_energy)
    keep = weights > 1.0e-12
    block = (states[:, keep] * weights[keep]) @ states[:, keep].conj().T
    if accumulator is not None:
        accumulator += block
        return accumulator
    return block / np.trace(block).real


def wigner_trace_norm(density_full: np.ndarray, n_sites: int, internal_dim: int,
                      periodic: bool = False):
    """Trace norm Tr sqrt(W W^dag) of the matrix-valued one-sided Wigner transform.

        W_{alpha beta}(x, k) = sum_r rho_{alpha beta}(x + r, x) e^{-i k r},

    a d x d matrix at each phase-space point; its trace norm (sum of singular
    values) is positive and independent of the internal basis.
    """
    d = internal_dim
    blocks = {}
    for a in range(d):
        for b in range(d):
            x_values, k_values, w = wigner_transform_onesided(
                density_full[a::d, b::d], n_sites, periodic=periodic)
            blocks[a, b] = w
    stack = np.empty(w.shape + (d, d), dtype=complex)
    for (a, b), w in blocks.items():
        stack[..., a, b] = w
    norm = np.linalg.svd(stack, compute_uv=False).sum(axis=-1)
    return x_values, k_values, norm
