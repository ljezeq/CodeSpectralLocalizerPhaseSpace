"""3D higher-order topological insulator of Schindler et al., Sci. Adv. 4 (2018).

    H(k) = (M + t sum_i cos k_i) tau_z sigma_0
         + lambda sum_i sin k_i tau_x sigma_i
         + Delta (cos k_x - cos k_y) tau_y sigma_0

with open boundary conditions in all three directions.  The confining
operators of the paper are

    G_1 = Z - z_0
    G_2 = [(X - x_0)^2 + (Y - y_0)^2 - R^2] / (2 R)

The 1/(2R) factor is not cosmetic: near the circle it turns G_2 into the
signed radial distance, so that the localizer term stays of order one instead
of growing like L^2 when the lattice is enlarged.

The sine terms are built with lattice.sin = (T - T^dag)/2i, whose symbol is
-sin k for T|x> = |x+1>: the operator is H(k) above with lambda -> -lambda.
This sets the sign of Sig/2 in Fig. 2(c), -1 in the topological phase; with
the opposite sign of lambda it is +1 (checked at L = 6).
"""

import _paths  # noqa: F401

import numpy as np
from scipy import sparse

from speclocalizer import build_lattice, embed, pauli


def internal_matrices():
    sigma_0, sigma_x, sigma_y, sigma_z = pauli()
    tau_0, tau_x, tau_y, tau_z = pauli()
    return {
        "identity": np.kron(tau_0, sigma_0),
        "mass": np.kron(tau_z, sigma_0),
        "sin": [np.kron(tau_x, sigma_x), np.kron(tau_x, sigma_y), np.kron(tau_x, sigma_z)],
        "delta": np.kron(tau_y, sigma_0),
    }


def hoti_hamiltonian(lattice, mass, hopping, soc, delta):
    """Return H for the given M, t, lambda, Delta."""
    matrices = internal_matrices()
    cosines = lattice.cos(0) + lattice.cos(1) + lattice.cos(2)
    hamiltonian = embed(mass * lattice.identity + hopping * cosines, matrices["mass"])
    for axis in range(3):
        hamiltonian = hamiltonian + soc * embed(lattice.sin(axis), matrices["sin"][axis])
    hamiltonian = hamiltonian + delta * embed(lattice.cos(0) - lattice.cos(1),
                                              matrices["delta"])
    return hamiltonian.tocsr()


def hoti_confining(lattice, radius=None, centre=None, z0=0.0, normalize=True):
    """Return ``(G_1, G_2, geometry)`` with the circle linking one hinge."""
    matrices = internal_matrices()
    length = lattice.shape[0]
    edge = lattice.coordinates[0][-1]
    x0, y0 = (edge, edge) if centre is None else centre
    radius = length / 2 if radius is None else radius

    identity = lattice.identity
    x_op, y_op, z_op = lattice.positions
    g1_site = z_op - z0 * identity
    dx = x_op - x0 * identity
    dy = y_op - y0 * identity
    g2_site = dx @ dx + dy @ dy - radius ** 2 * identity
    if normalize:
        g2_site = g2_site / (2.0 * radius)

    geometry = {"x0": x0, "y0": y0, "z0": z0, "radius": radius, "normalized": normalize}
    return (embed(g1_site, matrices["identity"]).tocsr(),
            embed(g2_site, matrices["identity"]).tocsr(),
            geometry)


def doubled_hamiltonian(hamiltonian):
    """H_tilde = diag(H, conj(H)) with T = mu_y K, i.e. class AII."""
    mu_0, _, _, mu_z = pauli()
    real = sparse.csr_matrix(hamiltonian.real)
    imag = sparse.csr_matrix(hamiltonian.imag)
    return (embed(real, mu_0) + embed(imag, 1j * mu_z)).tocsr()


def majorana_rotation(n_base):
    """Takagi rotation for U = mu_y (x) eta_y, in the basis order (mu, eta)."""
    q4 = np.zeros((4, 4), dtype=complex)
    root2 = np.sqrt(2.0)
    q4[:, 0] = [1 / root2, 0, 0, -1 / root2]
    q4[:, 1] = [0, 1 / root2, 1 / root2, 0]
    q4[:, 2] = [1j / root2, 0, 0, 1j / root2]
    q4[:, 3] = [0, 1j / root2, -1j / root2, 0]
    return sparse.kron(sparse.identity(n_base, dtype=complex, format="csr"),
                       sparse.csr_matrix(q4), format="csr")


def onsite_disorder(lattice, strength, seed, internal_dim=4):
    """Uniform on-site potential in [-W, W], identical on the internal indices.

    It is real and diagonal, so it commutes with the doubling and preserves the
    time reversal T = mu_y K of the doubled model: the Z2 index stays defined.
    """
    if strength == 0.0:
        return None
    rng = np.random.default_rng(seed)
    values = strength * rng.uniform(-1.0, 1.0, lattice.n_sites)
    return sparse.kron(sparse.diags(values.astype(complex), format="csr"),
                       sparse.identity(internal_dim, dtype=complex, format="csr"),
                       format="csr")

