"""Operators of the semimetal models of Fig. 3.

* Graphene in oblique coordinates (sublattices A, B; Pauli matrices sigma):

      H = -t [ (1 + Tx^dag + Ty^dag) (x) sigma_+ + h.c. ],   f(k) = 1 + e^{i kx} + e^{i ky},

  with Dirac points K = (2pi/3, -2pi/3) and K' = -K, and the shell around K

      G_3 = c - (e^{-i Kx} Tx^dag + h.c.) - (e^{-i Ky} Ty^dag + h.c.),
      g_3(k) = c - 2 cos(kx - Kx) - 2 cos(ky - Ky)       (c = 2 in the paper).

* The minimal Dirac model H = sin kx sigma_x + sin ky sigma_y, with four cones and
  the shell G_3 = c - 2 (cos kx + cos ky) around the cone at the origin (c = 3).

* Rhombohedral n-layer graphene: n copies of the graphene layer, B_l coupled to
  A_{l+1} on the same site with amplitude gamma_1.

G_1 = X and G_2 = Y in all cases.  The chiral symmetry is the sublattice sigma_z;
the time-reversal localizer of the Dirac model (class AII, T = sigma_y K) uses
the four anticommuting matrices of time_reversal_gammas().
"""

import numpy as np
from scipy import sparse

import _paths  # noqa: F401
from speclocalizer import embed, pauli


VALLEY = (2.0 * np.pi / 3.0, -2.0 * np.pi / 3.0)


def graphene_operators(lattice, g3_offset, valley=VALLEY):
    """Return H, G_1 = X, G_2 = Y and G_3 for the honeycomb lattice.

    In oblique coordinates the nearest-neighbour Hamiltonian is

        H = -t [ (1 + Tx^dag + Ty^dag) (x) sigma_+ + h.c. ],

    with symbol f(k) = 1 + e^{i kx} + e^{i ky}, which vanishes at the two
    Dirac points K = (2pi/3, -2pi/3) and K' = -K.  A shell around one of them
    is the same quadratic-like form as before, recentred on that valley:

        G_3 = c - (e^{-i Kx} Tx^dag + h.c.) - (e^{-i Ky} Ty^dag + h.c.),
        g_3(k) = c - 2 cos(kx - Kx) - 2 cos(ky - Ky),

    which for c = 3 is -1 at K, +5 at K', and its zero set is a closed curve
    of radius ~1 around K only.
    """
    sigma_0, sigma_x, sigma_y, _ = pauli()
    sigma_plus = 0.5 * (sigma_x + 1j * sigma_y)
    tx, ty = lattice.shifts
    hopping = lattice.identity + tx.conj().T + ty.conj().T
    hamiltonian = (-embed(hopping, sigma_plus)
                   - embed(hopping.conj().T, sigma_plus.conj().T))

    g1 = embed(lattice.positions[0], sigma_0)
    g2 = embed(lattice.positions[1], sigma_0)
    shifted = (np.exp(-1j * valley[0]) * tx.conj().T
               + np.exp(-1j * valley[1]) * ty.conj().T)
    g3_site = g3_offset * lattice.identity - (shifted + shifted.conj().T)
    return (hamiltonian.tocsr(), g1.tocsr(), g2.tocsr(),
            embed(g3_site, sigma_0).tocsr())

def semimetal_operators(lattice, g3_offset):
    """Return H, G_1 = X, G_2 = Y and G_3."""
    sigma_0, sigma_x, sigma_y, _ = pauli()
    # (T^dag - T)/2i = -lattice.sin has symbol sin k (T|x> = |x+1>), as in Eq. (minimaldirac)
    hamiltonian = -(embed(lattice.sin(0), sigma_x) + embed(lattice.sin(1), sigma_y))
    g1 = embed(lattice.positions[0], sigma_0)
    g2 = embed(lattice.positions[1], sigma_0)
    g3_site = g3_offset * lattice.identity - 2.0 * (lattice.cos(0) + lattice.cos(1))
    return hamiltonian.tocsr(), g1.tocsr(), g2.tocsr(), embed(g3_site, sigma_0).tocsr()

def time_reversal_gammas():
    """The four anticommuting matrices used for the class-AII reduction."""
    alpha_0, alpha_x, alpha_y, alpha_z = pauli()
    beta_0, beta_x, _, beta_z = pauli()
    return [np.kron(alpha_x, beta_0), np.kron(alpha_z, beta_0),
            np.kron(alpha_y, beta_x), np.kron(alpha_y, beta_z)]

def time_reversal_localizer(hamiltonian, confining, kappa):
    gammas = time_reversal_gammas()
    localizer = embed(hamiltonian, gammas[0])
    for operator, gamma in zip(confining, gammas[1:]):
        localizer = localizer + kappa * embed(operator, gamma)
    return localizer.tocsr()


def multilayer(lattice, layers, gamma1, lam, offset=2.0):
    """H + lam sum_l sigma_x^(l) for n = layers graphene layers, with G_1, G_2, G_3 and C.

    Internal index order: (layer, sublattice); C = diag(+1, -1, +1, -1, ...).
    """
    h1, gx, gy, g3 = graphene_operators(lattice, offset)       # internal dim 2
    n_sites = lattice.n_sites
    sx = np.array([[0, 1], [1, 0]], complex)
    h1 = h1 + lam * embed(lattice.identity, sx)
    H = embed_layers(h1, n_sites, layers)
    inter = np.zeros((2 * layers, 2 * layers), complex)
    for l in range(layers - 1):
        b, a = 2 * l + 1, 2 * (l + 1)
        inter[b, a] = inter[a, b] = gamma1
    H = (H + embed(lattice.identity, inter)).tocsr()
    lift = lambda g: embed_layers(g, n_sites, layers)
    C = embed(lattice.identity, np.diag([1.0, -1.0] * layers).astype(complex))
    return H, lift(gx), lift(gy), lift(g3), C


def embed_layers(op2, n_sites, layers):
    """op on (site x sublattice) -> (site x layer x sublattice), diagonal in layer."""
    op2 = sparse.csr_matrix(op2)
    big = sparse.kron(op2, sparse.identity(layers), format="csr")   # order (site, sub, layer)
    perm = np.arange(n_sites * 2 * layers).reshape(n_sites, 2, layers).transpose(0, 2, 1).ravel()
    Pm = sparse.csr_matrix((np.ones(perm.size), (np.arange(perm.size), perm)))
    return (Pm @ big @ Pm.T).tocsr()
