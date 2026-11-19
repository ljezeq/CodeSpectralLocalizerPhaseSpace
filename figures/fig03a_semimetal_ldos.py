"""Fig. 3(a-h): where the low-energy weight of a 2D semimetal sits, for H and for H_1.

Graphene (--model graphene, two cones) or the minimal Dirac model (--model dirac,
four cones).  For graphene

    H = -t [ (1 + Tx^dag + Ty^dag) (x) sigma_+ + h.c. ]

has two Dirac points in k, and its low-energy states are extended in x and y.  The
first two confining operators remove the extension in position:

    H_1 = (H + kappa sigma_z X) (x) sigma_z' + kappa Y (x) sigma_y'

is the partial localizer after the G_1 = X and G_2 = Y steps.  G_3 is not included;
it selects one of the spots seen in momentum, and its zero set g_3 = 0 is drawn
there.

The lattice is periodic, so that the density of H is uniform and its weight in
momentum sits on point-like spots, broadened only by the Gaussian filter
exp(-H^2 / 2 sigma^2) (epsilon = sqrt(2) sigma).  With rho its kernel traced over
the internal indices,

    n(r)  = rho(r, r),
    n(k)  = sum_{r, r'} e^{-i k (r - r')} rho(r, r') / N.

Paper: --lx 29 --ly 29 --kappa 0.1 --sigma 0.12 (epsilon = 0.17), both models.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

import _paths  # noqa: F401
from semimetal_models import VALLEY, graphene_operators, semimetal_operators

MODEL = "graphene"
from speclocalizer import (build_lattice, embed, gaussian_weights, pauli,
                           reduction_step, use_paper_style)
from speclocalizer.style import DENSITY_CMAP, SHELL_COLOR


def filtered_kernel(operator, n_sites, sigma):
    """rho(r, r') = <r| exp(-H^2 / 2 sigma^2) |r'>, traced over internal indices."""
    energies, states = np.linalg.eigh(operator.toarray())
    weights = gaussian_weights(energies, sigma_energy=sigma)
    internal = operator.shape[0] // n_sites
    kernel = np.zeros((n_sites, n_sites), dtype=complex)
    for weight, column in zip(weights, states.T):
        if weight < 1.0e-10:
            continue
        block = column.reshape(n_sites, internal)
        kernel += weight * (block @ block.conj().T)
    return kernel / np.trace(kernel).real


def momentum_weight(kernel, coordinates, k_grid):
    """n(k) on an arbitrary k grid, from the real-space kernel."""
    x, y = coordinates
    kx, ky = np.meshgrid(k_grid, k_grid, indexing="ij")
    shape = kx.shape
    phases = np.exp(1j * (np.outer(kx.ravel(), x) + np.outer(ky.ravel(), y)))
    weight = np.einsum("qi,ij,qj->q", phases.conj(), kernel, phases).real
    return kx, ky, weight.reshape(shape) / weight.max()


def bloch_weight(k_grid, sigma):
    """exp(-beta E^2) of the translation-invariant model, on a fine k grid."""
    kx, ky = np.meshgrid(k_grid, k_grid, indexing="ij")
    if MODEL == "dirac":
        energy = np.sqrt(np.sin(kx) ** 2 + np.sin(ky) ** 2)
    else:
        energy = np.abs(1.0 + np.exp(1j * kx) + np.exp(1j * ky))
    return kx, ky, np.exp(-0.5 * (energy / sigma) ** 2)


GRAPHENE_OFFSET = 2.0  # graphene shell offset (Dirac model keeps 3)


def shell(kx, ky):
    if MODEL == "dirac":
        return 3.0 - 2.0 * (np.cos(kx) + np.cos(ky))
    return GRAPHENE_OFFSET - 2.0 * np.cos(kx - VALLEY[0]) - 2.0 * np.cos(ky - VALLEY[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lx", type=int, default=29)
    parser.add_argument("--ly", type=int, default=29)
    parser.add_argument("--kappa", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=0.12)
    parser.add_argument("--k-points", type=int, default=161)
    parser.add_argument("--model", choices=("graphene", "dirac"), default="graphene",
                        help="graphene (two cones) or sin kx sx + sin ky sy (four cones)")
    parser.add_argument("--output", default="fig03a_semimetal_ldos.png")
    args = parser.parse_args()
    global MODEL
    MODEL = args.model

    use_paper_style()
    _, _, _, sigma_z = pauli()
    lattice = build_lattice((args.lx, args.ly), boundary="periodic")
    if MODEL == "dirac":
        hamiltonian, g1, g2, _ = semimetal_operators(lattice, 3.0)
    else:
        hamiltonian, g1, g2, _ = graphene_operators(lattice, 3.0)
    chiral = embed(lattice.identity, sigma_z)

    # G_1 = X, then G_2 = Y.  G_3 is deliberately left out.
    partial, next_chiral = reduction_step(hamiltonian, g1, args.kappa, chiral=chiral)
    partial, _ = reduction_step(partial, g2, args.kappa, chiral=None)

    n_sites = lattice.n_sites
    kernel_h = filtered_kernel(hamiltonian, n_sites, args.sigma)
    kernel_l = filtered_kernel(partial, n_sites, args.sigma)
    density_h = np.real(np.diag(kernel_h))
    density_l = np.real(np.diag(kernel_l))

    grid = lattice.site_grid()
    coordinates = (grid[:, 0], grid[:, 1])
    centre = (np.abs(coordinates[0]) <= 3) & (np.abs(coordinates[1]) <= 3)
    print(f"Lx={args.lx}, Ly={args.ly} (periodic), kappa={args.kappa:g}, "
          f"sigma={args.sigma:g}")
    print(f"  density of H: min/max = {density_h.min() / density_h.max():.6f} "
          "(uniform on a periodic lattice)")
    print(f"  weight within 3 sites of the origin: H {density_h[centre].sum():.3f}, "
          f"H_1 {density_l[centre].sum():.3f}")

    k_grid = np.linspace(-np.pi, np.pi, args.k_points)
    kx, ky, weight_h = bloch_weight(k_grid, args.sigma)
    _, _, weight_l = momentum_weight(kernel_l, coordinates, k_grid)

    np.savez(str(args.output).rsplit(".",1)[0] + ".npz", density_h=density_h, density_l=density_l, weight_h=weight_h, weight_l=weight_l, kx=kx, ky=ky, shell=shell(kx, ky), lx=args.lx, ly=args.ly)
    figure, axes = plt.subplots(2, 2, figsize=(8.2, 7.2))
    extent_r = [-args.lx / 2, args.lx / 2, -args.ly / 2, args.ly / 2]
    for axis, density, title in ((axes[0, 0], density_h, r"$H$"),
                                 (axes[0, 1], density_l, r"$H_1$")):
        image = axis.imshow((density / density.max()).reshape(args.lx, args.ly).T, origin="lower",
                            extent=extent_r, cmap=DENSITY_CMAP, vmin=0.0)
        figure.colorbar(image, ax=axis, fraction=0.046)
        axis.set_xlabel(r"$x$")
        axis.set_ylabel(r"$y$")
        axis.set_title(title + r":  $n(x,y)$")

    extent_k = [-np.pi, np.pi, -np.pi, np.pi]
    for axis, weight, title in ((axes[1, 0], weight_h, r"$H$"),
                                (axes[1, 1], weight_l, r"$H_1$")):
        image = axis.imshow((weight / weight.max()).T, origin="lower",
                            extent=extent_k, cmap=DENSITY_CMAP, vmin=0.0)
        figure.colorbar(image, ax=axis, fraction=0.046)
        axis.contour(kx, ky, shell(kx, ky), levels=[0.0], colors=SHELL_COLOR,
                     linewidths=1.6)
        axis.set_xlabel(r"$k_x$")
        axis.set_ylabel(r"$k_y$")
        axis.set_xticks([-np.pi, 0, np.pi])
        axis.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
        axis.set_yticks([-np.pi, 0, np.pi])
        axis.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
        axis.set_title(title + r":  $n(k_x,k_y)$")

    figure.tight_layout()
    figure.savefig(args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
