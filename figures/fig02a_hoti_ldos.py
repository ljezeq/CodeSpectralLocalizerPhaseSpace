"""Fig. 2(a,b): LDOS of the 3D HOTI and of its partial spectral localizer.

The hinge modes of H are confined to the plane z = z_0 by the first confining
operator, which is what the partial localizer H_1 shows.  The red circle is
the shell where G_1 and G_2 both vanish.  Paper: the defaults (L = 16,
M/t = 2, Delta = lambda = 1, kappa = 0.3, sigma = 0.15, 80 lowest modes).

As everywhere else in this code base the low-energy density is the diagonal of
the Gaussian filter exp(-beta H^2) with beta = 1/(2 sigma^2), not a hard energy
cut-off: only the Gaussian filter is a smooth function of H and therefore has a
locality guarantee (see speclocalizer/wigner.py).
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

import _paths  # noqa: F401
from hoti_model import hoti_confining, hoti_hamiltonian
from speclocalizer import (build_lattice, embed, gaussian_weights, pauli,
                           reduction_step, use_paper_style)
from scipy.sparse import linalg as spla


def low_energy_density(operator, n_sites, sigma_energy, n_modes):
    """Site-resolved diagonal of exp(-H^2 / 2 sigma^2), computed on n_modes states.

    The spectrum has degenerate levels, inside which the eigenvectors returned by
    ARPACK are not orthogonal; they are orthonormalized before use (QR), and the
    start vector is fixed so that the result does not depend on the run.
    """
    start = np.random.default_rng(0).standard_normal(operator.shape[0]).astype(complex)
    values, vectors = spla.eigsh(operator, k=n_modes, sigma=0.0, which="LM", v0=start)
    order = np.argsort(np.abs(values))
    values, vectors = values[order], vectors[:, order]
    vectors, _ = np.linalg.qr(vectors)
    weights = gaussian_weights(values, sigma_energy=sigma_energy)
    if weights.max() <= 0.0:
        raise RuntimeError("the Gaussian filter has no weight")
    if weights[-1] > 1.0e-3:
        print(f"    warning: the {n_modes} states requested do not exhaust the "
              f"filter (last weight {weights[-1]:.1e}); increase --modes")
    internal = operator.shape[0] // n_sites
    density = np.zeros(n_sites)
    for weight, column in zip(weights, vectors.T):
        density += weight * np.sum(np.abs(column.reshape(n_sites, internal)) ** 2,
                                   axis=1)
    return density / density.max(), float(weights.sum())


def scatter_panel(axis, grid, density, title, shell=None, threshold=0.01):
    mask = density > threshold
    points = axis.scatter(grid[mask, 0], grid[mask, 1], grid[mask, 2],
                          c=density[mask], cmap="inferno",
                          s=40 * density[mask], alpha=0.85)
    if shell is not None:
        axis.plot(shell[0], shell[1], shell[2], color="red", linewidth=2.0)
    # The box is the lattice, not the shell: the circle is centred on a hinge
    # with R = L/2, so only the quarter of it that lies inside the sample is
    # drawn.  Without this the axes stretch to the circle and the lattice
    # shrinks, and the two panels no longer share a scale.
    for setter, column in ((axis.set_xlim, 0), (axis.set_ylim, 1),
                           (axis.set_zlim, 2)):
        setter(grid[:, column].min(), grid[:, column].max())
    axis.set_title(title)
    axis.set_xlabel(r"$x$")
    axis.set_ylabel(r"$y$")
    axis.set_zlabel(r"$z$")
    axis.set_box_aspect([1, 1, 1])
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--mass", type=float, default=2.0, help="M/t")
    parser.add_argument("--hopping", type=float, default=1.0)
    parser.add_argument("--soc", type=float, default=1.0, help="lambda")
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--kappa", type=float, default=0.3)
    parser.add_argument("--sigma-energy", type=float, default=0.15,
                        help="sigma of the Gaussian filter exp(-H^2/2 sigma^2)")
    parser.add_argument("--modes", type=int, default=80)
    parser.add_argument("--output", default="fig02a_hoti_ldos.png")
    args = parser.parse_args()

    use_paper_style()
    lattice = build_lattice((args.length,) * 3, boundary="open")
    hamiltonian = hoti_hamiltonian(lattice, args.mass * args.hopping,
                                   args.hopping, args.soc, args.delta)
    g1, g2, geometry = hoti_confining(lattice)

    partial, _ = reduction_step(hamiltonian, g1, args.kappa, chiral=None)

    n_sites = lattice.n_sites
    density_h, kept_h = low_energy_density(hamiltonian, n_sites, args.sigma_energy,
                                           args.modes)
    density_l, kept_l = low_energy_density(partial, n_sites, args.sigma_energy,
                                           args.modes)
    print(f"L={args.length}, M/t={args.mass:g}, kappa={args.kappa:g}, "
          f"sigma={args.sigma_energy:g}")
    print(f"  circle: centre=({geometry['x0']:.3g},{geometry['y0']:.3g}), "
          f"z0={geometry['z0']:.3g}, R={geometry['radius']:.3g}")
    print(f"  filter weight: H {kept_h:.2f}, H_1 {kept_l:.2f} (effective modes)")

    grid = lattice.site_grid()
    angle = np.linspace(0.0, 2.0 * np.pi, 200)
    shell = (geometry["x0"] + geometry["radius"] * np.cos(angle),
             geometry["y0"] + geometry["radius"] * np.sin(angle),
             np.full_like(angle, geometry["z0"]))

    np.savez(str(args.output).rsplit(".",1)[0] + ".npz", grid=grid, density_h=density_h, density_l=density_l, shell=np.array(shell))
    figure = plt.figure(figsize=(11, 4.6))
    axis_h = figure.add_subplot(121, projection="3d")
    axis_l = figure.add_subplot(122, projection="3d")
    points_h = scatter_panel(axis_h, grid, density_h, r"$H$: hinge modes")
    points_l = scatter_panel(axis_l, grid, density_l,
                             r"$H_1$: partial localizer", shell=shell)
    figure.colorbar(points_h, ax=axis_h, shrink=0.6, label="LDOS")
    figure.colorbar(points_l, ax=axis_l, shrink=0.6, label="LDOS")
    figure.tight_layout()
    figure.savefig(args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
