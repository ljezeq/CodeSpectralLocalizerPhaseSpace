"""Fig. 1: dimensional reduction of the spectral localizer (QWZ model).

Local density of states, in the (x, y) plane, of

    H       = (M + cos kx + cos ky) sz + sin kx sx + sin ky sy   (open boundaries)
    L_1D    = H (x) sigma_z + kappa X (x) sigma_x
    L_2D    = L_1D + kappa Y (x) sigma_y

H has one-dimensional gapless edge modes (d = 1), so d+1 = 2 confining
operators are needed and the Clifford representation is two dimensional.  Each
confining operator removes one dimension from the gapless set: the edge modes
of H become a single chiral mode along one edge for L_1D, and a point for L_2D.

The LDOS is the Gaussian-weighted sum |psi_n(r)|^2 exp(-E_n^2 / 2 sigma^2)
traced over the internal degrees of freedom.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

import _paths  # noqa: F401
from speclocalizer import (build_lattice, embed, half_signature, localizer_gap,
                           pauli, use_paper_style)
from speclocalizer.style import LABEL_FONTSIZE, TITLE_FONTSIZE


def qwz_hamiltonian(lattice, mass, hopping):
    """QWZ Hamiltonian with the hopping convention of the original script."""
    _, sigma_x, sigma_y, sigma_z = pauli()
    tx, ty = lattice.shifts
    cos_part = mass * lattice.identity + hopping * (lattice.cos(0) + lattice.cos(1))
    return (embed(cos_part, sigma_z)
            - hopping * embed(lattice.sin(0), sigma_x)
            - hopping * embed(lattice.sin(1), sigma_y)).tocsr()


def ldos_map(matrix, n_sites, shape, sigma):
    """Gaussian-weighted local density of states, summed over internal degrees."""
    dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    energies, states = np.linalg.eigh(dense)
    weights = np.exp(-(energies ** 2) / (2.0 * sigma ** 2))
    density = (np.abs(states) ** 2) @ weights
    internal = dense.shape[0] // n_sites
    return density.reshape(n_sites, internal).sum(axis=1).reshape(shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--mass", type=float, default=1.0, help="QWZ mass m")
    parser.add_argument("--hopping", type=float, default=1.0)
    parser.add_argument("--kappa", type=float, default=0.3)
    parser.add_argument("--sigma", type=float, default=0.3,
                        help="energy width of the Gaussian LDOS window")
    parser.add_argument("--output", default="fig01_qwz_reduction.png")
    args = parser.parse_args()

    use_paper_style()
    _, sigma_x, sigma_y, sigma_z = pauli()
    lattice = build_lattice((args.length, args.length), boundary="open")
    hamiltonian = qwz_hamiltonian(lattice, args.mass, args.hopping)

    identity2 = np.eye(2, dtype=complex)
    x_op = embed(lattice.positions[0], identity2)
    y_op = embed(lattice.positions[1], identity2)

    partial = embed(hamiltonian, sigma_z) + args.kappa * embed(x_op, sigma_x)
    full = partial + args.kappa * embed(y_op, sigma_y)

    index, signature, zeros = half_signature(full)
    print(f"L={args.length}, m={args.mass:g}, t={args.hopping:g}, kappa={args.kappa:g}")
    print(f"  localizer gap  : {localizer_gap(full):.4g}")
    print(f"  half signature : {index}  (Sig={signature}, {zeros} zero modes)")

    shape = (args.length, args.length)
    maps = [ldos_map(operator, lattice.n_sites, shape, args.sigma)
            for operator in (hamiltonian, partial, full)]
    np.savez(str(args.output).rsplit(".",1)[0] + ".npz", maps=np.array(maps), coordinates=lattice.coordinates[0])
    vmax = max(density.max() for density in maps)
    titles = [r"$H$: edge modes",
              r"$H_0$: two end modes",
              r"$H'$: gapped localizer"]

    coordinates = lattice.coordinates[0]
    extent = [coordinates[0], coordinates[-1], coordinates[0], coordinates[-1]]
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
    images = []
    for axis, density, title in zip(axes, maps, titles):
        # density[i, j] is site (x_i, y_j); transpose so that x is horizontal.
        images.append(axis.imshow(density.T, extent=extent, origin="lower",
                                  cmap="magma", vmin=0.0, vmax=vmax))
        axis.set_title(title, fontsize=TITLE_FONTSIZE)
        axis.set_xlabel(r"$x$", fontsize=LABEL_FONTSIZE)
    axes[0].set_ylabel(r"$y$", fontsize=LABEL_FONTSIZE)

    figure.subplots_adjust(right=0.87)
    bar_axis = figure.add_axes([0.89, 0.15, 0.015, 0.7])
    bar = figure.colorbar(images[0], cax=bar_axis)
    bar.set_label("Local density of states", fontsize=LABEL_FONTSIZE)
    figure.savefig(args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
