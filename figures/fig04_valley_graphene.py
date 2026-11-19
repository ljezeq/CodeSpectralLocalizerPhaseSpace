"""Fig. 4: valley-Hall graphene ribbon, zero-mode Wigner transform and index.

Honeycomb ribbon in oblique coordinates with a smooth staggered mass, Eq. (2Dvalley)
of the paper,

    m(x) = m0 tanh(x / L_0) + lambda,     L_0 = Lx,

    H = m(X) (x) sigma_z
        - t [ (1 + Tx^dag + Ty^dag) (x) sigma_+ + h.c. ].

The mass changes sign at a single smooth wall at x = 0.

Left panel: Tr|W(x,k_x)|, the trace norm (over the sublattice index) of the
one-sided Wigner--Weyl transform of the Gaussian filter exp(-beta H^2),
beta = 1/(2 sigma^2), of the ribbon (ribbon periodic in y, Nx = 120).  The
kernel is accumulated over the whole ky Brillouin zone, which is the partial
trace over y; away from the two valleys the ribbon is gapped and contributes
nothing.  The trace norm itself is computed by wigner_tracenorm.py.

Right panel: half signature of

    L = H (x) tau_z + kappa Y (x) tau_y + kappa G_2 (x) tau_x,
    G_2 = (X/x0)^2 + i (Tx^dag - Tx) - c,

on an open 30 x 6 ribbon.  With T|x> = |x+1>, the symbol of G_2 is
g_2(x, k_x) = (x/x0)^2 - 2 sin(k_x) - c.  With c = 1 the region g_2 < 0 is a
closed pocket around x = 0 that contains the valley K at k_x = 2pi/3 and
excludes K' at k_x = -2pi/3; this is the cyan curve of the left panel.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

import _paths  # noqa: F401
from speclocalizer import (build_lattice, embed, half_signature, localizer_gap,
                           low_energy_density_matrix, pauli, shift_open,
                           shift_periodic, use_paper_style, wigner_transform_onesided)
from speclocalizer.style import DENSITY_CMAP, LABEL_FONTSIZE, SHELL_COLOR, TITLE_FONTSIZE


def mass_profile(x, length, mass_amplitude, wall_width, mass_offset):
    """Single smooth wall at x = 0, of width proportional to the system size."""
    del wall_width          # kept for signature compatibility
    return mass_amplitude * np.tanh(x / length) + mass_offset


def sublattice_matrices():
    _, sigma_x, sigma_y, sigma_z = pauli()
    sigma_plus = 0.5 * (sigma_x + 1j * sigma_y)
    return sigma_plus, sigma_plus.conj().T, sigma_z


def ribbon_hamiltonian(ky, nx, args, mass_offset):
    """Bloch Hamiltonian H(ky) of the ribbon, periodic in x."""
    sigma_plus, sigma_minus, sigma_z = sublattice_matrices()
    identity = np.eye(nx, dtype=complex)
    tx = shift_open(nx).toarray()
    x = np.arange(nx, dtype=float) - nx / 2.0
    mass = np.diag(mass_profile(x, nx, args.mass, args.wall_width, mass_offset).astype(complex))
    # T_y^dag has symbol e^{+i ky} with T|y> = |y+1>, the same convention as the
    # bulk model of the semimetal section.  With the opposite sign the ribbon's
    # Dirac points come out at kx = +ky instead of kx = -ky.
    hopping = identity + np.exp(1j * ky) * identity + tx.conj().T
    return (np.kron(mass, sigma_z)
            - args.hopping * (np.kron(hopping, sigma_plus)
                              + np.kron(hopping.conj().T, sigma_minus)))


def filtered_density_matrix(nx, args):
    """Kernel of exp(-H^2 / 2 sigma^2) in x, summed over the ky Brillouin zone.

    No state is selected by hand: the Gaussian filter does the selection, and
    the transverse momenta that are gapped simply carry no weight.  Also returns
    the two valley energies and the first excited level, so that the choice
    |E_0| << sigma << E_1 can be checked.
    """
    density = np.zeros((nx, nx), dtype=complex)
    ky_values = np.linspace(-np.pi, np.pi, args.ky_points, endpoint=False)
    for ky in ky_values:
        low_energy_density_matrix(ribbon_hamiltonian(ky, nx, args, 0.0), nx,
                                  args.sigma_energy, 2, accumulator=density)
    density /= np.trace(density).real

    valley_momenta = np.array([-2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0])
    valley_energies = np.empty(valley_momenta.size)
    for position, ky in enumerate(valley_momenta):
        energies = np.linalg.eigvalsh(ribbon_hamiltonian(ky, nx, args, 0.0))
        valley_energies[position] = energies[int(np.argmin(np.abs(energies)))]
    magnitudes = np.sort(np.abs(np.linalg.eigvalsh(
        ribbon_hamiltonian(valley_momenta[0], nx, args, 0.0))))
    first_excited = float(magnitudes[2])
    return valley_momenta, valley_energies, first_excited, density


def wigner_transform_ribbon(density):
    """W(x, k) of the open ribbon, on the doubled midpoint grid.

    One-sided convention, W(x,k) = sum_r rho(x+r, x) e^{-i k r}, with x on
    lattice sites and no wrap-around: the ribbon is open in x.
    """
    return wigner_transform_onesided(density, density.shape[0])


def localizer_operators(args, mass_offset):
    """Return H, G_1 = Y and G_2 on the periodic-x, open-y lattice."""
    sigma_plus, sigma_minus, sigma_z = sublattice_matrices()
    nx, ny = args.nx_localizer, args.ny_localizer
    lattice = build_lattice((nx, ny), boundary="open")
    from scipy import sparse

    x = np.arange(nx, dtype=float) - nx / 2.0
    profile = mass_profile(x, nx, args.mass, args.wall_width, mass_offset)
    mass_site = sparse.kron(sparse.diags(profile.astype(complex), format="csr"),
                            sparse.identity(ny, dtype=complex, format="csr"), format="csr")
    x0 = args.x0_fraction * nx
    quadratic_site = sparse.kron(
        sparse.diags(((x / x0) ** 2).astype(complex), format="csr"),
        sparse.identity(ny, dtype=complex, format="csr"), format="csr")

    tx, ty = lattice.shifts
    hopping = lattice.identity + tx.conj().T + ty.conj().T
    hamiltonian = (embed(mass_site, sigma_z)
                   - args.hopping * embed(hopping, sigma_plus)
                   - args.hopping * embed(hopping.conj().T, sigma_minus))

    g1 = embed(lattice.positions[1], np.eye(2, dtype=complex))
    g2_site = (quadratic_site - args.shell_offset * lattice.identity
               + 1j * (tx.conj().T - tx))   # symbol (x/x0)^2 - 2 sin k_x - c
    g2 = embed(g2_site, np.eye(2, dtype=complex))
    return hamiltonian.tocsr(), g1.tocsr(), g2.tocsr()


def valley_localizer(hamiltonian, g1, g2, kappa):
    _, tau_x, tau_y, tau_z = pauli()
    return (embed(hamiltonian, tau_z) + kappa * embed(g1, tau_y)
            + kappa * embed(g2, tau_x)).tocsr()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=120, help="ribbon size for the Wigner panel")
    parser.add_argument("--nx-localizer", type=int, default=30)
    parser.add_argument("--ny-localizer", type=int, default=6)
    parser.add_argument("--mass", type=float, default=3.0, help="m0")
    parser.add_argument("--sigma-energy", type=float, default=0.10,
                        help="sigma of the Gaussian filter exp(-H^2/2 sigma^2)")
    parser.add_argument("--ky-points", type=int, default=181,
                        help="ky samples used to accumulate the filtered kernel")
    parser.add_argument("--wall-width", type=float, default=1.0, help="w")
    parser.add_argument("--hopping", type=float, default=1.0, help="t")
    parser.add_argument("--shell-offset", type=float, default=1.0, help="c in G_2")
    parser.add_argument("--x0-fraction", type=float, default=0.18,
                        help="x0 = fraction * Lx in G_2")
    parser.add_argument("--kappa", type=float, default=0.4)
    parser.add_argument("--lambda-min", type=float, default=-1.5)
    parser.add_argument("--lambda-max", type=float, default=1.5)
    parser.add_argument("--lambda-points", type=int, default=61)
    parser.add_argument("--output", default="fig04_valley_graphene.png")
    args = parser.parse_args()

    use_paper_style()
    valley_momenta, valley_energies, first_excited, density = filtered_density_matrix(
        args.nx, args)
    x, k_values, wigner = wigner_transform_ribbon(density)

    lambdas = np.linspace(args.lambda_min, args.lambda_max, args.lambda_points)
    indices = np.empty(lambdas.size)
    gaps = np.empty(lambdas.size)
    for position, offset in enumerate(lambdas):
        hamiltonian, g1, g2 = localizer_operators(args, offset)
        localizer = valley_localizer(hamiltonian, g1, g2, args.kappa)
        index, _, _ = half_signature(localizer)
        indices[position] = index
        gaps[position] = localizer_gap(localizer)

    centre = int(np.argmin(np.abs(lambdas)))
    print(f"nx={args.nx}, localizer {args.nx_localizer}x{args.ny_localizer}, "
          f"m0={args.mass:g}, c={args.shell_offset:g}, kappa={args.kappa:g}, "
          f"sigma={args.sigma_energy:g}")
    for ky, energy in zip(valley_momenta, valley_energies):
        print(f"  valley zero mode: ky={ky:+.6f}, E={energy:.3e}")
    worst = float(np.max(np.abs(valley_energies)))
    print(f"  filter window: |E_0|/sigma = {worst / args.sigma_energy:.1e}, "
          f"sigma/E_1 = {args.sigma_energy / first_excited:.3f}  (E_1 = {first_excited:.4f})")
    ends = max(1, args.nx // 10)
    diagonal = np.real(np.diag(density))
    print(f"  filtered weight at the two ends: "
          f"{diagonal[:ends].sum() + diagonal[-ends:].sum():.2e}")
    print(f"  half signature at lambda=0 : {indices[centre]:.0f}")
    print(f"  localizer gap  at lambda=0 : {gaps[centre]:.4g}")

    np.savez(str(args.output).rsplit(".",1)[0] + ".npz", x=x, k=k_values, w=np.abs(wigner), lambdas=lambdas, indices=indices, gaps=gaps, x0=args.x0_fraction * args.nx, c=args.shell_offset)
    figure, (wigner_axis, index_axis) = plt.subplots(
        1, 2, figsize=(11.0, 5.0), gridspec_kw={"width_ratios": [1.0, 1.15]})
    magnitude = np.abs(wigner)
    image = wigner_axis.imshow(
        magnitude.T, origin="lower", aspect="auto",
        extent=[x[0], x[-1], k_values[0], np.pi],
        cmap=DENSITY_CMAP, vmin=0.0, vmax=float(np.quantile(magnitude, 0.999)),
        interpolation="bicubic")
    wigner_axis.set_box_aspect(1.35)
    bar = figure.colorbar(image, ax=wigner_axis)
    bar.set_label(r"$|W(x,k)|$", fontsize=LABEL_FONTSIZE)
    xx, kk = np.meshgrid(x, k_values, indexing="ij")
    # both the mass profile and the shell scale with the system size, so the
    # shell is drawn with x0 referred to the ribbon used for the Wigner panel.
    # g_2 is written in ky, the panel is drawn in the momentum k conjugate to x,
    # and on the wall modes kx = -ky, hence the substitution sin(ky) -> -sin(k).
    shell = ((xx / (args.x0_fraction * args.nx)) ** 2
             - 2.0 * np.sin(kk) - args.shell_offset)
    wigner_axis.contour(xx, kk, shell, levels=[0.0], colors=SHELL_COLOR, linewidths=1.8)
    wigner_axis.set_xlabel(r"$x$", fontsize=LABEL_FONTSIZE)
    wigner_axis.set_ylabel(r"$k$", fontsize=LABEL_FONTSIZE)
    wigner_axis.set_yticks([-np.pi, 0.0, np.pi])
    wigner_axis.set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
    wigner_axis.set_title(r"$|W(x,k)|$ of $e^{-\beta H^2}$ and shell",
                          fontsize=TITLE_FONTSIZE)

    index_axis.plot(lambdas, indices, color="tab:blue", marker="o", markersize=3.0)
    index_axis.axhline(0.0, color=SHELL_COLOR, linestyle="--", linewidth=1.2)
    index_axis.set_xlabel(r"$\lambda$", fontsize=LABEL_FONTSIZE)
    index_axis.set_ylabel(r"$\mathrm{Sig}(L)/2$", fontsize=LABEL_FONTSIZE)
    index_axis.set_title("Spectral-localizer index", fontsize=TITLE_FONTSIZE)
    index_axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
