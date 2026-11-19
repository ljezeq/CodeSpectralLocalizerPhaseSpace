"""Fig. 5: valley-resolved localizer for a Kitaev chain with a smooth mass.

    H = m(X) tau_z + Delta (T - T^dag)/2i tau_y,     (symbol m tau_z - Delta sin k tau_y)
    m(x) = m0 tanh(x / L_0) + lambda,

as in Eq. (kitaev) of the paper, on an open chain of L sites with x centred and
L_0 = L.  The mass changes sign at a single smooth wall at x = 0, and the zero
modes have width sqrt(Delta L_0 / m0).

The pairing vanishes at the two time-reversal-invariant momenta k = 0 and
k = pi, so the wall traps two low-energy modes separated in wavenumber, one per
valley.  The mixed operator

    G = (X/x0)^2 - (T + T^dag),   symbol  g(x,k) = (x/x0)^2 - 2 cos k,

is negative at (x,k) = (0,0), positive at (0,pi) and at large |x|, so its zero
set is a closed curve around the selected mode only.  The localizer

    L = H (x) eta_z + kappa G (x) eta_x

has chiral symmetry eta_y.  H has particle-hole symmetry P = tau_x K with
P^2 = +1; in the Majorana basis where P is complex conjugation, H is purely
imaginary while G stays real, so that A = i H + kappa G is real and sgn det A is
the zero-dimensional invariant.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

import _paths  # noqa: F401
from speclocalizer import (build_lattice, chiral_block, determinant_sign, embed,
                           localizer_gap, low_energy_density_matrix, pauli,
                           use_paper_style, wigner_transform_onesided)
from speclocalizer.style import DENSITY_CMAP, SHELL_COLOR


def model_operators(length, mass_amplitude, mass_offset, pairing, x0_fraction):
    """Return (H, G, x) for the single-wall Kitaev chain with open boundaries."""
    _, _, tau_y, tau_z = pauli()
    lattice = build_lattice((length,), boundary="open")
    x = lattice.coordinates[0]

    profile = mass_amplitude * np.tanh(x / length) + mass_offset
    mass = sparse.diags(profile.astype(complex), format="csr")
    hamiltonian = embed(mass, tau_z) + pairing * embed(lattice.sin(0), tau_y)

    x0 = x0_fraction * length
    quadratic = sparse.diags(((x / x0) ** 2).astype(complex), format="csr")
    selector = quadratic - 2.0 * lattice.cos(0)
    return (hamiltonian.tocsr(),
            embed(selector, np.eye(2, dtype=complex)).tocsr(), x)


def majorana_rotation(n_sites):
    """W = e^{-i pi/4}(1 + i tau_x)/sqrt2, for which W^dag H W is purely imaginary."""
    _, tau_x, _, _ = pauli()
    w2 = np.exp(-1j * np.pi / 4) * (np.eye(2, dtype=complex) + 1j * tau_x) / np.sqrt(2.0)
    return np.kron(np.eye(n_sites, dtype=complex), w2)


def invariants(hamiltonian, selector, kappa, rotation, reference_phase):
    _, eta_x, eta_y, eta_z = pauli()
    n = hamiltonian.shape[0]
    localizer = embed(hamiltonian, eta_z) + kappa * embed(selector, eta_x)
    gap = localizer_gap(localizer)

    h_majorana = rotation.conj().T @ hamiltonian.toarray() @ rotation
    g_majorana = rotation.conj().T @ selector.toarray() @ rotation
    a_matrix = 1j * h_majorana + kappa * g_majorana
    imaginary = np.linalg.norm(a_matrix.imag) / max(1.0, np.linalg.norm(a_matrix))
    sign_paper = int(np.sign(np.linalg.slogdet(a_matrix.real)[0]))

    block = chiral_block(localizer, embed(np.eye(n, dtype=complex), eta_y))
    sign_block, phase, _ = determinant_sign(block, reference_phase=reference_phase)
    return gap, sign_paper, sign_block, phase, imaginary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=120)
    parser.add_argument("--mass", type=float, default=3.0, help="m0")
    parser.add_argument("--pairing", type=float, default=1.0, help="Delta")
    parser.add_argument("--x0-fraction", type=float, default=0.25,
                        help="x0 = fraction * L in the confining operator")
    parser.add_argument("--kappa", type=float, default=0.4)
    parser.add_argument("--sigma-energy", type=float, default=0.06,
                        help="sigma of the Gaussian filter exp(-H^2/2 sigma^2)")
    parser.add_argument("--lambda-min", type=float, default=-1.5)
    parser.add_argument("--lambda-max", type=float, default=1.5)
    parser.add_argument("--lambda-num", type=int, default=61)
    parser.add_argument("--output", default="fig05_kitaev_valley.png")
    args = parser.parse_args()

    use_paper_style()
    rotation = majorana_rotation(args.length)
    hamiltonian, selector, x = model_operators(
        args.length, args.mass, 0.0, args.pairing, args.x0_fraction)

    # A uniform mass has no wall and fixes the overall phase of the determinant.
    reference = model_operators(args.length, 0.0, args.mass, args.pairing,
                                args.x0_fraction)
    _, _, _, reference_phase, _ = invariants(reference[0], reference[1], args.kappa,
                                             rotation, None)

    gap, sign_paper, sign_block, _, imaginary = invariants(
        hamiltonian, selector, args.kappa, rotation, reference_phase)
    print(f"L={args.length}, m0={args.mass:g}, Delta={args.pairing:g}, "
          f"x0={args.x0_fraction:g}L, kappa={args.kappa:g}")
    print(f"  localizer gap        : {gap:.4g}")
    print(f"  |Im A| / |A|         : {imaginary:.2e}")
    print(f"  sgn det(iH + kappa G): {sign_paper:+d}   (chiral block {sign_block:+d})")

    lambdas = np.linspace(args.lambda_min, args.lambda_max, args.lambda_num)
    signs = np.empty(lambdas.size)
    for index, offset in enumerate(lambdas):
        h_l, g_l, _ = model_operators(args.length, args.mass, offset,
                                      args.pairing, args.x0_fraction)
        gap_l, sign_l, _, _, _ = invariants(h_l, g_l, args.kappa, rotation,
                                            reference_phase)
        signs[index] = sign_l if gap_l > 1.0e-7 else np.nan

    density = low_energy_density_matrix(hamiltonian.toarray(), args.length,
                                        args.sigma_energy, 2)
    edge = max(1, args.length // 10)
    leak = float(np.real(np.trace(density))) and float(
        np.sum(np.real(np.diag(density))[:edge]) + np.sum(np.real(np.diag(density))[-edge:]))
    print(f"  weight of the filtered kernel at the two ends: {leak:.2e}")
    magnitudes = np.sort(np.abs(np.linalg.eigvalsh(hamiltonian.toarray())))
    print(f"  filter window: |E_0| = {magnitudes[0]:.2e}, sigma = "
          f"{args.sigma_energy:g}, E_1 = {magnitudes[2]:.4f}  "
          f"(sigma/E_1 = {args.sigma_energy / magnitudes[2]:.3f})")
    x_values, k_values, wigner = wigner_transform_onesided(density, args.length)

    np.savez(str(args.output).rsplit(".",1)[0] + ".npz", x=x_values, k=k_values, w=np.abs(wigner), lambdas=lambdas, signs=signs, x0=args.x0_fraction * args.length)
    figure, (axis, index_axis) = plt.subplots(
        1, 2, figsize=(11.0, 5.0), gridspec_kw={"width_ratios": [1.0, 1.15]})
    magnitude = np.abs(wigner)
    image = axis.imshow(magnitude.T, origin="lower", aspect="auto",
                        extent=[x_values[0], x_values[-1], k_values[0], np.pi],
                        cmap=DENSITY_CMAP, vmin=0.0,
                        vmax=float(np.quantile(magnitude, 0.999)),
                        interpolation="bicubic")
    axis.set_box_aspect(1.35)
    figure.colorbar(image, ax=axis).set_label(r"$|W(x,k)|$")
    xx, kk = np.meshgrid(x_values, k_values, indexing="ij")
    symbol = (xx / (args.x0_fraction * args.length)) ** 2 - 2.0 * np.cos(kk)
    axis.contour(xx, kk, symbol, levels=[0.0], colors=SHELL_COLOR, linewidths=1.8)
    axis.set_xlabel(r"$x$")
    axis.set_ylabel(r"$k$")
    axis.set_yticks([-np.pi, 0.0, np.pi])
    axis.set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
    axis.set_title(r"$|W(x,k)|$ of $e^{-\beta H^2}$ and shell")

    index_axis.plot(lambdas, signs, "o-", color="tab:blue", markersize=3.0)
    index_axis.axhline(0.0, color=SHELL_COLOR, linewidth=1.2, linestyle="--")
    index_axis.set_xlabel(r"$\lambda$")
    index_axis.set_ylabel(r"$\mathrm{sgn}\,\det A$")
    index_axis.set_yticks([-1, 1])
    index_axis.set_ylim(-1.3, 1.3)
    index_axis.set_title("Localizer index")
    figure.tight_layout()
    figure.savefig(args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
