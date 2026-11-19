"""Fig. 6: the wall modes of both valley models are topological Volkov--Pankratov states.

Paper: L_0 = L (wall width = system size); data in data/fig06.npz, drawn by replot.py.

Both models of Sec. "Valley topology" have the same structure.  Linearised at
each of its two valleys, the wall is a Dirac Hamiltonian with a mass kink -- the
Jackiw--Rebbi problem -- and since the mass is smooth, m(x) ~ m'(0) x near x = 0
with m'(0) = m0 / L, the bound states form the Volkov--Pankratov ladder

    E_n = sqrt(2 n v m'(0)) = sqrt(2 n v m0 / L),    n = 0, 1, 2, ...

with v = Delta for the Kitaev chain and v = t for the graphene ribbon.  Each
valley therefore carries a topological n = 0 zero mode, and the two carry
opposite topology.  A coupling between the valleys turns their crossing into an
avoided crossing:

* the n = 0 pair is split only by inter-valley scattering.  For a smooth wall
  that amplitude is exponentially small, so the gap collapses exponentially;
* the n >= 1 pairs are already separated from zero by the Volkov--Pankratov
  energy itself, so their gap closes only as L^{-1/2}.

The two panels measure exactly these two laws.

Where each law can be measured.  In the Kitaev chain the two valleys, k = 0 and
k = pi, live in the momentum conjugate to the open direction, so the open ends
do couple them and the exponential law is directly visible; the measured
splitting is

    |E_0| ~ psi(L/2)^2 = exp(-2 ln cosh(1/2) m0 L / Delta),

with psi(x) ~ [cosh(x/L)]^{-m0 L/Delta} the continuum zero mode, and no fit.
In the graphene ribbon the two valleys are separated in ky, which stays a good
quantum number, so the coupling vanishes identically and the kink channel is
exactly gapless -- the exponentially small gap is what it acquires as soon as
inter-valley scattering is switched on.  The n = 1 law, on the other hand, is
directly measurable in both models, and the left panel shows both.
"""


import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

import _paths  # noqa: F401
from speclocalizer import build_lattice, embed, pauli, use_paper_style

NOISE_FLOOR = 1.0e-14
DECAY_RATE = 2.0 * np.log(np.cosh(0.5))          # 0.24022, so 1/DECAY_RATE = 4.163


def hamiltonian(length, mass_amplitude, pairing):
    _, _, tau_y, tau_z = pauli()
    lattice = build_lattice((length,), boundary="open")
    x = lattice.coordinates[0]
    mass = sparse.diags((mass_amplitude * np.tanh(x / length)).astype(complex),
                        format="csr")
    return (embed(mass, tau_z) + pairing * embed(lattice.sin(0), tau_y)).toarray()


def two_lowest(length, mass_amplitude, pairing, tolerance=0.01):
    """Return (|E_0|, E_1): the topological level and the first massive rung.

    Levels closer than `tolerance` in relative value are merged, since the two
    valleys of one rung are split only by the exponentially small coupling.
    """
    magnitudes = np.sort(np.abs(np.linalg.eigvalsh(
        hamiltonian(length, mass_amplitude, pairing))))
    zero = magnitudes[0]
    for value in magnitudes:
        if value > max(10.0 * zero, 1.0e-6) and value > tolerance:
            return zero, value
    return zero, np.nan


def scan(lengths, mass_amplitude, pairing):
    zero = np.empty(lengths.size)
    first = np.empty(lengths.size)
    for index, length in enumerate(lengths):
        zero[index], first[index] = two_lowest(int(length), mass_amplitude, pairing)
    return zero, first


def ribbon_first_rung(nx, mass_amplitude, hopping=1.0):
    """First Volkov-Pankratov subband of the graphene ribbon, at the valley."""
    import types

    from fig04_valley_graphene import ribbon_hamiltonian
    args = types.SimpleNamespace(mass=mass_amplitude, wall_width=1.0,
                                 hopping=hopping)
    energies = np.sort(np.abs(np.linalg.eigvalsh(
        ribbon_hamiltonian(-2.0 * np.pi / 3.0, nx, args, 0.0))))
    return float(energies[2])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mass", type=float, default=3.0, help="m0")
    parser.add_argument("--pairing", type=float, default=1.0,
                        help="Delta for the chain, and t for the ribbon")
    parser.add_argument("--l-min", type=int, default=10)
    parser.add_argument("--l-max", type=int, default=200)
    parser.add_argument("--lx-max", type=int, default=120,
                        help="largest ribbon used for the graphene curve")
    parser.add_argument("--output", default="fig06_volkov_pankratov.png")
    args = parser.parse_args()

    use_paper_style()
    lengths = np.arange(args.l_min, args.l_max + 1, 2, dtype=float)
    zero, first = scan(lengths, args.mass, args.pairing)

    ribbon_lengths = np.arange(args.l_min, args.lx_max + 1, 2, dtype=float)
    ribbon_first = np.array([ribbon_first_rung(int(n), args.mass, args.pairing)
                             for n in ribbon_lengths])

    np.savez(str(args.output).rsplit(".",1)[0] + ".npz", lengths=lengths, zero=zero, first=first, ribbon_lengths=ribbon_lengths, ribbon_first=ribbon_first, mass=args.mass, pairing=args.pairing)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

    # --- left: the Volkov-Pankratov ladder, a pure power law -----------------
    for label, lens, values, colour in (
            (r"Kitaev chain", lengths, first, "tab:blue"),
            (r"graphene ribbon", ribbon_lengths, ribbon_first, "tab:red")):
        prediction = np.sqrt(2.0 * args.pairing * args.mass / lens)
        power = np.polyfit(np.log(lens), np.log(values), 1)[0]
        axes[0].loglog(lens, values, ".", color=colour, markersize=4.0,
                       label=rf"$E_1$, {label}   (fit $L^{{{power:.2f}}}$)")
        axes[0].loglog(lens, prediction, "-", color=colour, linewidth=1.0, alpha=0.7)
        print(f"{label}: E_1 exponent {power:+.3f}, E_1(L={int(lens[-1])}) = "
              f"{values[-1]:.5f} vs sqrt(2 v m0/L) = {prediction[-1]:.5f}")
    axes[0].set_xlabel(r"$L$")
    axes[0].set_ylabel(r"$E_1$")
    axes[0].set_title(r"$n=1$: massive, gap $=\sqrt{2 v m_0/L}$")
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(fontsize=9, loc="lower left")

    # --- right: the topological level, a pure exponential --------------------
    good = zero > NOISE_FLOOR
    slope = np.polyfit(lengths[good], np.log(zero[good]), 1)[0]
    correlation = np.corrcoef(lengths[good], np.log(zero[good]))[0, 1]
    rate = DECAY_RATE * args.mass / args.pairing
    axes[1].semilogy(lengths, np.maximum(zero, 0.3 * NOISE_FLOOR), ".",
                     color="tab:blue", markersize=4.0,
                     label=rf"$|E_0|$, Kitaev chain   (fit $e^{{-L/{-1.0 / slope:.2f}}}$)")
    reference = zero[good][0] * np.exp(-rate * (lengths - lengths[good][0]))
    axes[1].semilogy(lengths, reference, "-", color="black", linewidth=1.0, alpha=0.7,
                     label=rf"$e^{{-2\ln\cosh(1/2)\,m_0L/\Delta}}$ (no fit)")
    axes[1].axhspan(1.0e-16, NOISE_FLOOR, color="grey", alpha=0.25, linewidth=0)
    axes[1].text(lengths[-1], 2.0e-15, "double precision", ha="right", va="center",
                 fontsize=8, color="0.35")
    axes[1].set_xlabel(r"$L$")
    axes[1].set_ylabel(r"$|E_0|$")
    axes[1].set_ylim(1.0e-16, 1.0)
    axes[1].set_title(r"$n=0$: topological, gap exponentially small")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(fontsize=9, loc="lower left")
    print(f"Kitaev: |E_0| ~ exp(-L/{-1.0 / slope:.3f})  "
          f"[predicted {1.0 / rate:.3f} = 4.163 Delta/m0], "
          f"semi-log correlation {correlation:+.5f}")

    figure.tight_layout()
    figure.savefig(args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
