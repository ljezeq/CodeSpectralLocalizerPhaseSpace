"""Recompute the Wigner panels of Figs. 4 and 5 with the trace norm Tr sqrt(W W^dag).

W_{alpha beta}(x,k) = sum_r rho_{alpha beta}(x+r, x) e^{-ikr} keeps the internal
(sublattice / Nambu) indices; the trace norm is plotted.  Updates the 'w' array of
data/fig04.npz and data/fig05.npz (the old |Tr_alpha W| is kept as 'w_trace').
Run it after fig04_valley_graphene.py and fig05_kitaev_valley.py.
"""
import types
import numpy as np
import _paths  # noqa: F401
from speclocalizer import low_energy_density_matrix_full, wigner_trace_norm, wigner_transform_onesided
from fig04_valley_graphene import ribbon_hamiltonian
from fig05_kitaev_valley import model_operators

from pathlib import Path
CACHE = str(Path(__file__).resolve().parent.parent / "data") + "/"

def fig05(length=120, mass=3.0, pairing=1.0, sigma=0.06, x0f=0.25):
    h, _, _ = model_operators(length, mass, 0.0, pairing, x0f)
    rho = low_energy_density_matrix_full(h.toarray(), length, sigma, 2)
    return wigner_trace_norm(rho, length, 2)

def fig04(nx=120, mass=3.0, sigma=0.10, ky_points=181):
    args = types.SimpleNamespace(mass=mass, wall_width=1.0, hopping=1.0)
    acc = np.zeros((2 * nx, 2 * nx), dtype=complex)
    for ky in np.linspace(-np.pi, np.pi, ky_points, endpoint=False):
        low_energy_density_matrix_full(ribbon_hamiltonian(ky, nx, args, 0.0), nx, sigma, 2, accumulator=acc)
    acc /= np.trace(acc).real
    return wigner_trace_norm(acc, nx, 2)

for name, fn in (("fig05", fig05), ("fig04", fig04)):
    c = dict(np.load(CACHE + name + ".npz"))
    x, k, w = fn()
    old = c.get("w_trace", c["w"])
    print(name, "shape", w.shape, old.shape, " max TrNorm %.3f  max|TrW| %.3f" % (w.max(), old.max()),
          " rel. diff of normalised maps %.3f" % (np.abs(w / w.max() - old / old.max()).max()))
    c["w_trace"] = old; c["w"] = w
    np.savez(CACHE + name + ".npz", **c)
