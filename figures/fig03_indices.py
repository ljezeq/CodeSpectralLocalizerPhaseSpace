"""Fig. 3(i,j): localizer indices and gaps on a 15 x 15 lattice, kappa = 0.1, lambda in [0, 2].

Graphene (shell offset 2) and the n-layer rhombohedral stack under H - lambda sigma_x:
the two cones approach each other along k_x = -k_y and annihilate at M for
lambda = t, on the shell.  Dirac model (offset 3): half-signature under
H - lambda sigma_x, where the cone at the origin leaves the shell at sin(pi/3)
and annihilates at (pi/2, 0) for lambda = 1; determinant sign under H + lambda 1_2.

    python3 figures/fig03_indices.py --output data/fig03_indices.npz
    python3 figures/fig03_indices.py --part multi --gamma1 0.12 --lambda-num 81 --output data/fig03_multilayer.npz
"""
import argparse
import numpy as np
import _paths  # noqa: F401
from semimetal_models import (graphene_operators, multilayer, semimetal_operators,
                              time_reversal_gammas, time_reversal_localizer)
from dense_inertia import inertia
from speclocalizer import (build_lattice, chiral_block, chirality_element, embed,
                           localizer_gap, pauli, spectral_localizer)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lx", type=int, default=15)
    p.add_argument("--kappa", type=float, default=0.1)
    p.add_argument("--graphene-offset", type=float, default=2.0)
    p.add_argument("--dirac-offset", type=float, default=3.0)
    p.add_argument("--lambda-num", type=int, default=41)
    p.add_argument("--part", choices=("mono", "multi"), default="mono")
    p.add_argument("--gamma1", type=float, default=0.12)
    p.add_argument("--layers", type=int, nargs="+", default=[2, 3])
    p.add_argument("--output", default="fig03_indices.npz")
    a = p.parse_args()
    lat = build_lattice((a.lx, a.lx), boundary="open")
    s0, sx, _, sz = pauli()
    lams = np.linspace(0, 2, a.lambda_num)
    if a.part == "multi":
        out = {"lambda": lams}
        for n in a.layers:
            idx, gap = [], []
            for lam in lams:
                H, g1, g2, g3, C = multilayer(lat, n, a.gamma1, -lam, a.graphene_offset)
                L = spectral_localizer(H, [g1, g2, g3], a.kappa, chiral=C)
                idx.append(inertia(L) // 2); gap.append(localizer_gap(L))
                print(n, f"{lam:.3f} {idx[-1]:+d} {gap[-1]:.3g}", flush=True)
            out[f"index_{n}"] = np.array(idx); out[f"gap_{n}"] = np.array(gap)
        np.savez(a.output, gamma1=a.gamma1, kappa=a.kappa, offset=a.graphene_offset, **out)
        return
    gr, gg1, gg2, gg3 = graphene_operators(lat, a.graphene_offset)
    dh, g1, g2, g3 = semimetal_operators(lat, a.dirac_offset)
    C = embed(lat.identity, sz); n = lat.n_sites
    lop = np.kron(np.eye(2 * n), chirality_element(time_reversal_gammas()))
    r = {k: [] for k in ("chiral_index", "chiral_gap", "dirac_chiral_index", "dirac_chiral_gap",
                         "det_phase", "tr_gap")}
    for lam in lams:
        L = spectral_localizer(gr - lam * embed(lat.identity, sx), [gg1, gg2, gg3], a.kappa, chiral=C)
        r["chiral_index"].append(inertia(L) // 2); r["chiral_gap"].append(localizer_gap(L))
        L = spectral_localizer(dh - lam * embed(lat.identity, sx), [g1, g2, g3], a.kappa, chiral=C)
        r["dirac_chiral_index"].append(inertia(L) // 2); r["dirac_chiral_gap"].append(localizer_gap(L))
        L = time_reversal_localizer(dh + lam * embed(lat.identity, s0), [g1, g2, g3], a.kappa)
        ph, _ = np.linalg.slogdet(chiral_block(L, lop))
        r["det_phase"].append(ph); r["tr_gap"].append(localizer_gap(L))
        print(f"{lam:+.3f} {r['chiral_index'][-1]:+d} {r['dirac_chiral_index'][-1]:+d} "
              f"gaps {r['chiral_gap'][-1]:.3g} {r['dirac_chiral_gap'][-1]:.3g} {r['tr_gap'][-1]:.3g}", flush=True)
    ref = r["det_phase"][-1]  # +1 at lambda=2, where the T-breaking deformation has gapped everything
    r["det_sign"] = [int(np.sign(np.real(ph / ref))) for ph in r["det_phase"]]
    np.savez(a.output, **{"lambda": lams}, kappa=a.kappa, graphene_offset=a.graphene_offset,
             dirac_offset=a.dirac_offset, **{k: np.array(v) for k, v in r.items()})


if __name__ == "__main__":
    main()
