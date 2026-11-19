"""Fig. 2(c,d): Z and Z2 indices of the 3D HOTI localizer versus M/t, and its gap.

Sig/2 is obtained from the inertia of a sparse LDL^T factorisation (MKL Pardiso,
through pypardiso) of the realified localizer, and the gap by shift-invert
Lanczos with the same factorisation.  The Z2 index sgn Pf A is computed by a
dense Pfaffian (pfapack) with --z2, feasible up to L ~ 10, or from the identity
sgn Pf A = (-1)^{Sig/2} of the doubled model with --z2-parity, which is what is
used at L = 20 (we checked the identity against the dense Pfaffian for L <= 8).

    python3 figures/fig02b_hoti_indices.py --length 20 --kappa 0.1 --z2-parity --output data/fig02b_L20_clean.npz
    python3 figures/fig02b_hoti_indices.py --length 20 --kappa 0.1 --z2-parity --disorder 1 --output data/fig02b_L20_W1.npz
"""
import argparse, os, time, ctypes
_mkl = "/usr/local/lib/libmkl_rt.so.3"          # only needed where pypardiso cannot find MKL by itself
if os.path.exists(_mkl):
    os.environ.setdefault("PYPARDISO_MKL_RT", _mkl)
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
import pypardiso
import _paths  # noqa: F401
from hoti_model import doubled_hamiltonian, hoti_confining, hoti_hamiltonian, majorana_rotation, onsite_disorder
from speclocalizer import build_lattice, embed, spectral_localizer


def realify(h):
    h = h.tocsr()
    return sparse.bmat([[h.real, -h.imag], [h.imag, h.real]], format="csr")


class LDL:
    def __init__(self, s):
        t = sparse.triu(s, k=1, format="coo"); n = s.shape[0]; i = np.arange(n)
        self.u = sparse.coo_matrix((np.r_[t.data, s.diagonal()], (np.r_[t.row, i], np.r_[t.col, i])), shape=s.shape).tocsr()
        self.u.sort_indices()
        self.p = pypardiso.PyPardisoSolver(mtype=-2)
        for k, v in {1: 1, 2: 2, 10: 8, 11: 1, 13: 1, 21: 1}.items():
            self.p.set_iparm(k, v)
        self.p.factorize(self.u)
        self.n = s.shape[0]

    def inertia(self):
        ip = self.p.iparm
        self.perturbed = int(ip[13])
        return int(ip[21]), int(ip[22])

    def solve(self, b):
        return self.p.solve(self.u, b)


def sig_half_and_gap(lz, want_gap=True):
    r = realify(lz)
    f = LDL(r)
    npos, nneg = f.inertia()
    z = (npos - nneg) // 4  # realification doubles every eigenvalue
    g = np.nan
    if want_gap:
        op = spla.LinearOperator(r.shape, matvec=f.solve, dtype=float)
        mu = spla.eigsh(op, k=4, which="LM", return_eigenvectors=False, tol=1e-6)
        g = float(1 / np.max(np.abs(mu)))
    sig_half_and_gap.perturbed = f.perturbed
    f.p.free_memory(everything=True)
    del f
    return z, g


def pf_sign(a):
    import pfapack.ctypes as pfc
    a = a.toarray(); at = a.T; n = a.shape[0]
    res = (ctypes.c_double * 2)(0.0, 0.0)
    if pfc.skpf10_d(n, at, res, b"U", b"P") != 0:
        raise RuntimeError
    return int(np.sign(res[0])) * (-1) ** (n // 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--length", type=int, default=8)
    p.add_argument("--kappa", type=float, default=0.15)
    p.add_argument("--ms", type=float, nargs="+", default=None)
    p.add_argument("--m-min", type=float, default=0.0)
    p.add_argument("--m-max", type=float, default=4.0)
    p.add_argument("--m-num", type=int, default=41)
    p.add_argument("--z2", action="store_true")
    p.add_argument("--z2-parity", action="store_true", help="Z2 = (-1)^Z, exact for the doubled model")
    p.add_argument("--disorder", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-gap", action="store_true")
    p.add_argument("--output", default="fig02b.npz")
    a = p.parse_args()
    lat = build_lattice((a.length,) * 3, boundary="open")
    g1, g2, geo = hoti_confining(lat)
    dis = onsite_disorder(lat, a.disorder, a.seed) if a.disorder > 0 else None
    rot = majorana_rotation(g1.shape[0]) if a.z2 else None
    ms = np.array(a.ms) if a.ms else np.linspace(a.m_min, a.m_max, a.m_num)
    out = {k: [] for k in ("m", "z", "zgap", "z2", "z2gap", "perturbed")}
    print(f"L={a.length} kappa={a.kappa} R={geo['radius']} n={8*a.length**3}", flush=True)
    for m in ms:
        t0 = time.time()
        h = hoti_hamiltonian(lat, m, 1.0, 1.0, 1.0)
        if dis is not None:
            h = (h + dis).tocsr()
        z, zg = sig_half_and_gap(spectral_localizer(h, [g1, g2], a.kappa), not a.no_gap)
        z2, z2g = 0, np.nan
        if a.z2:
            e2 = np.eye(2, dtype=complex)
            ld = spectral_localizer(doubled_hamiltonian(h), [embed(g1, e2), embed(g2, e2)], a.kappa)
            rd = (rot.conj().T @ ld @ rot).tocsr()
            am = rd.imag.tocsr(); am = (0.5 * (am - am.T)).tocsr()
            z2 = pf_sign(am)
        elif a.z2_parity:
            z2 = (-1) ** abs(z)
        for k, v in zip(out, (m, z, zg, z2, z2g, sig_half_and_gap.perturbed)):
            out[k].append(v)
        print(f"M={m:.3f} Z={z:+d} gap={zg:.4g} Z2={z2:+d} pert={sig_half_and_gap.perturbed} ({time.time()-t0:.1f}s)", flush=True)
        np.savez(a.output, **{k: np.array(v) for k, v in out.items()}, length=a.length, kappa=a.kappa, disorder=a.disorder)


if __name__ == "__main__":
    main()
