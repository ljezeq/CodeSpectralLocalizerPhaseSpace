# Spectral localizers in phase space — code and data

Code and data for every figure of

> L. Jezequel and C. Mora, *Spectral localizers in phase space: ℤ and ℤ₂ invariants
> for higher-order, semimetal, and valley modes*, [arXiv:2611.26160](https://arxiv.org/abs/2611.26160) (2026).

## Contents

```
speclocalizer/   small library: lattice operators, recursive spectral localizer,
                 invariants (Sig/2, sgn det A, sgn Pf A), Gaussian filter, Wigner-Weyl transforms
figures/         one script per data set (table below), the model modules hoti_model.py and
                 semimetal_models.py, and replot.py, which draws the figures from data/
data/            the data shown in the figures (.npz)
paper_figures/   the figure files of the article
run_all.sh       recomputes every file of data/ with the parameters of the article, then redraws
```

## Redrawing the figures

```
pip install -r requirements.txt
python3 figures/replot.py
```

`replot.py` reads `data/` and writes the seven figure files of the article to
`paper_figures_redrawn/`. With matplotlib 3.11.2 they are identical, pixel for pixel,
to the files in `paper_figures/`, except Fig. 5, which was drawn with matplotlib 3.10.9:
it is identical with that version, and differs by one or two pixels in size with 3.11.

## Recomputing the data

`run_all.sh` lists the command and the parameters behind each data file. Running these
commands reproduces the files of `data/` to rounding error, with one exception noted below
for Fig. 2(c,d).

| Figure | File | Data | Script | Parameters |
| --- | --- | --- | --- | --- |
| 1 | `qwz_reduction.png` | `fig01.npz` | `fig01_qwz_reduction.py` | QWZ, L = 20, M = t = 1, κ = 0.3, ε = 0.42 |
| 2(a,b) | `LDOS3DHOTI.png` | `fig02a.npz` | `fig02a_hoti_ldos.py` | L = 16, (M/t, Δ, λ, κ) = (2, 1, 1, 0.3) |
| 2(c,d) | `higherHOTIlocalizer.png` | `fig02b_L20_clean.npz`, `fig02b_L20_W1.npz` | `fig02b_hoti_indices.py` | L = 20, κ = 0.1, W = 0 and 1 (seed 0) |
| 3(a–h) | `semimetal_full.png` | `fig03a_graphene.npz`, `fig03a_dirac.npz` | `fig03a_semimetal_ldos.py` | periodic 29 × 29, κ = 0.1, ε = 0.17 |
| 3(i,j) | `semimetal_full.png` | `fig03_indices.npz`, `fig03_multilayer.npz` | `fig03_indices.py` | 15 × 15, κ = 0.1, γ₁ = 0.12 t |
| 4 | `valley_hall_zero_mode_signature.png` | `fig04.npz` | `fig04_valley_graphene.py`, `wigner_tracenorm.py` | L₀ = L_x = 120 (a), 30 × 6 ribbon (b), m₀ = 3, κ = 0.4 |
| 5 | `kitaev_valley_trim_wigner_shell.png` | `fig05.npz` | `fig05_kitaev_valley.py`, `wigner_tracenorm.py` | L₀ = L = 120, m₀ = 3, Δ = 1, κ = 0.4 |
| 6 | `volkov_pankratov_scaling.png` | `fig06.npz` | `fig06_volkov_pankratov.py` | m₀ = 3, Δ = t = 1, L₀ = L from 10 to 200 |

`data/fig04_30x12.npz` holds the check quoted in the caption of Fig. 4 (the same
indices on a 30 × 12 ribbon); it is not drawn.

Every data set takes seconds to minutes on two cores. For Fig. 2(c,d),
`fig02b_hoti_indices.py` obtains Sig/2 from the inertia of a sparse LDLᵀ
factorization (MKL Pardiso, through `pypardiso`) of a localizer of size
8L³ = 64 000, about 20 s per value of M/t, so that each sweep over 41 values takes
about 15 minutes. At L = 20 the ℤ₂ index uses the identity sgn Pf A = (−1)^{Sig/2}
of the doubled model (option `--z2-parity`); the option `--z2` computes the Pfaffian
directly (with `pfapack`), which is feasible up to L ≈ 10 and was used to check the
identity for L ≤ 8. Where Pardiso has to perturb pivots (at M/t = 0 every diagonal entry of
the realified localizer vanishes), neither its inertia nor its solves can be trusted, and
the script redoes the factorization with a small shift, below the gap (the `shift` array of
its output). In the clean sweep this happens at M/t = 0 and 0.5; there the recomputed gaps
differ from those of `data/` (and of the published figure) by at most 4 × 10⁻⁵, and every
index is unchanged.

In the first arXiv version, Fig. 2(a,b) was computed without orthonormalizing the
eigenvectors of degenerate levels, which made the weights of the four hinges slightly
unequal. `fig02a_hoti_ldos.py` orthonormalizes them, and `paper_figures/` holds the
corrected figure.

## Conventions

* Translations act as T|x⟩ = |x+1⟩, so that T ↔ e^{−ik} and T† ↔ e^{ik}. The
  operator `lattice.sin(axis)` = (T − T†)/2i therefore has symbol −sin k.
* In `hoti_model.py` the sine term of the 3D HOTI is built with `lattice.sin`, that is
  Eq. (3DHOTI) with λ → −λ. This fixes the sign of Sig/2 in Fig. 2(c), −1 in the
  higher-order phase; the opposite sign of λ gives +1.
* Every low-energy density is the Gaussian filter exp(−H²/ε²); the scripts use
  σ = ε/√2, that is exp(−H²/2σ²).
* Phase-space densities use the one-sided transform W(x, k) = Σ_r ρ(x+r, x) e^{−ikr}
  (Appendix B of the article); Figs. 4 and 5 show its trace norm over the internal index.
* Site index order: for a lattice of shape (L₀, …, L_{n−1}) the site index is
  kron(A₀, …, A_{n−1}) (axis 0 slowest), and internal degrees of freedom come last.
