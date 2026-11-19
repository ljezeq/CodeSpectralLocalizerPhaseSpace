#!/bin/bash
# Recompute every data file of data/ with the parameters of the paper, then redraw the figures.
# Run from the repository root.  Times are for two cores; Fig. 2(c,d) needs pypardiso.
set -e
export MPLBACKEND=Agg
mkdir -p data/scratch

# Fig. 1: QWZ model, L = 20, M = t = 1, kappa = 0.3, epsilon = 0.42 (sigma = 0.3)   [seconds]
python3 figures/fig01_qwz_reduction.py --output data/scratch/fig01.png
mv data/scratch/fig01.npz data/fig01.npz

# Fig. 2(a,b): 3D HOTI, L = 16, (M/t, Delta, lambda, kappa) = (2, 1, 1, 0.3), 80 lowest modes   [minutes]
python3 figures/fig02a_hoti_ldos.py --output data/scratch/fig02a.png
mv data/scratch/fig02a.npz data/fig02a.npz

# Fig. 2(c,d): L = 20, kappa = 0.1, clean and W = 1 (seed 0)   [about 15 minutes each, MKL Pardiso]
python3 figures/fig02b_hoti_indices.py --length 20 --kappa 0.1 --z2-parity --output data/fig02b_L20_clean.npz
python3 figures/fig02b_hoti_indices.py --length 20 --kappa 0.1 --z2-parity --disorder 1 --output data/fig02b_L20_W1.npz

# Fig. 3(a-h): periodic 29 x 29 lattice, kappa = 0.1, epsilon = 0.17 (sigma = 0.12)   [minutes]
python3 figures/fig03a_semimetal_ldos.py --lx 29 --ly 29 --kappa 0.1 --model graphene --output data/scratch/fig03a_graphene.png
python3 figures/fig03a_semimetal_ldos.py --lx 29 --ly 29 --kappa 0.1 --model dirac --output data/scratch/fig03a_dirac.png
mv data/scratch/fig03a_graphene.npz data/fig03a_graphene.npz
mv data/scratch/fig03a_dirac.npz data/fig03a_dirac.npz

# Fig. 3(i,j): 15 x 15, kappa = 0.1; graphene and Dirac model, then n = 2, 3 layers with gamma_1 = 0.12 t   [minutes]
python3 figures/fig03_indices.py --output data/fig03_indices.npz
python3 figures/fig03_indices.py --part multi --gamma1 0.12 --lambda-num 81 --output data/fig03_multilayer.npz

# Fig. 4 and Fig. 5: valley-Hall ribbon and Kitaev chain; the trace norm of W is added afterwards   [minutes]
python3 figures/fig04_valley_graphene.py --output data/scratch/fig04.png
python3 figures/fig05_kitaev_valley.py --output data/scratch/fig05.png
mv data/scratch/fig04.npz data/fig04.npz
mv data/scratch/fig05.npz data/fig05.npz
python3 figures/wigner_tracenorm.py
# check quoted in the caption of Fig. 4: the same indices on a 30 x 12 ribbon (not drawn)
python3 figures/fig04_valley_graphene.py --ny-localizer 12 --output data/scratch/fig04_30x12.png
mv data/scratch/fig04_30x12.npz data/fig04_30x12.npz

# Fig. 6: Volkov-Pankratov levels, m0 = 3, Delta = t = 1, L_0 = L from 10 to 200 (ribbon up to 120)   [minutes]
python3 figures/fig06_volkov_pankratov.py --output data/scratch/fig06.png
mv data/scratch/fig06.npz data/fig06.npz

rm -rf data/scratch
python3 figures/replot.py
