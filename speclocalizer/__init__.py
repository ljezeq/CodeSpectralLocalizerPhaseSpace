"""Spectral localizers in phase space.

Shared library for the figures of

    L. Jezequel and C. Mora,
    "Spectral localizers in phase space: Z and Z2 invariants for
     higher-order, semimetal, and valley modes".

Every figure script in ``figures/`` imports from here; nothing is duplicated
between scripts.
"""

from .lattice import (LatticeOperators, build_lattice, embed, pauli,
                      position_values, shift_open, shift_periodic)
from .localizer import (chirality_element, clifford_generators, localizer_gap,
                        reduction_step, smallest_eigenvalues, spectral_localizer)
from .invariants import (chiral_block, determinant_sign, half_signature,
                         majorana_form, pfaffian_sign_log, signature_dense,
                         sparse_ldl_inertia)
from .wigner import (low_energy_density_matrix_full, wigner_trace_norm, beta_from_sigma, gaussian_weights, low_energy_density_matrix,
                     sigma_from_beta, wigner_transform, wigner_transform_open,
                     wigner_transform_onesided)
from .style import use_paper_style

__all__ = [
    "LatticeOperators", "build_lattice", "embed", "pauli", "position_values",
    "shift_open", "shift_periodic",
    "chirality_element", "clifford_generators", "localizer_gap",
    "reduction_step", "smallest_eigenvalues", "spectral_localizer",
    "chiral_block", "determinant_sign", "half_signature", "majorana_form",
    "pfaffian_sign_log", "signature_dense", "sparse_ldl_inertia",
    "low_energy_density_matrix", "wigner_transform", "wigner_transform_open",
    "wigner_transform_onesided", "low_energy_density_matrix_full", "wigner_trace_norm",
    "gaussian_weights",
    "beta_from_sigma", "sigma_from_beta",
    "use_paper_style",
]
