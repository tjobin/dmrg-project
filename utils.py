from tenpy.models import Model
from tenpy.algorithms.exact_diag import ExactDiag
import numpy as np
from tqdm import tqdm
from typing import Any

# exact energies for quick reference (Lx, Ly) -> E_exact
EXACT_ENERGIES_BOSEHUBBARD = {
    (3, 2): -12.7907317932865769,
    (4, 2): -18.0367928615768918,
    }
EXACT_ENERGIES_J1J2_torus = {
    (3, 3): -3.484501,
    (3, 4): -6.2043858259,
    (4, 4): -8.4579233514,
    (6, 6): -18.13714754
    }
EXACT_ENERGIES_J1J2_cylinder = {
    (3, 3): -3.7649701644,
    (3, 4): -5.4917879192,
    (4, 4): -8.2612325630
}

def get_exact_psi_and_E(
        model: Model,
        charge_sector: list[int] | int | None = None,
        max_size: float = 1e12
        ) -> tuple[Any, float] :
    """
    Exact energy from exact diagonalization for comparison; only works for small systems.
    Otherwise the variance is used as an indicator
    Args:
        model: tenpy.models.Model, the model for which to compute the exact ground state.
        max_size: float, maximum size of the Hilbert space for exact diagonalization.
    Returns:
        psi_exact: tenpy.networks.mps.MPS, the exact ground state MPS.
        E_exact: float, the exact ground state energy.
    """
    
    exact_diag = ExactDiag(model, charge_sector=charge_sector, max_size=max_size)
    exact_diag.build_full_H_from_mpo()
    exact_diag.full_diagonalization()
    exact_out = exact_diag.groundstate()  # Ground state object
    E_exact, psi_exact = exact_out[0], exact_out[1]
    return psi_exact, E_exact

def test_alphas(alphas, psii, H_mpo):
    phi_1 = psii.copy()
    H_mpo.apply_naively(phi_1)
    E_alphas = []
    for alpha in tqdm(alphas):
        psi_alpha = psii.add(other=phi_1, alpha=1.0, beta=alpha)
        E_alpha = np.real(H_mpo.expectation_value(psi=psi_alpha))
        E_alphas.append(E_alpha)
    return E_alphas

