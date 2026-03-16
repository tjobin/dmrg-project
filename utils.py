from tenpy.models import Model
from tenpy.algorithms.exact_diag import ExactDiag
from tenpy.networks.mps import MPS

# exact energies for quick reference (Lx, Ly) -> E_exact
EXACT_ENERGIES = {
    (3, 2): -12.7907317932865769,
    (4, 2): -18.0367928615768918,
    }

def get_exact_psi_and_E(
        model: Model,
        charge_sector: list[int] | int | None = None,
        max_size: float = 1e12
        ) -> tuple[MPS, float] :
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