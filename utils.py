from tenpy.models import Model
from tenpy.algorithms.exact_diag import ExactDiag
from tenpy.networks.mps import MPS

def get_exact_psi_and_h1(
        model: Model,
        max_size: float = 1e12
        ) -> tuple[MPS, float] :
    """
    Exact energy from exact diagonalization for comparison; only works for small systems.
    Otherwise the variance is used as an indicator
    Args:
        model: The model for which to compute the exact ground state.
        max_size: Maximum size of the Hilbert space for exact diagonalization.
    Returns:
        psi_exact: The exact ground state MPS.
        E_exact: The exact ground state energy.    
    """
    
    exact_diag = ExactDiag(model, max_size=max_size)
    exact_diag.build_full_H_from_mpo()
    exact_diag.full_diagonalization()
    exact_out = exact_diag.groundstate()  # Ground state object
    E_exact, psi_exact = exact_out[0], exact_out[1]
    return psi_exact, E_exact