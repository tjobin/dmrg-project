import numpy as np
import warnings
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO
from moments_estimator import estimate_hamiltonian_moments_cheap, get_mpo_moments_bruteforce


def get_optimized_alphas(
        h1: float,
        h2: float,
        h3: float,
        ) -> tuple[float, float] :
    """
    Calculates the optimal variational parameter alpha for a single 
    Lanczos step from the analytical solution.
    Args:
        h1: float, the first moment of the Hamiltonian.
        h2: float, the second moment of the Hamiltonian.
        h3: float, the third moment of the Hamiltonian.
    Returns:
        alpha_p: float, first anayltical solution.
        alpha_m: float, second analytical solution
    """    
    if (h3-h1*h2)**2 - 4*(h2-h1**2)*(h1*h3-h2**2) <= 0:
        alpha_p = (-(h3 - h1*h2)) / (2*(h1*h3-h2**2) + 1e-12)
        alpha_m = alpha_p
        warnings.warn("Negative square root in alpha optimization; setting it to 0.")
    else:
        alpha_p = (-(h3 - h1*h2) + np.sqrt((h3-h1*h2)**2 - 4*(h2-h1**2)*(h1*h3-h2**2))) / (2*(h1*h3-h2**2) + 1e-12)
        alpha_m = (-(h3 - h1*h2) - np.sqrt((h3-h1*h2)**2 - 4*(h2-h1**2)*(h1*h3-h2**2))) / (2*(h1*h3-h2**2) + 1e-12)
    if alpha_p > 10**12:
        alpha_p = 10**12
    elif alpha_p < -10**12:
        alpha_p = -10**12
    if alpha_m > 10**12:
        alpha_m = 10**12
    elif alpha_m < -10**12:
        alpha_m = -10**12
    return alpha_p, alpha_m

def lanczos_step_sampled(
        psi: MPS,
        H: MPO,
        N_s: int,
        chi_max: int,
        seed: int | None = None,
        filename: str | None = None):
    
    h1, h2, h3 = estimate_hamiltonian_moments_cheap(
        psi = psi,
        H = H,
        N_s = N_s,
        chi_max = chi_max,
        seed = seed,
        filename = filename
    )

    alpha_p, alpha_m = get_optimized_alphas(h1, h2, h3)
    phi_1 = psi.copy()
    H.apply_naively(phi_1)

    psi_alpha_p = psi.add(other=phi_1, alpha=1.0, beta=alpha_p)
    psi_alpha_m = psi.add(other=phi_1, alpha=1.0, beta=alpha_m)

    E_alpha_p = np.real(H.expectation_value(psi=psi_alpha_p))
    E_alpha_m = np.real(H.expectation_value(psi=psi_alpha_m))

    if E_alpha_p < E_alpha_m:
        E_alpha = E_alpha_p
        alpha_star = alpha_p
        psi_alpha = psi_alpha_p
    else:
        E_alpha = E_alpha_m
        alpha_star = alpha_m
        psi_alpha = psi_alpha_m

    return E_alpha, psi_alpha, alpha_star

def lanczos_step_exact(
        psi: MPS,
        H: MPO,):
    
    h1_exact, h2_exact, h3_exact = get_mpo_moments_bruteforce(
        psi = psi,
        H = H,
    )

    alpha_p, alpha_m = get_optimized_alphas(h1_exact, h2_exact, h3_exact)
    phi_1 = psi.copy()
    H.apply_naively(phi_1)

    psi_alpha_p = psi.add(other=phi_1, alpha=1.0, beta=alpha_p)
    psi_alpha_m = psi.add(other=phi_1, alpha=1.0, beta=alpha_m)

    E_alpha_p = np.real(H.expectation_value(psi=psi_alpha_p))
    E_alpha_m = np.real(H.expectation_value(psi=psi_alpha_m))

    if E_alpha_p < E_alpha_m:
        E_alpha_exact = E_alpha_p
        alpha_star_exact = alpha_p
        psi_alpha_exact = psi_alpha_p
    else:
        E_alpha_exact = E_alpha_m
        alpha_star_exact = alpha_m
        psi_alpha_exact = psi_alpha_m

    return E_alpha_exact, psi_alpha_exact, alpha_star_exact