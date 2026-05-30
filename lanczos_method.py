import numpy as np
import warnings
import time
import resource
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO
from moments_estimator import estimate_hamiltonian_moments, get_mpo_moments_bruteforce


def lanczos_step_sampled_v2(
        psi: MPS,
        H: MPO,
        N_s: int,
        chi_max: int,
        E_ref: float,
        c: float,
        mad_threshold: float = 10.0,
        seed: int | None = None,
        sampling_filepath: str | None = None
) -> tuple[float, float, float, float, float, float, float] :
    
    start_time = time.perf_counter()
    
    h1, h2, h3 = estimate_hamiltonian_moments(
        psi = psi,
        H = H,
        N_s = N_s,
        chi_max = chi_max,
        E_ref = E_ref,
        c = c,
        mad_threshold = mad_threshold,
        seed = seed,
        sampling_filepath = sampling_filepath
    )
    
    E_alpha, alpha_star = get_optimized_energy_and_alpha(h1, h2, h3)

    end_time = time.perf_counter()
    wall_time_seconds = end_time - start_time

    # MacOS ru_maxrss returns bytes. Divide by 1024^2 for MB.
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_memory_mb = usage.ru_maxrss / (1024 * 1024)

    return E_alpha, alpha_star, h1, h2, h3, wall_time_seconds, peak_memory_mb


def lanczos_step_sampled(
        psi: MPS,
        H: MPO,
        N_s: int,
        chi_max: int,
        E_ref: float,
        c: float,
        seed: int | None = None,
        sampling_filepath: str | None = None):
    
    h1, h2, h3 = estimate_hamiltonian_moments(
        psi = psi,
        H = H,
        N_s = N_s,
        chi_max = chi_max,
        E_ref = E_ref,
        c = c,
        seed = seed,
        sampling_filepath = sampling_filepath
    )

    alpha_p, alpha_m = get_optimized_alphas(h1, h2, h3)
    phi = psi.copy()
    # H.apply(phi, options={'compression_method' : 'zip_up', 'trunc_params' : {'chi_max' : chi_max}})
    H.apply_naively(phi)

    psi_alpha_p = psi.add(other=phi, alpha=1.0, beta=alpha_p)
    psi_alpha_m = psi.add(other=phi, alpha=1.0, beta=alpha_m)

    E_alpha_p = np.real(H.expectation_value(psi=psi_alpha_p))
    E_alpha_m = np.real(H.expectation_value(psi=psi_alpha_m))

    # E_alpha_p = (h1 + 2*alpha_p*h2 + alpha_p ** 2 * h3) / (1 + 2 * alpha_p * h1 + alpha_p ** 2 * h2)
    # E_alpha_m = (h1 + 2*alpha_m*h2 + alpha_m ** 2 * h3) / (1 + 2 * alpha_m * h1 + alpha_m ** 2 * h2)

    if E_alpha_p < E_alpha_m:
        E_alpha = E_alpha_p
        alpha_star = alpha_p
        # psi_alpha = psi_alpha_p
    else:
        E_alpha = E_alpha_m
        alpha_star = alpha_m
        # psi_alpha = psi_alpha_m

    return E_alpha, alpha_star # E_alpha, psi_alpha, alpha_star



def lanczos_step_exact(
        psi: MPS,
        H: MPO
        ):
    
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

def get_optimized_energy_and_alpha(h1, h2, h3):
    """
    Calculates the optimal energy and variational parameter alpha for a 
    single Lanczos step |psi'> = (1 + alpha * H) |psi>.
    
    Parameters:
    -----------
    h1 : float
        The first moment <H> of the current state.
    h2 : float
        The second moment <H^2> of the current state.
    h3 : float
        The third moment <H^3> of the current state.
        
    Returns:
    --------
    E_opt : float
        The minimum energy in the expanded Krylov subspace.
    alpha_opt : float
        The corresponding optimal variational parameter alpha.
    """
    # Calculate the variance A = H_2 - H_1^2
    V = h2 - h1**2
    
    # If the variance is essentially zero, the state is already an exact eigenstate.
    # A Lanczos step cannot lower the energy further.
    if V < 1e-14:
        return h1, 0.0
        
    # Define the coefficients for the quadratic equation A*E^2 - B*E + C = 0
    A = V
    B = h3 - h1 * h2
    C = h1 * h3 - h2**2
    
    # Compute the discriminant of the secular equation
    discriminant = B**2 - 4 * A * C
    
    # Safeguard against statistical noise in h3 making the discriminant negative
    if discriminant < 0:
        warnings.warn(
            f"Negative discriminant ({discriminant:.2e}) encountered. "
            "This is usually caused by statistical noise in H_3. "
            "Truncating discriminant to 0."
        )
        discriminant = 0.0
        
    # The lowest energy in the Krylov subspace corresponds to the lower root
    E_opt = (B - np.sqrt(discriminant)) / (2 * A)
    
    # Compute the optimal alpha that corresponds to the E_opt eigenvector
    alpha_denominator = h2 - E_opt * h1
    
    if abs(alpha_denominator) < 1e-15:
        alpha_opt = 0.0
    else:
        alpha_opt = (E_opt - h1) / alpha_denominator
        
    return E_opt, alpha_opt

def get_theoretical_energy_surface(alphas, h1, h2, h3):
    """
    Computes the theoretical energy surface E(alpha) = <psi(alpha)|H|psi(alpha)> / <psi(alpha)|psi(alpha)>
    given the first three Hamiltonian moments.
    """
    alphas = np.array(alphas)
    numerator = h1 + 2 * alphas * h2 + (alphas ** 2) * h3
    denominator = 1 + 2 * alphas * h1 + (alphas ** 2) * h2
    return numerator / denominator