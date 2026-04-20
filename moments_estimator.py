import numpy as np
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO
from tqdm import tqdm
import json
import os

def estimate_hamiltonian_moments_cheap(
        psi: MPS,
        H: MPO,
        N_s: int,
        chi_max: int,
        seed: int | None = None,
        filename: str | None = None,
        ) -> tuple[float, float, float] :

    """
    Estimates the 1st, 2nd, and 3rd actual moments of the Hamiltonian H
    using perfect independent sampling from a given MPS psi.
    
    Args:
        psi : tenpy.networks.mps.MPS, the matrix product state to sample from.
        H : tenpy.networks.mpo.MPO, the Hamiltonian as a matrix product operator.
        N_s : int, number of samples to generate.
        seed : int, random seed for reproducibility, by default None.
        filename : str, the function will write detailed sample information to 'out/{filename}.out',
        c : float, fraction of samples to keep based on closest local energy to E_dmrg, by default 0.9.
        
    Returns:
        M1: float, the estimated first moment <psi|H|psi> / <psi|psi>.
        M2: float, the estimated second moment <psi|H^2|psi> / <psi|psi>.
        M3: float, the estimated third moment <psi|H^3|psi> / <psi|psi>.
    """

    rng = np.random.default_rng(seed)  # Create a random number generator with the given seed for reproducibility

    phi = psi.copy()
    H.apply(phi, options={'compression_method' : 'zip_up', 'trunc_params' : {'chi_max' : chi_max}})
    
    chi = phi.copy()
    H.apply(chi, options={'compression_method' : 'zip_up', 'trunc_params' : {'chi_max' : chi_max}})

    local_energies_1 = []
    local_energies_2 = []
    local_energies_3 = []
    data_to_save = {}


    for i in tqdm(range(N_s), desc="Sampling states"):
        prod_state_psi, exact_overlap_psi = psi.sample_measurements(rng=rng)
        
        # Construct the product state MPS for the sampled configuration
        s_psi = MPS.from_product_state(psi.sites, prod_state_psi, bc=psi.bc)

        # Calculate overlaps via standard tensor contractions
        overlap_0 = exact_overlap_psi  # <s|psi> is exactly calculated during sampling
        overlap_1 = s_psi.overlap(phi) # <s|H|psi>
        overlap_2 = s_psi.overlap(chi) # <s|H^2|psi>

        loc_E1 = overlap_1 / overlap_0
        loc_E2 = overlap_2 / overlap_0

        # Compute n-th order local energies 
        local_energies_1.append(loc_E1) # <s|H|psi> / <s|psi>
        local_energies_2.append(loc_E2) # <s|H^2|psi> / <s|psi>
        local_energies_3.append(loc_E1 * loc_E2) # <s|H^3|psi> / <s|psi>

        data_to_save[str(i)] = {
            "h1": float(loc_E1),
            "h2": float(loc_E2),
            "h3": float(loc_E1 * loc_E2)
        }

    json_filename = f'log_sampling/{filename}.json'
    with open(json_filename, 'w') as f:
        json.dump(data_to_save, f, indent=4)
    
    local_energies_1 = np.array(local_energies_1)
    local_energies_2 = np.array(local_energies_2)
    local_energies_3 = np.array(local_energies_3)


    M_1 = float(np.mean(local_energies_1)) # E[<s|H|psi> / <s|psi>] \approx <psi|H|psi> / <psi|psi>
    M_2 = float(np.mean(local_energies_2))  # E[|<s|H|psi>|^2] \approx <psi|H^2|psi> / <psi|psi>
    M_3 = float(np.mean(local_energies_3))  # E[<s|H|psi>* * <s|H^3|psi>] = E[<s|H|psi>^* <s|H^2|psi>] \approx <psi|H^3|psi> / <psi|psi>

    # Return purely real components since H is Hermitian
    return M_1, M_2, M_3

def clean_local_energies(E_L, E_dmrg, c):
    """
    Applies a symmetric cutoff to local energies and local second moments 
    based on a cutoff ratio c.
    
    Parameters
    ----------
    E_L : np.ndarray
        Array of first-order local energies.
    E2_L : np.ndarray
        Array of second-order local energies (local second moments).
    E_dmrg : float
        The reference exact energy from the DMRG sweep.
    c : float
        The cutoff ratio (e.g., 0.90 for 90%).
        
    Returns
    -------
    tuple of np.ndarray
        (cleaned_E_L, cleaned_E2_L)
    """
    # 1. Calculate the deviations from the respective reference values
    E2_ref = E_dmrg ** 2
    eps = E_L - E_dmrg
    
    # 2. Determine the maximum allowed deviations (epsilon_max and eta_max).
    # Taking the c-th percentile of the absolute deviations guarantees that
    # exactly a fraction 'c' of the original samples fall within the bounds.
    eps_max = np.percentile(np.abs(eps), c * 100)
    
    # 3. Apply the piecewise function f(epsilon) to clip extreme tails symmetrically
    eps_clipped = np.clip(eps, -eps_max, eps_max)
    
    # 4. Reconstruct the bounded local energies
    E_L_cleaned = E_dmrg + eps_clipped
    
    return E_L_cleaned

def get_mpo_moments_bruteforce(
        psi: MPS,
        H: MPO
    ) -> tuple[float, float, float] :

    psi1 = psi.copy()

    H.apply_naively(psi1)

    psi2 = psi1.copy()
    H.apply_naively(psi2)

    h1 = psi.overlap(psi1, ignore_form=True)
    h2 = psi.overlap(psi2, ignore_form=True)
    h3 = psi1.overlap(psi2, ignore_form=True)

    return h1, h2, h3

import numpy as np

def apply_sampling_cutoff(local_energies, E_dmrg, c):
    """
    Applies the symmetric piecewise cutoff function to an array of sampled 
    local energies to mitigate fat-tailed fluctuations.
    
    Parameters
    ----------
    local_energies : numpy.ndarray
        1D array of sampled local energies (size N_s).
    E_dmrg : float
        The reference ground-state energy obtained from the DMRG sweep.
    c : float
        The cutoff ratio (between 0.0 and 1.0), representing the fraction 
        of local energies left unchanged.
        
    Returns
    -------
    numpy.ndarray
        The cleaned array of local energies after applying the cutoff bias.
    """
    # Calculate the deviations from the reference DMRG energy
    deviations = local_energies - E_dmrg
    
    # The cutoff is based on the magnitude of the deviations.
    # To leave a fraction 'c' of the data untouched, we find the c-th 
    # quantile (e.g., the 90th percentile for c=0.9) of the absolute deviations.
    abs_deviations = np.abs(deviations)
    epsilon_max = np.percentile(abs_deviations, c * 100)
    
    # Apply the piecewise function f(epsilon) to clip the tails symmetrically
    # f(epsilon) = -epsilon_max if epsilon < -epsilon_max
    # f(epsilon) = epsilon_max if epsilon > epsilon_max
    # f(epsilon) = epsilon otherwise
    clipped_deviations = np.clip(deviations, -epsilon_max, epsilon_max)
    
    # Reconstruct the cleaned local energies
    cleaned_local_energies = E_dmrg + clipped_deviations
    
    return cleaned_local_energies