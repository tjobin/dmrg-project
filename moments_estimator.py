import numpy as np
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO
from tqdm import tqdm
import json
import os

def estimate_hamiltonian_moments(
        psi: MPS,
        H: MPO,
        N_s: int,
        chi_max: int,
        E_ref: float,
        c: float = 0.85,
        seed: int | None = None,
        json_filepath: str | None = None,
        ) -> tuple[float, float, float] :

    """
    Estimates the 1st, 2nd, and 3rd actual moments of the Hamiltonian H
    using perfect independent sampling from a given MPS psi.
    
    Args:
        psi : tenpy.networks.mps.MPS, the matrix product state to sample from.
        H : tenpy.networks.mpo.MPO, the Hamiltonian as a matrix product operator.
        N_s : int, number of samples to generate.
        seed : int, random seed for reproducibility, by default None.
        json_filename : str, the function will write detailed sample information to 'log_sampling/{filename}.json',
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


    json_filename = f'sampling_chi{chi_max}_Ns{N_s}_seed{seed}_c{c}.json'
    os.makedirs(f'{json_filepath}', exist_ok=True)
    with open(f'{json_filepath}{json_filename}', 'w') as f:
        json.dump(data_to_save, f, indent=4)
    
    local_energies_1 = np.array(local_energies_1)
    local_energies_2 = np.array(local_energies_2)
    local_energies_3 = np.array(local_energies_3)

    cleaned_local_energies_1, cleaned_local_energies_2, cleaned_local_energies_3 = apply_coordinated_cutoff(
        local_energies_1,
        local_energies_2,
        local_energies_3,
        E_ref,
        c=c)


    M_1 = float(np.mean(cleaned_local_energies_1)) # E[<s|H|psi> / <s|psi>] \approx <psi|H|psi> / <psi|psi>
    M_2 = float(np.mean(cleaned_local_energies_2))  # E[|<s|H|psi>|^2] \approx <psi|H^2|psi> / <psi|psi>
    M_3 = float(np.mean(cleaned_local_energies_3))  # E[<s|H|psi>* * <s|H^3|psi>] = E[<s|H|psi>^* <s|H^2|psi>] \approx <psi|H^3|psi> / <psi|psi>

    # Return purely real components since H is Hermitian
    return M_1, M_2, M_3

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

def apply_coordinated_cutoff(
        h1: np.ndarray,
        h2: np.ndarray,
        h3: np.ndarray,
        E_ref: float,
        c: float
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies a coordinated sample-wise dampening to all three moments,
    maintaining the array size so np.mean() computes the correct normalization.
    """
    # 1. Calculate the deviation of the primary energy
    eps = h1 - E_ref
    eps_max = np.percentile(np.abs(eps), c * 100)
    
    # 2. Calculate a sample-wise dampening factor 'alpha'
    # For normal samples, alpha = 1.0
    # For outliers, alpha scales the deviation down to exactly eps_max
    alpha = np.where(np.abs(eps) > eps_max, eps_max / (np.abs(eps) + 1e-12), 1.0)
    
    # 3. Clip h1 (mathematically identical to the paper's piecewise function)
    h1_clean = E_ref + alpha * eps
    
    # 4. Clip h2 and h3 coordinately by dampening their deviations 
    # from their robust centers (medians) using the exact same alpha factor.
    med_h2 = np.median(h2)
    med_h3 = np.median(h3)
    
    h2_clean = med_h2 + alpha * (h2 - med_h2)
    h3_clean = med_h3 + alpha * (h3 - med_h3)
    
    return h1_clean, h2_clean, h3_clean