import numpy as np
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO
from tqdm import tqdm


def estimate_hamiltonian_moments(
        psi: MPS,
        H: MPO,
        N_s: int,
        chi_max: int,
        seed: int = None,
        filename: str = None,
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
    if filename is not None:
        fout = open(f'out/{filename}.out','w')
        fout.write("Sampled states |s> \n")
        fout.write(f"{'Sample':>10} {'<s|psi>':>20} {'<psi|H|psi>':>20} {'<psi|H^2|psi>':>20} {'<psi|H^3|psi>':>20}\n")

    # Prepare exact representations of H|psi>, H^2|psi>, and H^3|psi>.
    # MPO.apply_naively(state) modifies the state in place without compression.
    # This guarantees the local energies remain mathematically exact, avoiding 
    # truncation bias in the numerator overlaps.
    phi = psi.copy()
    H.apply(phi, options={'compression_method' : 'zip_up', 'trunc_params' : {'chi_max' : chi_max}})
    phi.canonical_form(renormalize=False)

    local_energies_1 = []
    local_energies_2 = []
    local_energies_3 = []
    

    for _ in tqdm(range(N_s), desc="Sampling states"):
        prod_state_psi, exact_overlap_psi = psi.sample_measurements(rng=rng)
        
        # Construct the product state MPS for the sampled configuration
        s_psi = MPS.from_product_state(psi.sites, prod_state_psi, bc=psi.bc)

        
        # Calculate overlaps via standard tensor contractions
        # Note: s_mps.overlap(ket) computes the inner product <s|ket>
        overlap_0 = exact_overlap_psi        # <s|psi> is exactly calculated during sampling
        overlap_1 = s_psi.overlap(phi) # <s|H|psi>

        loc_E1 = overlap_1 / overlap_0

        # Compute n-th order local energies 
        local_energies_1.append(loc_E1) # <s|H|psi> / <s|psi>
        local_energies_2.append(loc_E1 ** 2) # <s|H^2|psi> / <s|psi>

        
    local_energies_1 = np.array(local_energies_1)
    local_energies_2 = np.array(local_energies_2)


    # cleaned_local_energies_1 = apply_sampling_cutoff(local_energies_1, E_dmrg, c)

    M_1 = np.mean(local_energies_1) # E[<s|H|psi> / <s|psi>] \approx <psi|H|psi> / <psi|psi>
    M_2 = np.mean(local_energies_2)  # E[|<s|H|psi>|^2] \approx <psi|H^2|psi> / <psi|psi>
    
    for _ in tqdm(range(N_s), desc="Sampling states"):
        prod_state_phi, exact_overlap_phi = phi.sample_measurements(rng=rng)
        
        # Construct the product state MPS for the sampled configuration
        s_phi = MPS.from_product_state(phi.sites, prod_state_phi, bc=phi.bc)
        
        # Calculate overlaps via standard tensor contractions
        # Note: s_mps.overlap(ket) computes the inner product <s|ket>
        overlap_0 = exact_overlap_phi        # <s|phi> is exactly calculated during sampling
        overlap_1 = s_phi.overlap(phi, ignore_form=True) # <s|H|phi>
        
        # Compute 3rd order local energies 
        loc_E3 = - M_2 * overlap_1 / overlap_0
        local_energies_3.append(loc_E3)

    
    local_energies_3 = np.array(local_energies_3)


    # The unbiased estimators are the sample means of the local energies
    M_3 = np.mean(local_energies_3) 
    
    if filename is not None:
        fout.write(f'\n M1 = {np.real(M_1)}\n M2 = {np.real(M_2)}\n M3 = {np.real(M_3)}\n')
        fout.close()
    
    # Return purely real components since H is Hermitian
    return np.real(M_1), np.real(M_2), np.real(M_3)#, np.real(var)

def estimate_hamiltonian_moments_cheap(
        psi: MPS,
        H: MPO,
        N_s: int,
        chi_max: int,
        seed: int = None,
        filename: str = None,
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
    if filename is not None:
        fout = open(f'out/{filename}.out','w')
        fout.write("Sampled states |s> \n")
        fout.write(f"{'Sample':>10} {'<s|psi>':>20} {'<psi|H|psi>':>20} {'<psi|H^2|psi>':>20} {'<psi|H^3|psi>':>20}\n")

    # Prepare exact representations of H|psi>, H^2|psi>, and H^3|psi>.
    # MPO.apply_naively(state) modifies the state in place without compression.
    # This guarantees the local energies remain mathematically exact, avoiding 
    # truncation bias in the numerator overlaps.
    phi = psi.copy()
    H.apply(phi, options={'compression_method' : 'zip_up', 'trunc_params' : {'chi_max' : chi_max}})
    # H.apply_naively(phi)
    phi.canonical_form(renormalize=False)
    
    chi = phi.copy()
    H.apply(chi, options={'compression_method' : 'zip_up', 'trunc_params' : {'chi_max' : chi_max}})
    # H.apply_naively(chi)
    chi.canonical_form(renormalize=False)

    local_energies_1 = []
    local_energies_2 = []
    local_energies_3 = []
    

    for i, _ in enumerate(tqdm(range(N_s), desc="Sampling states")):
        prod_state_psi, exact_overlap_psi = psi.sample_measurements(rng=rng)
        
        # Construct the product state MPS for the sampled configuration
        s_psi = MPS.from_product_state(psi.sites, prod_state_psi, bc=psi.bc)

        
        # Calculate overlaps via standard tensor contractions
        # Note: s_mps.overlap(ket) computes the inner product <s|ket>
        overlap_0 = exact_overlap_psi        # <s|psi> is exactly calculated during sampling
        overlap_1 = s_psi.overlap(phi) # <s|H|psi>
        overlap_2 = s_psi.overlap(chi) # <s|H^2|psi>

        loc_E1 = overlap_1 / overlap_0
        loc_E2 = overlap_2 / overlap_0
        

        # Compute n-th order local energies 
        local_energies_1.append(loc_E1) # <s|H|psi> / <s|psi>
        local_energies_2.append(loc_E2) # <s|H^2|psi> / <s|psi>
        local_energies_3.append(loc_E1 * loc_E2) # <s|H^2|psi> / <s|psi>
        # Note: We do not compute local_energies_3 directly since it would require H^3|psi>.

        # overlap_0 = exact_overlap_phi        # <s|H|psi> is exactly calculated during sampling
        
    local_energies_1 = np.array(local_energies_1)
    local_energies_2 = np.array(local_energies_2)
    local_energies_3 = np.array(local_energies_3)

    # cleaned_local_energies_1 = apply_sampling_cutoff(local_energies_1, E_dmrg, c)

    M_1 = np.mean(local_energies_1) # E[<s|H|psi> / <s|psi>] \approx <psi|H|psi> / <psi|psi>
    M_2 = np.mean(local_energies_2)  # E[|<s|H|psi>|^2] \approx <psi|H^2|psi> / <psi|psi>
    M_3 = np.mean(local_energies_3)  # E[<s|H|psi>* * <s|H^2|psi>] = E[<s|H|psi>^* <s|H^2|psi>] \approx <psi|H^3|psi> / <psi|psi>

        # if filename is not None:
        #     fout.write(f"{i:>10} {np.real(overlap_0):>20.6e} {np.real(loc_E1):>20.6e} {np.real(loc_E2):>20.6e} {np.real(loc_E1 * loc_E2):>20.6e}\n"
    
    if filename is not None:
        fout.write(f'\n M1 = {np.real(M_1)}\n M2 = {np.real(M_2)}\n M3 = {np.real(M_3)}\n')
        fout.close()
    
    # Return purely real components since H is Hermitian
    return np.real(M_1), np.real(M_2), np.real(M_3)#, np.real(var)

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