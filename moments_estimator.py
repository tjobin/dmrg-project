import numpy as np
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO
from tqdm import tqdm

def estimate_hamiltonian_moments(
        psi:MPS,
        H:MPO,
        N_s:int
        ) -> tuple[float, float, float] :
    """
    Estimates the 1st, 2nd, and 3rd actual moments of the Hamiltonian H
    using perfect independent sampling from a given MPS psi.
    
    Parameters
    ----------
    psi : tenpy.networks.mps.MPS
        The matrix product state to sample from.
    H : tenpy.networks.mpo.MPO
        The Hamiltonian as a matrix product operator.
    N_s : int
        The number of samples to generate.
        
    Returns
    -------
    tuple
        Estimated actual moments (M_1, M_2, M_3)
    """

    # Prepare exact representations of H|psi>, H^2|psi>, and H^3|psi>.
    # MPO.apply_naively(state) modifies the state in place without compression.
    # This guarantees the local energies remain mathematically exact, avoiding 
    # truncation bias in the numerator overlaps.
    psi_1 = psi.copy()
    H.apply_naively(psi_1)
    
    psi_2 = psi_1.copy()
    H.apply_naively(psi_2)
    
    # Pre-allocate arrays for the n-th order local energies
    local_energies_1 = np.zeros(N_s, dtype=complex)
    local_energies_2 = np.zeros(N_s, dtype=complex)
    
    for i in tqdm(range(N_s), desc="Sampling"):
        # Sample a basis product state |s> from the MPS.
        sample_data = psi.sample_measurements()
        prod_state = sample_data[0] if isinstance(sample_data, tuple) else sample_data

        # Construct the product state MPS for the sampled configuration
        s_mps = MPS.from_product_state(psi.sites, prod_state, bc=psi.bc)
        
        # Calculate overlaps via standard tensor contractions
        # Note: s_mps.overlap(ket) computes the inner product <s|ket>
        overlap_0 = s_mps.overlap(psi)   # <s|psi>
        overlap_1 = s_mps.overlap(psi_1) # <s|H|psi>
        overlap_2 = s_mps.overlap(psi_2) # <s|H^2|psi>
        
        # Skip states with zero probability to avoid division by zero
        # (Though perfect sampling theoretically precludes this)
        if np.isclose(overlap_0, 0.0):
            continue
            
        # Compute n-th order local energies 
        local_energies_1[i] = overlap_1 / overlap_0 # <s|H|psi> / <s|psi>
        local_energies_2[i] = overlap_2 / overlap_0 # <s|H^2|psi> / <s|psi>
        # Note: We do not compute local_energies_3 directly since it would require H^3|psi>.

    # The unbiased estimators are the sample means of the local energies
    M_1 = np.mean(local_energies_1) # E[<s|H|psi> / <s|psi>] \approx <psi|H|psi> / <psi|psi>
    M_2 = np.mean(np.conjugate(local_energies_1) * local_energies_1)  # E[|<s|H|psi>|^2] \approx <psi|H^2|psi> / <psi|psi>
    M_3 = np.mean(np.conjugate(local_energies_1) * local_energies_2)  # E[<s|H|psi>* * <s|H^2|psi>] = E[<s|H|psi>^* <s|H^2|psi>] \approx <psi|H^3|psi> / <psi|psi>
    
    # Return purely real components since H is Hermitian
    return np.real(M_1), np.real(M_2), np.real(M_3)