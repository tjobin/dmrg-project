import numpy as np
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO
from tqdm import tqdm
from numpy import random

def estimate_hamiltonian_moments(
        psi: MPS,
        H: MPO,
        N_s: int,
        atol: float = 1e-10,
        seed: int = None,
        filename: str = None
        ) -> tuple[float, float, float] :
    """
    Estimates the 1st, 2nd, and 3rd actual moments of the Hamiltonian H
    using perfect independent sampling from a given MPS psi.
    
    Args:
        psi : tenpy.networks.mps.MPS, the matrix product state to sample from.
        H : tenpy.networks.mpo.MPO, the Hamiltonian as a matrix product operator.
        N_s : int, number of samples to generate.
        atol : float, absolute tolerance for numerical stability checks, by default 1e-10.
        seed : int, random seed for reproducibility, by default None.
        filename : str, the function will write detailed sample information to 'out/{filename}.out',
        
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
    psi_1 = psi.copy()
    H.apply_naively(psi_1)
    
    psi_2 = psi_1.copy()
    H.apply_naively(psi_2)
    
    # Pre-allocate arrays for the n-th order local energies
    local_energies_1 = np.zeros(N_s, dtype=complex)
    local_energies_2 = np.zeros(N_s, dtype=complex)
    
    for i in tqdm(range(N_s), desc="Sampling"):
        # Sample a basis product state |s> from the MPS.
        # sample_measurements returns exactly (configuration, amplitude)
        prod_state, exact_overlap = psi.sample_measurements(rng=rng)

        # Construct the product state MPS for the sampled configuration
        s_mps = MPS.from_product_state(psi.sites, prod_state, bc=psi.bc)
        
        # Calculate overlaps via standard tensor contractions
        # Note: s_mps.overlap(ket) computes the inner product <s|ket>
        overlap_0 = exact_overlap        # <s|psi> is exactly calculated during sampling
        overlap_1 = s_mps.overlap(psi_1) # <s|H|psi>
        overlap_2 = s_mps.overlap(psi_2) # <s|H^2|psi>
        
        # Skip states with zero probability to avoid division by zero
        # (Though perfect sampling theoretically precludes this)
        if np.isclose(overlap_0, 0.0, atol=atol):
            print(f'Warning: Sampled state has exceedingly small overlap with psi, skipping sample {i}.')
            continue
            
        # Compute n-th order local energies 
        local_energies_1[i] = overlap_1 / overlap_0 # <s|H|psi> / <s|psi>
        local_energies_2[i] = overlap_2 / overlap_0 # <s|H^2|psi> / <s|psi>
        # Note: We do not compute local_energies_3 directly since it would require H^3|psi>.

        if filename is not None:
            fout.write(f"{i:>10} {np.real(overlap_0):>20.6e} {np.real(local_energies_1[i]):>20.6e} {np.real(local_energies_2[i]):>20.6e} {np.real(local_energies_1[i] * local_energies_2[i]):>20.6e}\n")
    if filename is not None:
        fout.close()
    # The unbiased estimators are the sample means of the local energies
    M_1 = np.mean(local_energies_1) # E[<s|H|psi> / <s|psi>] \approx <psi|H|psi> / <psi|psi>
    M_2 = np.mean(np.conjugate(local_energies_1) * local_energies_1)  # E[|<s|H|psi>|^2] \approx <psi|H^2|psi> / <psi|psi>
    M_3 = np.mean(np.conjugate(local_energies_1) * local_energies_2)  # E[<s|H|psi>* * <s|H^2|psi>] = E[<s|H|psi>^* <s|H^2|psi>] \approx <psi|H^3|psi> / <psi|psi>
    
    # Return purely real components since H is Hermitian
    return np.real(M_1), np.real(M_2), np.real(M_3)