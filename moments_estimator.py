import numpy as np
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO
from tqdm import tqdm
import json
import os
from joblib import Parallel, delayed

def estimate_hamiltonian_moments(
        psi: MPS,
        H: MPO,
        N_s: int,
        chi_max: int,
        E_ref: float,
        c: float = 0.85,
        mad_threshold: float = 15.0,
        seed: int | None = None,
        sampling_filepath: str | None = None
        ) -> tuple[float, float, float] :

    """
    Estimates the 1st, 2nd, and 3rd actual moments of the Hamiltonian H
    using perfect independent sampling from a given MPS psi.
    
    Args:
        psi : tenpy.networks.mps.MPS, the matrix product state to sample from.
        H : tenpy.networks.mpo.MPO, the Hamiltonian as a matrix product operator.
        N_s : int, number of samples to generate.
        seed : int, random seed for reproducibility, by default None.
        sampling_filepath : str, the function will write detailed sample information to 'log_sampling/{filename}.json',
        c : float, fraction of samples to keep based on closest local energy to E_dmrg, by default 0.9.
        
    Returns:
        M1: float, the estimated first moment <psi|H|psi> / <psi|psi>.
        M2: float, the estimated second moment <psi|H^2|psi> / <psi|psi>.
        M3: float, the estimated third moment <psi|H^3|psi> / <psi|psi>.
    """

    h1 = np.zeros(N_s)
    h2 = np.zeros(N_s)
    h3 = np.zeros(N_s)
    data_to_save = {}

    def compute_sample(i):
        # Create an independent random number generator for this sample
        local_rng = np.random.default_rng((seed + i) if seed is not None else None)
        
        prod_state_psi, exact_overlap_psi = psi.sample_measurements(rng=local_rng, complex_amplitude=True)
        
        # Construct the product state MPS for the sampled configuration
        s_psi = MPS.from_product_state(psi.sites, prod_state_psi, bc=psi.bc, unit_cell_width=psi.unit_cell_width)
        # 1. Apply H to the product state exactly (chi becomes ~14)
        H_s = s_psi.copy()
        H.apply_naively(H_s) 

        # 2. Apply H again exactly (chi becomes ~196)
        H2_s = H_s.copy()
        H.apply_naively(H2_s) 

        # 3. Compute exact overlaps
        # A.overlap(B) in TeNPy computes <A|B>. 
        # H_s.overlap(psi) computes <H s | psi> = <s | H^\dagger | psi> = <s | H | psi>
        overlap_0 = exact_overlap_psi
        overlap_1 = H_s.overlap(psi) 
        overlap_2 = H2_s.overlap(psi)

        loc_E1 = np.real(overlap_1 / overlap_0)
        loc_E2 = np.real(overlap_2 / overlap_0)
        loc_E3 = np.conj(loc_E1) * loc_E2        

        return i, float(np.real(loc_E1)), float(np.real(loc_E2)), float(np.real(loc_E3))
    
    # Use process-based parallelism (loky) to completely bypass the Python GIL.
    # returning as a generator allows tqdm to track actual job completions!
    parallel_task = Parallel(n_jobs=4, backend="loky", return_as="generator")(
        delayed(compute_sample)(i) for i in range(N_s)
    )

    for i, loc_E1, loc_E2, loc_E3 in tqdm(parallel_task, total=N_s, desc="Sampling states"):
        h1[i] = loc_E1
        h2[i] = loc_E2
        h3[i] = loc_E3

    # 1. Filter the arrays
    cleaned_h1, cleaned_h2, cleaned_h3 = filter_samples_by_mad(h1, h2, h3, mad_threshold=mad_threshold)
    
    # Number of samples remaining after rejection
    N_kept = len(cleaned_h1)

    # 2. Populate the dictionary with the kept samples to verify the distribution
    data_to_save = {}
    for i in range(N_kept):
        data_to_save[str(i)] = {
            "h1": float(cleaned_h1[i]),
            "h2": float(cleaned_h2[i]),
            "h3": float(cleaned_h3[i])
        }

    # 3. Save to JSON
    json_filename = f'sampling_chi{chi_max}_Ns{N_s}_seed{seed}.json'
    os.makedirs(f'{sampling_filepath}', exist_ok=True)
    with open(f'{sampling_filepath}{json_filename}', 'w') as f:
        json.dump(data_to_save, f, indent=4)

    M_1 = float(np.mean(cleaned_h1)) # E[<s|H|psi> / <s|psi>] \approx <psi|H|psi> / <psi|psi>
    M_2 = float(np.mean(cleaned_h2))  # E[|<s|H|psi>|^2] \approx <psi|H^2|psi> / <psi|psi>
    M_3 = float(np.mean(cleaned_h3))  # E[<s|H|psi>* * <s|H^3|psi>] = E[<s|H|psi>^* <s|H^2|psi>] \approx <psi|H^3|psi> / <psi|psi>
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

def apply_coordinated_percentile_clipping(
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

def apply_independent_mad_clipping(
        h1: np.ndarray,
        h2: np.ndarray,
        h3: np.ndarray,
        mad_threshold: float = 15.0
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Clips outliers based on distance from the median, preserving the bulk of the data.
    """
    def mad_clip(arr, thresh):
        med = np.median(arr)
        # Calculate the Median Absolute Deviation
        mad = np.median(np.abs(arr - med)) 
        
        # Guard against zero-variance edge cases
        if mad < 1e-12: 
            return arr
            
        lower_bound = med - thresh * mad
        upper_bound = med + thresh * mad
        return np.clip(arr, lower_bound, upper_bound)

    h1_clean = mad_clip(h1, mad_threshold)
    h2_clean = mad_clip(h2, mad_threshold)
    h3_clean = mad_clip(h3, mad_threshold)
    
    return h1_clean, h2_clean, h3_clean

def filter_samples_by_percentile(
    h1: np.ndarray,
    h2: np.ndarray,
    h3: np.ndarray,
    c: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Removes entire samples if they fall outside the central 'c' fraction 
    of the distribution for any of the local moments.
    """
    if c >= 1.0:
        return h1, h2, h3

    # Calculate tail percentages
    p_lower = (1.0 - c) / 2.0 * 100.0
    p_upper = 100.0 - p_lower
    
    # Calculate bounds for each moment
    h1_bounds = np.percentile(h1, [p_lower, p_upper])
    h2_bounds = np.percentile(h2, [p_lower, p_upper])
    h3_bounds = np.percentile(h3, [p_lower, p_upper])
    
    # Create boolean masks (True if the sample is within bounds)
    valid_h1 = (h1 >= h1_bounds[0]) & (h1 <= h1_bounds[1])
    valid_h2 = (h2 >= h2_bounds[0]) & (h2 <= h2_bounds[1])
    valid_h3 = (h3 >= h3_bounds[0]) & (h3 <= h3_bounds[1])
    
    # The crucial step: A sample configuration |n> is only kept if it is 
    # well-behaved across ALL local operators. This preserves the 
    # physical consistency of the subset state.
    valid_mask = valid_h1 & valid_h2 & valid_h3
    
    # Apply the mask. The returned arrays will have length N_kept <= N_s.
    return h1[valid_mask], h2[valid_mask], h3[valid_mask]

def filter_samples_by_mad(
        h1: np.ndarray,
        h2: np.ndarray,
        h3: np.ndarray,
        mad_threshold: float = 15.0
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Removes entire samples if they fall outside the acceptable Median Absolute Deviation (MAD)
    threshold for any of the local moments. This preserves the physical consistency of the subset state.
    
    Args:
        h1, h2, h3: 1D numpy arrays of the raw sampled local moments.
        mad_threshold: The number of MADs away from the median a sample can be before being discarded.
    """
    def get_valid_mask(arr, thresh):
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        
        # Guard against zero-variance to avoid rejecting everything in perfectly uniform arrays
        if mad < 1e-12:
            mad = 1e-12
            
        lower_bound = med - thresh * mad
        upper_bound = med + thresh * mad
        
        return (arr >= lower_bound) & (arr <= upper_bound)

    # Generate boolean masks for each moment
    valid_h1 = get_valid_mask(h1, mad_threshold)
    valid_h2 = get_valid_mask(h2, mad_threshold)
    valid_h3 = get_valid_mask(h3, mad_threshold)
    
    # The crucial step: A sample configuration |n> is only kept if it is 
    # well-behaved across ALL local operators simultaneously.
    valid_mask = valid_h1 & valid_h2 & valid_h3
    
    # Apply the mask. The returned arrays will have length N_kept <= N_s.
    return h1[valid_mask], h2[valid_mask], h3[valid_mask]