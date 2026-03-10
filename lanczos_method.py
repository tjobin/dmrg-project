import numpy as np
from scipy.linalg import eigh
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO
from tenpy.algorithms import dmrg
import matplotlib.pyplot as plt

import numpy as np
from scipy.linalg import eigh

def optimize_lanczos_step(h1, h2, h3, epsilon=1e-8):
    """
    Calculates the optimal variational parameter alpha for a single 
    Lanczos step, with artificial regularization of the metric tensor.
    """
    # 1. Check and regularize the variance
    variance = h2 - h1**2
    if variance < epsilon:
        # Force the variance to be at least epsilon
        # This redefines the second moment h2
        h2_reg = h1**2 + epsilon
    else:
        h2_reg = h2
        
    # Construct the regularized overlap matrix S
    S = np.array([
        [1.0, h1],
        [h1,  h2_reg]
    ])
    
    # Construct the subspace Hamiltonian H_sub
    # Using the regularized h2 to maintain algebraic consistency 
    # between the metric and the Hamiltonian within the subspace
    H_sub = np.array([
        [h1, h2_reg],
        [h2_reg, h3]
    ])
    
    # Solve the generalized eigenvalue problem
    evals, evecs = eigh(H_sub, S)
    
    # Extract the ground state (lowest eigenvalue)
    min_idx = np.argmin(evals)
    E_opt = evals[min_idx]
    c_opt = evecs[:, min_idx]
    
    # Extract the optimal alpha = c_2 / c_1
    if np.isclose(c_opt[0], 0.0, atol=1e-12):
        alpha_opt = np.inf
    else:
        alpha_opt = c_opt[1] / c_opt[0]
        
    return E_opt, alpha_opt

# def optimize_lanczos_step(h1, h2, h3):
#     """
#     Calculates the optimal variational parameter alpha for a single 
#     Lanczos step given the first three moments of the Hamiltonian.
#     """
#     # Construct the overlap matrix S
#     S = np.array([
#         [1.0, h1],
#         [h1,  h2]
#     ])
    
#     # Construct the subspace Hamiltonian H_sub
#     H_sub = np.array([
#         [h1, h2],
#         [h2, h3]
#     ])
#     print(f'h2 - h1**2 = {h2 - h1**2}')
#     # Solve the generalized eigenvalue problem: H_sub * c = E * S * c
#     # eigh is mathematically robust for finding real eigenvalues of symmetric matrices
#     evals, evecs = eigh(H_sub, S)
    
#     # Extract the ground state (lowest eigenvalue)
#     min_idx = np.argmin(evals)
#     E_opt = evals[min_idx]
#     c_opt = evecs[:, min_idx]
    
#     # Extract the optimal alpha = c_2 / c_1
#     # Handle the edge case where the initial state is already an exact eigenstate
#     if np.isclose(c_opt[0], 0.0, atol=1e-12):
#         alpha_opt = np.inf
#     else:
#         alpha_opt = c_opt[1] / c_opt[0]
        
#     return E_opt, alpha_opt

import numpy as np
from scipy.optimize import minimize_scalar

def apply_lanczos_step(psi_i, H_mpo, trunc_params):
    # 1. Create a copy to prevent in-place modification of the ground state
    phi_1 = psi_i.copy()
    
    # 2. Apply the Hamiltonian in place. 
    # The return value is a TruncationError, which we can discard.
    _ = H_mpo.apply(phi_1, options={'compression_method': 'variational', 
                                    'trunc_params': trunc_params})
    
    # 3. Extract h1 and h2
    h1 = psi_i.overlap(phi_1)  # <psi_i | H | psi_i>
    h2 = phi_1.overlap(phi_1)  # <psi_i | H^2 | psi_i>
    
    # 4. Apply the Hamiltonian again to get H^2 |psi_i>
    phi_2 = phi_1.copy()
    _ = H_mpo.apply(phi_2, options={'compression_method': 'variational', 
                                    'trunc_params': trunc_params})
    
    # 5. Extract h3
    h3 = phi_1.overlap(phi_2)  # <psi_i | H H^2 | psi_i>
    
    # Ensure moments are real 
    h1, h2, h3 = np.real(h1), np.real(h2), np.real(h3)

    # 6. Define the exact energy function E(alpha)
    def energy_alpha(alpha):
        numerator = h1 + 2 * alpha * h2 + (alpha**2) * h3
        denominator = 1 + 2 * alpha * h1 + (alpha**2) * h2
        print(numerator, denominator)
        return numerator / denominator

    # 7. Optimize the variational parameter
    print(f"Initial energy (alpha=0): {energy_alpha(0.0):.8f}")
    result = minimize_scalar(fun = lambda alpha: energy_alpha(alpha))
    alpha_star = result.x
    E_alpha = result.fun

    print(f"Optimized alpha: {alpha_star:.6f}, E(alpha): {E_alpha:.8f}, energy_alpha at optimal alpha: {energy_alpha(alpha_star):.8f}")
    # 8. Construct the final state |psi_alpha> = 1.0*|psi_i> + alpha_star*|phi_1>
    # MPS.add(other, alpha, beta) computes alpha|self> + beta|other>
    psi_alpha = psi_i.add(phi_1, 1.0, alpha_star)

    print(f'h1: {h1:.8f}, h2: {h2:.8f}, h3: {h3:.8f}')
    alphas = np.linspace(0, 10, 100)
    energies = [energy_alpha(a) for a in alphas]
    plt.plot(alphas, energies)
    plt.show()
        
    return psi_alpha, E_alpha, alpha_star

def optimize_lanczos_iterative(psi_i: MPS, H_mpo: MPO, trunc_params: dict, max_iter=5):
    """
    Optimizes the Lanczos step alpha using the iterative relation from Eq D4 and D5.
    """
    
    # 1. Build |phi_1> = H|psi_i> ONCE. 
    phi_1 = psi_i.copy()
    _ = H_mpo.apply(phi_1, options={'compression_method': 'variational', 
                                    'trunc_params': trunc_params})
    # Extract the exact first moment
    h1 = np.real(psi_i.overlap(phi_1))
    
    # Initial guess for alpha near zero
    alpha = 0.01
    
    for step in range(max_iter):
        # 2. Construct |psi_alpha> = |psi_i> + alpha * |phi_1>
        psi_alpha = psi_i.copy()
        psi_alpha.add(other=phi_1, alpha=1.0, beta=alpha)

        print(f'psi_alpha norm (before normalization): {psi_alpha.norm}')
        
        # Calculate the squared norm of the trial state
        norm_sq = np.real(psi_alpha.overlap(psi_alpha))
        
        # 3. CRITICAL FIX: Normalize the expectation value.

        # TenPy's expectation_value requires a normalized state to yield physical results
        E_alpha = np.real(H_mpo.expectation_value(psi=psi_alpha))
        
        # 4. Evaluate the overlap metric chi
        # chi = <psi_alpha | psi_i> / <psi_alpha | psi_alpha>
        overlap_alpha_i = np.real(psi_alpha.overlap(psi_i))
        chi = overlap_alpha_i / norm_sq
        
        # 5. Extract h2 and h3 using the paper's iterative formulas
        h2 = ((1.0 / chi - 2.0) * (1.0 + alpha * h1) + 1.0) / (alpha**2)
        h3 = (E_alpha * (1.0 + 2.0 * alpha * h1 + (alpha**2) * h2) - h1 - 2.0 * alpha * h2) / (alpha**2)
        
        # 6. Define the exact energy function E(alpha)
        def energy_alpha(a):
            num = h1 + 2 * a * h2 + (a**2) * h3
            den = 1 + 2 * a * h1 + (a**2) * h2
            return num / den
        
        # 7. Optimize the variational parameter to find the next guess
        result = minimize_scalar(energy_alpha)
        alpha = result.x

        print(f"Iteration {step+1}: E(alpha) = {E_alpha:.8f}, updated alpha = {alpha:.6f}, energy_alpha = {energy_alpha(alpha):.8f}")

    # 8. Construct the final optimized state
    psi_final = psi_i.copy()
    psi_final.add(phi_1, 1.0, alpha)
    
    return psi_final, E_alpha, alpha