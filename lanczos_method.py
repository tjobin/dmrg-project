import numpy as np
from scipy.linalg import eigh

def optimize_lanczos_step(
        h1: float,
        h2: float,
        h3: float,
        epsilon: float =1e-14
        ) -> tuple[float, float] :
    
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
        print(f'Energy variance too small: {variance:.2e}, regularized to {epsilon:.2e}.')
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
    
    # Solve the generalized eigenvalue problem H_sub c = E S c
    evals, evecs = eigh(H_sub, S)
    
    # Extract the ground state (lowest eigenvalue)
    min_idx = np.argmin(evals)
    E_opt = evals[min_idx]
    c_opt = evecs[:, min_idx]
    
    # Extract the optimal alpha = c_2 / c_1
    if np.isclose(c_opt[0], 0.0, atol=1e-12):
        alpha_opt = np.inf
        print('Warning: c_1 is very close to zero, alpha is set to infinity.')
    else:
        alpha_opt = c_opt[1] / c_opt[0]

    return E_opt, alpha_opt