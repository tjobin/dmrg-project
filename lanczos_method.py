import numpy as np
from scipy.linalg import eigh

def optimize_lanczos_step(h1, h2, h3):
    """
    Calculates the optimal variational parameter alpha for a single 
    Lanczos step given the first three moments of the Hamiltonian.
    """
    # Construct the overlap matrix S
    S = np.array([
        [1.0, h1],
        [h1,  h2]
    ])
    
    # Construct the subspace Hamiltonian H_sub
    H_sub = np.array([
        [h1, h2],
        [h2, h3]
    ])
    
    # Solve the generalized eigenvalue problem: H_sub * c = E * S * c
    # eigh is mathematically robust for finding real eigenvalues of symmetric matrices
    evals, evecs = eigh(H_sub, S)
    
    # Extract the ground state (lowest eigenvalue)
    min_idx = np.argmin(evals)
    E_opt = evals[min_idx]
    c_opt = evecs[:, min_idx]
    
    # Extract the optimal alpha = c_2 / c_1
    # Handle the edge case where the initial state is already an exact eigenstate
    if np.isclose(c_opt[0], 0.0, atol=1e-12):
        alpha_opt = np.inf
    else:
        alpha_opt = c_opt[1] / c_opt[0]
        
    return E_opt, alpha_opt

# Example execution
if __name__ == "__main__":
    # Assuming h1, h2, h3 were returned from the TeNPy function
    # Using dummy values for demonstration
    h1, h2, h3 = -1.5, 2.5, -4.2 
    
    E_opt, alpha_opt = optimize_lanczos_step(h1, h2, h3)
    
    print(f"Initial Energy <H> : {h1:.8f}")
    print(f"Optimized Energy   : {E_opt:.8f}")
    print(f"Optimal alpha      : {alpha_opt:.8f}")