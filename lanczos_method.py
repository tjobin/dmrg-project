import numpy as np

def optimize_lanczos_step(
        h1: float,
        h2: float,
        h3: float,
        ) -> float :
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
        print("Warning: negative square root; setting it to 0.")
        alpha_p = (-(h3 - h1*h2)) / (2*(h1*h3-h2**2))
        alpha_m = alpha_p
    else:
        alpha_p = (-(h3 - h1*h2) + np.sqrt((h3-h1*h2)**2 - 4*(h2-h1**2)*(h1*h3-h2**2))) / (2*(h1*h3-h2**2))
        alpha_m = (-(h3 - h1*h2) - np.sqrt((h3-h1*h2)**2 - 4*(h2-h1**2)*(h1*h3-h2**2))) / (2*(h1*h3-h2**2))
    if alpha_p > 10**12:
        alpha_p = 10**12
    elif alpha_p < -10**12:
        alpha_p = -10**12
    if alpha_m > 10**12:
        alpha_m = 10**12
    elif alpha_m < -10**12:
        alpha_m = -10**12
    return alpha_p, alpha_m