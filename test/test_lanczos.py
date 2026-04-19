import numpy as np
import pytest
from lanczos_method import get_optimized_alphas

def test_get_optimized_alphas_negative_sqrt():
    # Provide h1, h2, h3 values that force the discriminant to be <= 0
    # Discriminant: (h3-h1*h2)**2 - 4*(h2-h1**2)*(h1*h3-h2**2)
    h1, h2, h3 = 1.0, 1.0, 1.0 
    
    alpha_p, alpha_m = get_optimized_alphas(h1, h2, h3)
    
    # Assert that the function successfully handled the negative root and returned identical alphas
    assert alpha_p == alpha_m
    assert not np.isnan(alpha_p)

def test_get_optimized_alphas_bounds():
    # Provide moments that would result in an extreme alpha
    h1, h2, h3 = 0.0, 0.0, 1e15 
    alpha_p, alpha_m = get_optimized_alphas(h1, h2, h3)
    
    assert alpha_p <= 10**12
    assert alpha_m >= -10**12