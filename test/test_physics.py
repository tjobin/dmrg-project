import pytest
import numpy as np
from j1j2_model import j1j2_model
from utils import EXACT_ENERGIES

@pytest.mark.parametrize("Lx, Ly", [(3, 2), (4, 2)])
def test_dmrg_against_exact_diagonalization(Lx, Ly):
    # Fetch exact energy from your utils dict
    E_exact = EXACT_ENERGIES[(Lx, Ly)]
    
    # Initialize the model using the same parameters you expect to use
    model = j1j2_model(Lx=Lx, Ly=Ly, j1=1.0, j2=0.5, chi_max=100)
    
    # Run the DMRG
    E_dmrg, _ = model.run()
    
    # pytest.approx handles floating point comparisons automatically
    assert E_dmrg == pytest.approx(E_exact, rel=1e-5)