import numpy as np
from tenpy.models.tf_ising import TFIChain
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg

def evaluate_tfim_moments(L=10, J=1.0, g=1.0, chi_max=50):
    """
    Runs DMRG for the Transverse Field Ising Model and computes 
    <H>, <H^2>, and <H^3>.
    """
    # 1. Initialize the TFIM model
    model_params = {
        'L': L,
        'J': J,
        'g': g,
        'bc_MPS': 'finite',
        'conserve': 'parity'
    }
    model = TFIChain(model_params)

    # 2. Set up the initial Matrix Product State (MPS)
    # Using a fully polarized state in the z-basis
    product_state = ["up"] * L
    psi = MPS.from_product_state(model.lat.mps_sites(), product_state, bc=model.lat.bc_MPS)

    # 3. Configure and run the DMRG algorithm
    dmrg_params = {
        'mixer': True,
        'trunc_params': {
            'chi_max': chi_max,
            'svd_min': 1.e-10
        },
        'max_E_err': 1.e-8,
        'max_sweeps': 10,
    }
    
    engine = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
    E_dmrg, psi = engine.run()

    # 4. Calculate <H>
    # model.H is the Matrix Product Operator (MPO) of the Hamiltonian
    H1 = model.H_MPO.expectation_value(psi)

    # 5. Calculate <H^2>
    # MPO.variance calculates <H^2> - <H>^2
    variance = model.H_MPO.variance(psi)
    H2 = variance + (H1 ** 2)

    # 6. Calculate <H^3>
    # Apply H to psi to obtain a new state |phi> = H |psi>
    phi = psi.copy()
    model.H_MPO.apply_naively(phi) 
    
    # <psi | H^3 | psi> is equivalent to <phi | H | phi>
    # MPO.expectation_value(phi) returns <phi|H|phi> / <phi|phi>
    # phi.norm() returns the scalar norm sqrt(<phi|phi>)
    H3 = model.H_MPO.expectation_value(phi) * (phi.norm ** 2)

    return H1, H2, H3

# Example execution
if __name__ == "__main__":
    h1, h2, h3 = evaluate_tfim_moments(L=10, J=1.0, g=1.5, chi_max=50)
    print(f"<H>   = {h1:.8f}")
    print(f"<H^2> = {h2:.8f}")
    print(f"<H^3> = {h3:.8f}")