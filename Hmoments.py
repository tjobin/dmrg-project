import numpy as np
from tenpy import models
from tenpy.networks.mps import MPS
import numpy as np
from tenpy.networks.mpo import MPOEnvironment
from tenpy.networks.mpo import MPO

def get_moments_brute_force(model:models, psi_i:MPS, h1:float):
    """
    Runs DMRG for the Transverse Field Ising Model and computes 
    <H>, <H^2>, and <H^3>.
    """


    # 5. Calculate <H^2>
    # MPO.variance calculates <H^2> - <H>^2
    variance = model.H_MPO.variance(psi_i)
    h2 = variance + (h1 ** 2)

    # 6. Calculate <H^3>
    # Apply H to psi to obtain a new state |phi> = H |psi>
    phi = psi_i.copy()
    model.H_MPO.apply_naively(phi) 
    
    # <psi | H^3 | psi> is equivalent to <phi | H | phi>
    # MPO.expectation_value(phi) returns <phi|H|phi> / <phi|phi>
    # phi.norm() returns the scalar norm sqrt(<phi|phi>)
    h3 = model.H_MPO.expectation_value(phi) * (phi.norm ** 2)

    return h2, h3

def estimate_hamiltonian_moments(psi:MPS, mpo_H:MPO, num_samples=1000):
    h1_estimates = []
    h2_estimates = []
    h3_estimates = []
    
    sites = psi.sites
    
    for _ in range(num_samples):
        # Pass complex_amplitude=True to get the exact overlap <s|psi>
        sigmas, _ = psi.sample_measurements(complex_amplitude=True)
        mps_s = MPS.from_product_state(sites, sigmas, bc=psi.bc)


        # If the overlap is identically zero, the state is physically impossible
        overlap_s = mps_s.overlap(psi)
        if overlap_s == 0.0:
            continue
            
        mps_H1_s = mps_s.copy()  # Make a copy to avoid in-place modifications
        mps_H2_s = mps_s.copy()  # Make a copy to avoid in-place modifications
        mps_H3_s = mps_s.copy()  # Make a copy to avoid in-place modifications

        mpo_H.apply_naively(mps_H1_s) 

        mpo_H.apply_naively(mps_H2_s)
        mpo_H.apply_naively(mps_H2_s)    
   
        mpo_H.apply_naively(mps_H3_s)  
        mpo_H.apply_naively(mps_H3_s)  
        mpo_H.apply_naively(mps_H3_s)
        
        num_1 = np.conj(psi.overlap(mps_H1_s))
        num_2 = np.conj(psi.overlap(mps_H2_s))
        num_3 = np.conj(psi.overlap(mps_H3_s))

        # print(f'<s|H|psi> = {num_1}, <s|H^2|psi> = {num_2}, <s|H^3|psi> = {num_3}, <s|psi> = {overlap_s}')
        
        # Calculate local moments using the natively returned overlap
        e1_local = num_1 / overlap_s
        e2_local = num_2 / overlap_s
        e3_local = num_3 / overlap_s
        
        h1_estimates.append(e1_local)
        h2_estimates.append(e2_local)
        h3_estimates.append(e3_local)
        
    return np.real(np.mean(h1_estimates)), np.real(np.mean(h2_estimates)), np.real(np.mean(h3_estimates))