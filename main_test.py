import numpy as np
from lanczos_method import optimize_lanczos_step
from Hmoments import get_moments_brute_force, estimate_hamiltonian_moments
from tenpy.algorithms import dmrg
from tenpy.models.tf_ising import TFIChain
from tenpy.networks.mps import MPS
from tenpy.models.spins import SpinModel
from tenpy.models.hubbard import BoseHubbardModel
from _plot import plot_E_vs_chi
from modified_DMRG import ExactStateDMRGEngine
from tenpy.algorithms.exact_diag import ExactDiag

## Parameters for the TFI model
Nsites = 200
J = 1.0
g = 1.0

# model_params = {
#     'L': Nsites,
#     'J': J,
#     'g': g,
#     'bc_MPS': 'finite',
#     'conserve': 'parity',
# }

model_params = {
    "L": 5,              # Number of sites in the 1D chain
    "t": 1.0,             # Hopping amplitude
    "U": 4.0,             # On-site repulsive interaction
    "mu": 0.0,            # Chemical potential
    "n_max": 5,           # Maximum number of bosons allowed per site (local Hilbert space cutoff)
    "conserve": "N",      # Conserve total particle number 'N' (can also be 'None' or 'parity')
    "bc_MPS": "finite",   # Boundary conditions for the MPS
}

model = BoseHubbardModel(model_params)

## Create the TFI model
# model = TFIChain(model_params)

## Create an initial MPS in the product state |up up up ...>
# product_state = ['up'] * model.lat.N_sites
# psi = MPS.from_product_state(model.lat.mps_sites(), product_state, bc=model.lat.bc_MPS)

initial_state = [1] * model_params["L"]

# Instantiate the MPS ansatz
psi = MPS.from_product_state(
    model.lat.mps_sites(), 
    initial_state, 
    bc=model.lat.bc_MPS
)
## Range of max bond dimension chi values to test
chi_maxs = [20]#, 16, 32, 64, 128, 256, 512, 1024]

## Lists to store raw DMRG energies and Lanczos-optimized energies for each chi value
E = []
E_alpha = []


for chi_max in chi_maxs:
    dmrg_params = {
            'mixer': True,           # Adds a density matrix perturbation to escape local minima
            'max_E_err': 1.0e-10,    # Energy convergence threshold
            'trunc_params': {
                'chi_max': chi_max,      # Maximum bond dimension (m or \chi)
                'svd_min': 1.0e-10   # Discarded weight / singular value cutoff
            },
            # 'verbose': 2,            # Print progress output
            'combine': True,          # Combine local tensors for the 2-site update
            'active_sites': 1
        }
    
    engine = ExactStateDMRGEngine(psi, model, dmrg_params)

    # 3. Run the optimization
    h1, psii = engine.run()

    # psii = engine.exact_psi
    # h1 = engine.exact_h1

    # info = dmrg.run(psi, model, dmrg_params)
    # h1, psii = info['E'], info['psi_i']
    H_mpo = model.H_MPO
    trunc_params = dmrg_params['trunc_params']

    # dmrg_params_1site = {
    #     'mixer': False,          # Strictly no perturbative noise
    #     'max_E_err': 1.0e-12,    # Tight convergence for the exact energy
    #     'max_sweeps': 20,         # Usually converges in 2-4 sweeps
    #     'trunc_params': {
    #         'chi_max': 300,
    #         'svd_min': 0.0       # Zero truncation
    #     },
    #     'active_sites': 1        # Single-site update (The critical change)
    # }

    # # Run the polishing phase on the exact same psi
    # info_final = dmrg.run(psi, model, dmrg_params_1site)
    # h1_final, psii_final = info_final['E'], info_final['psi_i']

    h1_est, h2_est, h3_est = estimate_hamiltonian_moments(psii.copy(), H_mpo, 5)
    _, alpha_star = optimize_lanczos_step(h1_est, h2_est, h3_est)
    print(f'alpha_star from raw moments: {alpha_star:.6f}')


    phi_1 = psii.copy()
    H_mpo.apply_naively(phi_1)

    # alphas = np.linspace(0, 10, 100)
    # E_alpha_values = []
    # for alpha in alphas:
    #     psi_alpha = psii.copy()
    #     psi_alpha.add(other=phi_1, alpha=1.0, beta=alpha)
    #     E_alpha_test = np.real(H_mpo.expectation_value(psi=psi_alpha))
    #     E_alpha_values.append(np.real(E_alpha_test))
    #     overlap = phi_1.overlap(psi_alpha)
    #     print(f'Overlap: {overlap:.16f}')
    #     print(f'E_alpha: {E_alpha_test:.16f} for alpha = {alpha:.6f}')

    psi_alpha = psii.copy()
    psi_alpha.add(other=phi_1, alpha=1.0, beta=alpha_star)
    E_alpha = np.real(H_mpo.expectation_value(psi=psi_alpha))

    # Exact energy from exact diagonalization for comparison
    exact_diag = ExactDiag(model, max_size=1e8)
    exact_diag.build_full_H_from_mpo()
    exact_diag.full_diagonalization()
    E_exact = exact_diag.groundstate()[0]  # Ground state energy


    print(f'h1 : {h1:.16f}')    
    print(f'Ei : {H_mpo.expectation_value(psi=psii):.16f}')

    # Compute the Lanczos-optimized energy using the estimated moments and the optimal alpha

    # print(f'Raw optimization: {h1:.16f}')
    print(f'Optimized estimated: {E_alpha:.16f}')
    print(f'Exact diagonalization: {E_exact:.16f}')
    # print(f'Difference between raw and optimized: {h1 - E_alpha:.16f}')












