import numpy as np
from lanczos_method import optimize_lanczos_step
from moments_estimator import estimate_hamiltonian_moments
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
    "L": 20,              # Number of sites in the 1D chain
    "t": 1.0,             # Hopping amplitude
    "U": 4.0,             # On-site repulsive interaction
    "mu": 0.0,            # Chemical potential
    "n_max": 5,           # Maximum number of bosons allowed per site (local Hilbert space cutoff)
    "conserve": None,      # Conserve total particle number 'N' (can also be 'None' or 'parity')
    "bc_MPS": "finite",   # Boundary conditions for the MPS
}

model_params = {
        "lattice": "Square",
        "Lx": 20,             # Number of sites along the cylinder length
        "Ly": 10,             # Circumference of the cylinder (number of sites)
        "bc_MPS": "finite",   # Finite MPS (open boundaries for the 1D path)
        "bc_y": "cylinder",   # Periodic boundary conditions in the y-direction
        "bc_x": "open",       # Open boundary conditions in the x-direction
        "t": 1.0,               # Nearest-neighbor hopping amplitude
        "U": 4.0,               # On-site Coulomb repulsion
        "mu": 0.0,             # Chemical potential
        "cons_N": "N",        # Conserve total particle number U(1)
        "cons_Sz": "Sz",      # Conserve total spin Z U(1)
    }

model = BoseHubbardModel(model_params)

## Create the TFI model
# model = TFIChain(model_params)

## Create an initial MPS in the product state |up up up ...>
product_state = [1] * model.lat.N_sites
psi = MPS.from_product_state(model.lat.mps_sites(), product_state, bc=model.lat.bc_MPS)

# initial_state = [4] + [0] * (model.lat.N_sites - 1)  # A state with 5 bosons on the first site and 0 elsewhere
# print(f'Initial state: {initial_state}')
# print(f'mps sites {model.lat.mps_sites()}')
# Instantiate the MPS ansatz
# psi = MPS.from_product_state(
#     model.lat.mps_sites(), 
#     initial_state, 
#     bc=model.lat.bc_MPS
# )

# Try mixing the all single-occupancy state with a random state to get a nontrivial initial guess
# psi = MPS.from_random_unitary_evolution(
#         sites=model.lat.mps_sites(),
#         chi=10,
#         p_state=initial_state,
#         bc='finite',
#         dtype=np.float64,
#         permute=True,
#         form='B',
#         chargeL=None,
#     )

## Range of max bond dimension chi values to test
chi_maxs = [8]#, 16, 32, 64, 128, 256, 512, 1024]

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
            'verbose': 1,            # Print progress output
            # 'max_sweeps': 10,         # Maximum number of DMRG sweeps
            'combine': True,          # Combine local tensors for the 2-site update
            'active_sites': 1
        }

    info = dmrg.run(psi, model, dmrg_params)
    h1, psii = info['E'], info['psi_i']
    H_mpo = model.H_MPO
    trunc_params = dmrg_params['trunc_params']

    h1_est, h2_est, h3_est = estimate_hamiltonian_moments(psii.copy(), H_mpo, N_s=200)
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
    # exact_diag = ExactDiag(model, max_size=1e12)
    # exact_diag.build_full_H_from_mpo()
    # exact_diag.full_diagonalization()
    # exact_out = exact_diag.groundstate()  # Ground state energy
    # E_exact, psi_exact = exact_out[0], exact_out[1]

    print(f'h1 : {h1:.16f}')    
    print(f'Ei : {H_mpo.expectation_value(psi=psii):.16f}')

    # Compute the Lanczos-optimized energy using the estimated moments and the optimal alpha

    # print(f'Raw optimization: {h1:.16f}')
    print(f'Optimized estimated: {E_alpha:.16f}')
    # print(f'Exact diagonalization: {E_exact:.16f}')
    # print(f'Difference between raw and optimized: {h1 - E_alpha:.16f}')

# 1. Measure the exact particle number of your DMRG state
N_dmrg = np.sum(psii.expectation_value('N'))
print(f"DMRG Total Particles: {N_dmrg:.2f}")

# 2. Check the charge sector of your Exact Diagonalization state
# Assuming your ED state was returned as `psi_exact_array`
# ed_charge = psi_exact.qtotal
# print(f"ED Ground State Charge Sector: {ed_charge}")










