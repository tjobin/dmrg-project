import numpy as np
from lanczos_method import optimize_lanczos_iterative, optimize_lanczos_step, apply_lanczos_step
from Hmoments import get_moments_brute_force, estimate_hamiltonian_moments
from tenpy.algorithms import dmrg
from tenpy.models.tf_ising import TFIChain
from tenpy.networks.mps import MPS
from tenpy.models.spins import SpinModel


Nsites = 200
J = 1.0
g = 1.0
chi_max = 10

# params_2d = {
#     'lattice': 'Square',
#     'Lx': 10,               # Length of the cylinder
#     'Ly': 10,                # Circumference of the cylinder (critical for complexity)
#     'bc_MPS': 'finite',     # Finite MPS
#     'bc_x': 'open',         # Open boundaries along the cylinder length
#     'bc_y': 'periodic',     # Periodic boundaries wrap the 2D plane into a cylinder
#     'Jx': 1.0, 'Jy': 1.0, 'Jz': 1.0,
#     'conserve': 'Sz'
# }

# model = SpinModel(params_2d)

model_params = {
    'L': Nsites,
    'J': J,
    'g': g,
    'bc_MPS': 'finite',
    'conserve': 'parity',
}
model = TFIChain(model_params)

product_state = ['up'] * model.lat.N_sites
psi = MPS.from_product_state(model.lat.mps_sites(), product_state, bc=model.lat.bc_MPS)

dmrg_params = {
        'mixer': True,           # Adds a density matrix perturbation to escape local minima
        'max_E_err': 1.0e-10,    # Energy convergence threshold
        'trunc_params': {
            'chi_max': chi_max,      # Maximum bond dimension (m or \chi)
            'svd_min': 1.0e-10   # Discarded weight / singular value cutoff
        },
        # 'verbose': 2,            # Print progress output
        'combine': True,          # Combine local tensors for the 2-site update
        'active_sites': 2
    }

info = dmrg.run(psi, model, dmrg_params)
h1, psi_i = info['E'], info['psi_i']
H_mpo = model.H_MPO
trunc_params = dmrg_params['trunc_params']

h2, h3 = get_moments_brute_force(model, psi_i, h1)

h1_est, h2_est, h3_est = estimate_hamiltonian_moments(psi_i, H_mpo, num_samples=200)


# psi_alpha, E_alpha, alpha_star = apply_lanczos_step(psi_i, H_mpo, trunc_params)
# psi_alpha2, E_alpha2, alpha_star2 = optimize_lanczos_iterative(psi_i, H_mpo, trunc_params)
_, alpha_star3 = optimize_lanczos_step(h1, h2, h3)
print(f'Estimated moments: h1={h1_est:.16f}, h2={h2_est:.16f}, h3={h3_est:.16f}')
_, alpha_star4 = optimize_lanczos_step(h1_est, h2_est, h3_est)

E_alpha3 = h1 + 2 * alpha_star3 * h2 + (alpha_star3 ** 2) * h3
E_alpha4 = h1_est + 2 * alpha_star4 * h2_est + (alpha_star4 ** 2) * h3_est

# E_opt, alpha_opt = optimize_lanczos_step(h1, h2, h3)

print(f'Raw optimization: {h1:.8f}')
# print(f'Optimized optimization: {E_opt:.8f})')
# print(f'Optimized optimization 2: {E_alpha:.8f})')
print(f'Optimized brute force: {E_alpha3:.8f}')
print(f'Optimized estimated: {E_alpha4:.8f}')

# print(f'Iterative optimization: {E_alpha2:.8f})')








