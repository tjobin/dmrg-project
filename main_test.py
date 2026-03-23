import numpy as np
from lanczos_method import optimize_lanczos_step
from moments_estimator import estimate_hamiltonian_moments, get_mpo_moments_bruteforce
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
from tenpy.models.hubbard import BoseHubbardModel
from _plot import plot_E_vs_chi, plot_variance_vs_samples
from utils import get_exact_psi_and_E, EXACT_ENERGIES, test_alphas, plot_Ealpha_vs_alpha
import matplotlib.pyplot as plt

Lx, Ly = 5, 3
N = Lx * Ly  # Half-filling for bosons

## Define the model parameters for a Lx by Ly cylinder of the Bose-Hubbard model
model_params = {
        "lattice": "Square",
        "Lx": Lx,             # Number of sites along the cylinder length
        "Ly": Ly,             # Circumference of the cylinder (number of sites)
        "bc_MPS": "finite",   # Finite MPS (open boundaries for the 1D path)
        "bc_y": "cylinder",   # Periodic boundary conditions in the y-direction
        "bc_x": "open",       # Open boundary conditions in the x-direction
        "t": 1.0,               # Nearest-neighbor hopping amplitude
        "U": 4.0,               # On-site Coulomb repulsion
        "mu": 0.0,             # Chemical potential
        "conserve": "N",        # Conserve total particle number U(1)
        # "cons_Sz": "Sz",      # Conserve total spin Z U(1)
    }

model = BoseHubbardModel(model_params)
H_mpo = model.H_MPO

## Create an initial MPS in the product state |up up up ...>
product_state = [1] * model.lat.N_sites
psi = MPS.from_product_state(model.lat.mps_sites(), product_state, bc=model.lat.bc_MPS)

## Lists to store raw DMRG energies and Lanczos-optimized energies for each chi value
E_dmrg = []
E_lanczos = []

## Range of max bond dimension chi values to test
chi_maxs = [24] # , 44, 48, 52, 56, 60, 64]

Nss = [2**8, 2*9, 2**10, 2**11, 2**12]

cs = [1.0, 0.99, 0.95, 0.9, 0.85]

## Run DMRG for each bond-dimension chi
for chi_max in chi_maxs:
    dmrg_params = {
                'mixer': True,           # Adds a density matrix perturbation to escape local minima
                'max_E_err': 1.0e-10,    # Energy convergence threshold
                'trunc_params': {
                    'chi_max': chi_max,      # Maximum bond dimension (m or \chi)
                    'svd_min': 1.0e-10   # Discarded weight / singular value cutoff
                },
                'combine': True,          # Combine local tensors for the 2-site update
                'active_sites': 2
            }
    
    info = dmrg.run(psi, model, dmrg_params)
    vars_per_c = [[] for _ in range(len(cs))]
    for i, c in enumerate(cs):
        for Ns in Nss:
            Ei, psii = info['E'], info['psi_i']
            phi_1 = psii.copy()
            H_mpo.apply_naively(phi_1)

            # Lanczos step using the estimated moments from perfect sampling
            h1_est, h2_est, h3_est, var = estimate_hamiltonian_moments(
                psii,
                H_mpo,
                N_s=Ns,
                E_dmrg=Ei,
                c=c,
                filename=f'moments_chi{chi_max}',
                seed=42
                )
            print(i)
            vars_per_c[i].append(var)
            

            h1_exact, h2_exact, h3_exact = get_mpo_moments_bruteforce(psii, H_mpo)

            print(f'estimated h1 : {h1_est}, estimated h2 : {h2_est}, estimated h3 : {h3_est}')
            print(f'exact h1 : {h1_exact}, exact h2 : {h2_exact}, exact h3 : {h3_exact}')
            alpha_p, alpha_m = optimize_lanczos_step(h1_est, h2_est, h3_est)
            print(f'alpha_p : {alpha_p}, alpha_m : {alpha_m}')

            psi_alpha_p = psii.add(other=phi_1, alpha=1.0, beta=alpha_p)
            psi_alpha_m = psii.add(other=phi_1, alpha=1.0, beta=alpha_m)

            E_alpha_p = np.real(H_mpo.expectation_value(psi=psi_alpha_p))
            E_alpha_m = np.real(H_mpo.expectation_value(psi=psi_alpha_m))

            if E_alpha_p < E_alpha_m:
                E_alpha = E_alpha_p
                alpha_star = alpha_p
                psii = psi_alpha_p
            else:
                E_alpha = E_alpha_m
                alpha_star = alpha_m
                psii = psi_alpha_m

            alpha_p_exact, alpha_m_exact = optimize_lanczos_step(h1_exact, h2_exact, h3_exact)

            psi_alpha_p_exact = psii.add(other=phi_1, alpha=1.0, beta=alpha_p_exact)
            psi_alpha_m_exact = psii.add(other=phi_1, alpha=1.0, beta=alpha_m_exact)

            E_alpha_p_exact = np.real(H_mpo.expectation_value(psi=psi_alpha_p_exact))
            E_alpha_m_exact = np.real(H_mpo.expectation_value(psi=psi_alpha_m_exact))

            if E_alpha_p_exact < E_alpha_m_exact:
                E_alpha_exact = E_alpha_p_exact
                alpha_star_exact = alpha_p_exact
                psii_exact = psi_alpha_p_exact
            else:
                E_alpha_exact = E_alpha_m_exact
                alpha_star_exact = alpha_m_exact
                psii_exact = psi_alpha_m_exact

            print(f'E_alpha_est : {E_alpha}, alpha_star : {alpha_star}')
            print(f'E_alpha_exact : {E_alpha_exact}, alpha_star_exact : {alpha_star_exact}')

            alphas = np.linspace(0, 0.15, 50)
            # print(f'alpha min : {(h1_est + 2*alpha_m*h2_est + alpha_m ** 2 * h3_est) / (1 + 2 * alpha_m * h1_est + alpha_m ** 2 * h2_est)}')
            # print(f'alpha max : {(h1_est + 2*alpha_p*h2_est + alpha_p ** 2 * h3_est) / (1 + 2 * alpha_p * h1_est + alpha_p ** 2 * h2_est)}')

            # plot_Ealpha_vs_alpha(alphas, h1_est, h2_est, h3_est, psii, H_mpo)
            



            E_lanczos.append(E_alpha)

            # # Compare the raw DMRG energy and the Lanczos-optimized energy
            # print(f'DMRG energy : {Ei:.16f}')
            # print(f'Lanczos-step energy : {E_alpha:.16f}')
            print(var)
    print(vars_per_c)
    plot_variance_vs_samples(vars_per_c, Nss, cs, filename=f'variance_chi{chi_max}')


# plot_E_vs_chi(
#     chi_maxs,
#     E_dmrg,
#     E_lanczos,
#     None,
#     filename='energy_vs_chi_focus_atol=2e-3',)
