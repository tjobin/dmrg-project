import numpy as np
from lanczos_method import optimize_lanczos_step
from moments_estimator import estimate_hamiltonian_moments
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
from tenpy.models.hubbard import BoseHubbardModel
from _plot import plot_E_vs_chi
from utils import get_exact_psi_and_E, EXACT_ENERGIES

Lx, Ly = 4, 3
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
        "cons_N": "N",        # Conserve total particle number U(1)
        # "cons_Sz": "Sz",      # Conserve total spin Z U(1)
    }

model = BoseHubbardModel(model_params)
H_mpo = model.H_MPO

## Get exact ground state energy
if (Lx, Ly) in EXACT_ENERGIES:
    E_exact = EXACT_ENERGIES[(Lx, Ly)]
else:
    _, E_exact = get_exact_psi_and_E(model, charge_sector=[N], max_size=1e15)


## Create an initial MPS in the product state |up up up ...>
product_state = [1] * model.lat.N_sites
psi = MPS.from_product_state(model.lat.mps_sites(), product_state, bc=model.lat.bc_MPS)

## Lists to store raw DMRG energies and Lanczos-optimized energies for each chi value
E_dmrg = []
E_lanczos = []

## Range of max bond dimension chi values to test
chi_maxs = [12, 16, 20, 24, 28, 32]

## Run DMRG for each bond-dimension chi
for chi_max in chi_maxs:
    dmrg_params = {
            'mixer': True,           # Adds a density matrix perturbation to escape local minima
            'max_E_err': 1.0e-10,    # Energy convergence threshold
            'trunc_params': {
                'chi_max': chi_max,      # Maximum bond dimension (m or \chi)
                'svd_min': 1.0e-10   # Discarded weight / singular value cutoff
            },
            'verbose': 1,            # Print progress output
            'combine': True,          # Combine local tensors for the 2-site update
            'active_sites': 2
        }

    info = dmrg.run(psi, model, dmrg_params)
    Ei, psii = info['E'], info['psi_i']
    trunc_params = dmrg_params['trunc_params']

    ## Lanczos step using the estimated moments from perfect sampling
    h1_est, h2_est, h3_est = estimate_hamiltonian_moments(
        psii,
        H_mpo,
        N_s=256,
        atol=2e-2,
        filename=f'moments_chi{chi_max}',
        seed=42
        )
    
    _, alpha_star = optimize_lanczos_step(h1_est, h2_est, h3_est)

    phi_1 = psii.copy()
    H_mpo.apply_naively(phi_1)

    psi_alpha = psii.add(other=phi_1, alpha=1.0, beta=alpha_star)
    E_alpha = np.real(H_mpo.expectation_value(psi=psi_alpha))

    E_lanczos.append(E_alpha)
    E_dmrg.append(Ei)
    # Use variance as an indicator of how close we are to the true ground state
    print(f'Variance {H_mpo.variance(psii):.10e}')

    # Compare the raw DMRG energy and the Lanczos-optimized energy
    print(f'Exact energy : {E_exact:.16f}')
    print(f'DMRG energy : {H_mpo.expectation_value(psi=psii):.16f}')
    print(f'Lanczos-step energy : {E_alpha:.16f}')


plot_E_vs_chi(
    chi_maxs,
    E_dmrg,
    E_lanczos,
    E_exact,
    filename='energy_vs_chi_focus1',)


