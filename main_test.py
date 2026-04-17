import numpy as np
from lanczos_method import lanczos_step_sampled, lanczos_step_exact
from j1j2_model import j1j2_model
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
from tenpy.models.hubbard import BoseHubbardModel
from _plot import plot_E_vs_chi, plot_variance_vs_samples, plot_Ealpha_vs_alpha
from utils import get_exact_psi_and_E, EXACT_ENERGIES, test_alphas


Lx, Ly = 5, 5

## Define the model parameters for a Lx by Ly cylinder of the Bose-Hubbard model

## Lists to store raw DMRG energies and Lanczos-optimized energies for each chi value

E_dmrg = []
El_sampled = []
El_exact = []

## Range of max bond dimension chi values to test
chi_maxs = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40] # , 44, 48, 52, 56, 60, 64]

Nss = [256, 256, 512, 1024]

## Run DMRG for each bond-dimension chi
for chi_max in chi_maxs:
    print(f'\n====================================== chi_max = {chi_max} ======================================\n')  

    model = j1j2_model(Lx=Lx, Ly=Ly, j1=1.0, j2=0.5, chi_max=chi_max)
    H_mpo = model.get_mpo()
    E, psi = model.run()  
    psi_alpha_sampled = psi.copy() 
    psi_alpha_exact = psi.copy()

    for Ns in Nss:        
        # Lanczos step using the estimated moments from perfect sampling
        E_alpha_sampled, psi_alpha_sampled, alpha_star_sampled = lanczos_step_sampled(
            psi=psi_alpha_sampled,
            H=H_mpo,
            N_s=Ns,
            chi_max=chi_max,
            seed=42,
            filename=f'moments_chi{chi_max}_Ns{Ns}'
        )
        

        E_alpha_exact, psi_alpha_exact, alpha_star_exact = lanczos_step_exact(
            psi=psi_alpha_exact,
            H=H_mpo
        )

        psi_alpha_sampled.canonical_form()
        psi_alpha_sampled.norm = 1.0
        psi_alpha_exact.canonical_form()
        psi_alpha_exact.norm = 1.0

    E_dmrg.append(E)
    El_sampled.append(E_alpha_sampled)
    El_exact.append(E_alpha_exact)

plot_E_vs_chi(
    chi_maxs,
    E_dmrg,
    El_sampled,
    El_exact,
)