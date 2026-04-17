import numpy as np
from lanczos_method import lanczos_step_sampled, lanczos_step_exact
from j1j2_model import j1j2_model
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
from tenpy.models.hubbard import BoseHubbardModel
from _plot import plot_E_vs_chi, plot_variance_vs_samples, plot_Ealpha_vs_alpha
from utils import get_exact_psi_and_E, EXACT_ENERGIES_J1J2_cylinder, test_alphas
import hydra
from omegaconf import DictConfig, OmegaConf


Lx, Ly = 4, 4

def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))

    Lx = cfg.system.Lx
    Ly = cfg.system.Ly
    j1 = cfg.system.j1
    j2 = cfg.system.j2

## Lists to store raw DMRG energies and Lanczos-optimized energies for each chi value
E_dmrg = []
El_sampled = []
El_exact = []

## Range of max bond dimension chi values to test
chi_maxs = [40] # , 44, 48, 52, 56, 60, 64]

Nss = [256, 256, 512, 1024]

model = j1j2_model(Lx=Lx, Ly=Ly, j1=1.0, j2=0.5)
H_mpo = model.get_mpo()

_, E_exact = get_exact_psi_and_E(model.model)
# E_exact = EXACT_ENERGIES_J1J2_cylinder[(Lx, Ly)]

print(f"Exact energy from exact diagonalization: {E_exact:.10f} Ha")
# Run DMRG for each bond-dimension chi
# for chi_max in cfg.chi_maxs:
#     print(f'\n====================================== chi_max = {chi_max} ======================================\n')  
#     E, psi = model.run(chi_max=chi_max)  
#     psi_alpha_sampled = psi.copy() 
#     psi_alpha_exact = psi.copy()

#     for Ns in cfg.lanczos.Nss:        
#         # Lanczos step using the estimated moments from perfect sampling
#         E_alpha_sampled, psi_alpha_sampled, alpha_star_sampled = lanczos_step_sampled(
#             psi=psi_alpha_sampled,
#             H=H_mpo,
#             N_s=Ns,
#             chi_max=chi_max,
#             seed=cfg.lanczos.seed,
#             filename=f'moments_chi{chi_max}_Ns{Ns}'
#         )
#         # E_alpha_exact, psi_alpha_exact, alpha_star_exact = lanczos_step_exact(
#         #     psi=psi_alpha_exact,
#         #     H=H_mpo
#         # )

#         psi_alpha_sampled.canonical_form()
#         psi_alpha_sampled.norm = 1.0
#         # psi_alpha_exact.canonical_form()
#         # psi_alpha_exact.norm = 1.0
#     print(f'Exact energy: {E_exact:.10f} Ha')
#     print(f"DMRG energy: {E:.10f} Ha")
#     print(f"Lanczos energy (sampled): {E_alpha_sampled:.10f} Ha")
#     # print(f"Lanczos energy (exact): {E_alpha_exact:.10f} Ha")
#     E_dmrg.append(E)
#     El_sampled.append(E_alpha_sampled)
#     # El_exact.append(E_alpha_exact)

# plot_E_vs_chi(
#     chi_maxs,
#     E_exact,
#     E_dmrg,
#     El_sampled,
#     None,
# )


