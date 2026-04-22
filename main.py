from lanczos_method import lanczos_step_sampled, lanczos_step_exact
from j1j2_model import j1j2_model
from _plot import plot_dE_vs_chi, plot_rel_dE_vs_chi
from utils import get_exact_psi_and_E, EXACT_ENERGIES_J1J2_cylinder
import hydra
from omegaconf import DictConfig, OmegaConf
import json
import os
import random
import numpy as np

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    assert len(cfg.lanczos.Nss) == len(cfg.lanczos.seeds), "Length of Nss and seeds must be the same"
    
    # Set global seeds for reproducibility
    global_seed = 100
    random.seed(global_seed)
    np.random.seed(global_seed)

    print(OmegaConf.to_yaml(cfg))

    Lx = cfg.system.Lx
    Ly = cfg.system.Ly
    j1 = cfg.system.j1
    j2 = cfg.system.j2

    ## Lists to store raw DMRG energies and Lanczos-optimized energies for each chi value
    E_dmrg = []
    El_sampled = []
    # El_exact = []
    data_to_save = {}

    model = j1j2_model(Lx=Lx, Ly=Ly, j1=j1, j2=j2)
    H_mpo = model.get_mpo()

    # _, E_exact = get_exact_psi_and_E(model.model)
    E_exact = EXACT_ENERGIES_J1J2_cylinder[(Lx, Ly)]

    # print(f"Exact energy from exact diagonalization: {E_exact:.10f} Ha")
    # Run DMRG for each bond-dimension chi
    for chi_max in cfg.chi_maxs:
        print(f'\n====================================== chi_max = {chi_max} ======================================\n')  
        E, psi = model.run(chi_max=chi_max) 
        E_alpha_sampled = E 
        psi_alpha_sampled = psi.copy() 

        for Ns, seed in zip(cfg.lanczos.Nss, cfg.lanczos.seeds):            # Lanczos step using the estimated moments from perfect sampling
            E_alpha_sampled, psi_alpha_sampled, alpha_star_sampled = lanczos_step_sampled(
                psi=psi_alpha_sampled,
                H=H_mpo,
                N_s=Ns,
                chi_max=chi_max,
                E_ref=E_alpha_sampled,
                c=cfg.lanczos.c,
                seed=seed,
                json_filename=f'sampling_chi{chi_max}_Ns{Ns}_seed{seed}_c{cfg.lanczos.c}'
            )
            psi_alpha_sampled.canonical_form()
            psi_alpha_sampled.norm = 1.0

        print(f'Exact energy: {E_exact:.10f} Ha')
        print(f"DMRG energy: {E:.10f} Ha")
        print(f"Lanczos (sampled): E = {E_alpha_sampled:.10f} Ha, alpha = {alpha_star_sampled:.4f}")

        E_dmrg.append(E)
        El_sampled.append(E_alpha_sampled)

        rel_dE = (E_alpha_sampled - E_exact + 1e-12) / (E - E_exact + 1e-12)
        dE = E_alpha_sampled - E_exact
        data_to_save[str(chi_max)] = {
            "E_exact": float(E_exact),
            "E_dmrg": float(E),
            "El_sampled": float(E_alpha_sampled),
            "rel_dE": float(rel_dE),
            "dE": float(dE)
        }
        
    # Save plotted data to a JSON file
    os.makedirs(f'log_lanczos/J1J2_{Lx}x{Ly}/cleaned', exist_ok=True)
    json_filename = f'log_lanczos/J1J2_{Lx}x{Ly}/cleaned/data_chi{cfg.chi_maxs[0]}-{cfg.chi_maxs[-1]}_Ns{Ns}_seed{seed}_c{cfg.lanczos.c}.json'
    with open(json_filename, 'w') as f:
        json.dump(data_to_save, f, indent=4)

    plot_rel_dE_vs_chi(
        cfg.chi_maxs,
        E_exact,
        E_dmrg,
        El_sampled,
        None,
        dim=[Lx, Ly],
        filename=f'J1J2_{Lx}x{Ly}/cleaned/rel_dE_vs_chi{cfg.chi_maxs[0]}-{cfg.chi_maxs[-1]}_Ns{cfg.lanczos.Nss}_c{cfg.lanczos.c}.png'
    )
    plot_dE_vs_chi(
        cfg.chi_maxs,
        E_exact,
        E_dmrg,
        El_sampled,
        None,
        dim=[Lx, Ly],
        filename=f'J1J2_{Lx}x{Ly}/cleaned/dE_vs_chi{cfg.chi_maxs[0]}-{cfg.chi_maxs[-1]}_Ns{cfg.lanczos.Nss}_c{cfg.lanczos.c}.png'
    )

if __name__ == "__main__":
    main()