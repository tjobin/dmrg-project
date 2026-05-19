from lanczos_method import lanczos_step_sampled, lanczos_step_exact
from j1j2_model import j1j2_model
from _plot import plot_dE_vs_chi, plot_rel_dE_vs_chi
from utils import get_exact_psi_and_E, EXACT_ENERGIES_J1J2_cylinder, EXACT_ENERGIES_J1J2_torus
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
import json
import os
import random
import numpy as np

# OmegaConf.register_new_resolver("calc_seeds", lambda nss: [ns * 10 + 42 for ns in nss])

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    logger = logging.getLogger(__name__)

    assert len(cfg.lanczos.Nss) == len(cfg.lanczos.seeds), "Length of Nss and seeds must be the same"
    
    # Set global seeds for reproducibility
    global_seed = 100
    random.seed(global_seed)
    np.random.seed(global_seed)

    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    Lx = cfg.system.Lx
    Ly = cfg.system.Ly
    j1 = cfg.system.j1
    j2 = cfg.system.j2
    bc_x = cfg.system.bc_x
    bc_y = cfg.system.bc_y

    if bc_x == 'periodic' and bc_y == 'periodic':
        lattice = f'square_{Lx}x{Ly}_torus'
        E_exact = EXACT_ENERGIES_J1J2_torus[(Lx, Ly)]
    elif bc_x == 'periodic' and bc_y == 'open':
        lattice = f'square_{Lx}x{Ly}_cylinder'
        E_exact = EXACT_ENERGIES_J1J2_cylinder[(Lx, Ly)]
    else:
        raise ValueError("Unsupported boundary condition combination. Supported: (periodic, periodic) and (periodic, open).")

    ## Lists to store raw DMRG energies and Lanczos-optimized energies for each chi value
    E_dmrg = []
    El_sampled = []
    data_to_save = {}
    sub_filepath = f'J1J2_{lattice}/c={cfg.lanczos.c}'
    dmrg_geom_filepath = f'J1J2_{lattice}'
    dmrg_energies_summary = {}

    model = j1j2_model(Lx=Lx, Ly=Ly, j1=j1, j2=j2, bc_x=bc_x, bc_y=bc_y)
    H_mpo = model.get_mpo()
    logger.info("Exact energy from exact diagonalization: %.10f Ha", E_exact)

    # Run DMRG for each bond-dimension chi
    for chi_max in cfg.dmrg.chi_maxs:
        logger.info('\n====================================== chi_max = %d ======================================\n', chi_max)
        dmrg_filepath = f'log_dmrg/{dmrg_geom_filepath}'
        E, psi = model.run(chi_max=chi_max, dmrg_filepath=dmrg_filepath) 
        dmrg_energies_summary[str(chi_max)] = E

        E_alpha_sampled = E 
        psi.norm = 1.0
        psi_alpha_sampled = psi.copy() 
        logger.info('Before Lanczos step, MPS bond dimension : %s', psi_alpha_sampled.chi)

        for Ns, seed in zip(cfg.lanczos.Nss, cfg.lanczos.seeds):            # Lanczos step using the estimated moments from perfect sampling
            E_alpha_sampled, alpha_star_sampled = lanczos_step_sampled(
                psi=psi_alpha_sampled,
                H=H_mpo,
                N_s=Ns,
                chi_max=chi_max,
                E_ref=E_alpha_sampled,
                c=cfg.lanczos.c,
                seed=seed,
                sampling_filepath=f'log_sampling/{sub_filepath}/'
            )
            logger.info('Lanczos step with N_s=%d samples and MPS bond dimension : %s', Ns, psi_alpha_sampled.chi)
        logger.info("DMRG energy: %.10f Ha", E)
        logger.info("Lanczos (sampled): E = %.10f Ha, alpha = %.4f", E_alpha_sampled, alpha_star_sampled)

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

    # ---------------------------------------------------------
    # Final Data Saving & Plotting
    # ---------------------------------------------------------
    
    # Save an aggregated summary of DMRG energies
    with open(f'log_dmrg/{dmrg_geom_filepath}/dmrg_energies_summary.json', 'w') as f:
        json.dump(dmrg_energies_summary, f, indent=4)

    # Construct a common suffix for all outputs based on the configuration params
    run_suffix = f"chi{cfg.dmrg.chi_maxs[0]}-{cfg.dmrg.chi_maxs[-1]}_Ns{cfg.lanczos.Nss}_c{cfg.lanczos.c}_canon"

    # Save plotted data to a JSON file
    lanczos_dir = f'log_lanczos/{sub_filepath}'
    os.makedirs(lanczos_dir, exist_ok=True)
    with open(f'{lanczos_dir}/data_{run_suffix}.json', 'w') as f:
        json.dump(data_to_save, f, indent=4)
    logger.info("Saved Lanczos data to %s/data_%s.json", lanczos_dir, run_suffix)

    # Create plots
    figs_dir = f'figs/{sub_filepath}'
    os.makedirs(figs_dir, exist_ok=True)
    
    plot_kwargs = {"chi": cfg.dmrg.chi_maxs, "E_exact": E_exact, "E_dmrg": E_dmrg, "El_alpha": El_sampled, "El_exact": None, "dim": [Lx, Ly]}
    
    plot_rel_dE_vs_chi(**plot_kwargs, figs_filename=f'{figs_dir}/rel_dE_vs_{run_suffix}.png')
    plot_dE_vs_chi(**plot_kwargs, figs_filename=f'{figs_dir}/dE_vs_{run_suffix}.png')
    logger.info("Saved figures to %s", figs_dir)

if __name__ == "__main__":
    main()