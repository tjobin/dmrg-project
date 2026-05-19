from _plot import plot_Ealpha_vs_alpha, plot_Ealpha_vs_Ns
from lanczos_method import lanczos_step_sampled_v2, get_theoretical_energy_surface
from j1j2_model import j1j2_model
from _plot import plot_dE_vs_chi, plot_rel_dE_vs_chi
from utils import get_exact_psi_and_E, EXACT_ENERGIES_J1J2_cylinder, EXACT_ENERGIES_J1J2_torus, test_alphas
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
import os
import random
import numpy as np

# OmegaConf.register_new_resolver("calc_seeds", lambda nss: [ns * 10 + 42 for ns in nss])

@hydra.main(version_base=None, config_path="conf", config_name="config")
def test(cfg: DictConfig):

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
    sub_filepath = f'J1J2_{lattice}/c={cfg.lanczos.c}'
    dmrg_geom_filepath = f'J1J2_{lattice}'

    model = j1j2_model(Lx=Lx, Ly=Ly, j1=j1, j2=j2, bc_x=bc_x, bc_y=bc_y)
    H_mpo = model.get_mpo()
    logger.info("Exact energy from exact diagonalization: %.10f Ha", E_exact)

    # Run DMRG for each bond-dimension chi
    for chi_max in cfg.dmrg.chi_maxs:
        logger.info('\n====================================== chi_max = %d ======================================\n', chi_max)  
        dmrg_filepath = f'log_dmrg/{dmrg_geom_filepath}'
        E, psi = model.run(chi_max=chi_max, dmrg_filepath=dmrg_filepath) 

        E_alpha_sampled = E 
        psi.norm = 1.0

        E_dmrg.append(E)
        El_sampled.append(E_alpha_sampled)

        alphas = np.linspace(-0.5, 0.5, 500)

        E_alphas_vs_Ns = []
        last_alpha_star = None
        last_E_alpha_star = None
        last_h1, last_h2, last_h3 = None, None, None

        for Ns, seed in zip(cfg.lanczos.Nss, cfg.lanczos.seeds):
            E_alpha_star, alpha_star, h1, h2, h3 = lanczos_step_sampled_v2(
                psi = psi,
                H = H_mpo,
                N_s = Ns,
                chi_max = chi_max,
                E_ref = E,
                c = cfg.lanczos.c,
                seed = seed,
                sampling_filepath = f'log_sampling/test/{sub_filepath}/chi={chi_max}/'
            )
            E_alphas_vs_Ns.append(E_alpha_star)
            last_alpha_star = alpha_star
            last_E_alpha_star = E_alpha_star
            last_h1, last_h2, last_h3 = h1, h2, h3
            logger.info("Ns: %d | Optimized alpha: %.4f | E_alpha: %.10f Ha", Ns, alpha_star, E_alpha_star)

        # Compute exact scan over alphas
        cache_dir = 'log_test_alphas'
        cache_filepath = f'{cache_dir}/{alphas[0]}_{alphas[-1]}_{len(alphas)}_chi{chi_max}.json'
        E_alphas = test_alphas(alphas, psi, H_mpo, cache_filepath=cache_filepath)
        
        figs_dir = f'figs/test_alphas/c={cfg.lanczos.c}/chi={chi_max}'
        os.makedirs(figs_dir, exist_ok=True)

        plot_Ealpha_vs_Ns(
            Nss=cfg.lanczos.Nss,
            E_alphas=E_alphas_vs_Ns,
            E_dmrg=E,
            E_exact=E_exact,
            figs_filename=f'{figs_dir}/chi{chi_max}_c{cfg.lanczos.c}_Ealpha_vs_Ns.png'
        )

        # Compute the theoretical curve from the moments of the last Ns step
        E_alphas_theo = get_theoretical_energy_surface(alphas, last_h1, last_h2, last_h3)

        plot_Ealpha_vs_alpha(
            alphas=alphas,
            E_alphas=E_alphas,
            E_alphas_theo=E_alphas_theo,
            alpha_star=last_alpha_star,
            E_alpha_star=last_E_alpha_star,
            E_dmrg=E,
            E_exact=E_exact,
            figs_filename=f'{figs_dir}/chi{chi_max}_Ns{cfg.lanczos.Nss[-1]}_c{cfg.lanczos.c}_trunc.png'
        )

if __name__ == "__main__":
    test()