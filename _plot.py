import matplotlib.pyplot as plt
import numpy as np
import os

def plot_rel_dE_vs_chi(
        chi: list[int],
        E_exact: float,
        E_dmrg: list[float],
        El_alpha: list[float],
        El_exact: list[float] | None = None,
        dim: list[int] = [1, 1],
        figs_filename: str = 'energy_vs_chi'
        ) -> None:
    """
    Plots the energy improvement from the Lanczos step relative to the DMRG energy as a function of chi.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(chi, (np.array(El_alpha) - E_exact + 1e-12) / (np.array(E_dmrg) - E_exact + 1e-12), marker='o', linestyle='-.', color='tab:blue', label='DMRG + Lanczos (Sampled)')
    
    if El_exact is not None:
        ax.plot(chi, (np.array(El_exact) - E_exact + 1e-12) / (np.array(E_dmrg) - E_exact + 1e-12), marker='s', linestyle='-.', color='k', label='DMRG + Lanczos (Exact)')
        
    ax.set_title(f'Energy vs. Chi (Lx={dim[0]}, Ly={dim[1]})')
    ax.set_xlabel('Chi')
    ax.set_ylabel(r'Relative energy diff $\Delta E_l / \Delta E$')
    ax.set_ylim(0, 1.05)
    ax.grid()
    ax.legend()
    
    fig.savefig(f'{figs_filename}', bbox_inches='tight')
    plt.close(fig)

def plot_dE_vs_chi(
        chi: list[int],
        E_exact: float,
        E_dmrg: list[float],
        El_alpha: list[float],
        El_exact: list[float] | None = None,
        dim: list[int] = [1, 1],
        figs_filename: str = 'energy_vs_chi'
        ) -> None:
    """
    Plots the energy improvement from the Lanczos step relative to the DMRG energy as a function of chi.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(chi, np.array(E_dmrg) - E_exact, marker='s', linestyle='--', color='tab:blue', label='DMRG')
    ax.plot(chi, np.array(El_alpha) - E_exact, marker='o', linestyle='-.', color='tab:orange', label='DMRG + Lanczos (Sampled)')
    
    if El_exact is not None:
        ax.plot(chi, np.array(El_exact) - E_exact, marker='s', linestyle='-.', color='k', label='DMRG + Lanczos (Exact)')
        
    ax.set_title(f'Energy vs. Chi (Lx={dim[0]}, Ly={dim[1]})')
    ax.set_xlabel('Chi')
    ax.set_ylabel(r'Energy diff  $\Delta E_l$ (Ha)')
    ax.grid()
    ax.legend()
    fig.savefig(f'{figs_filename}', bbox_inches='tight')
    plt.close(fig)

def plot_variance_vs_samples(
        vars_per_c: list,
        samples: list,
        cs: list,
        filename: str = 'variance_vs_samples'
    ):
    fig, ax = plt.subplots()
    for i, vars in enumerate(vars_per_c):
        ax.plot(samples, vars, marker='o', linestyle='--', label=f'c={cs[i]}')
    ax.legend()
    fig.savefig(f'figs/{filename}.png', bbox_inches='tight')
    plt.close(fig)

def plot_Ealpha_vs_alpha(
        alphas,
        E_alphas,
        alpha_star,
        E_alpha_star,
        E_dmrg: float,
        E_exact: float,
        figs_filename: str,
        E_alphas_theo=None
    ):
    """
    Plots the exact scanned energies against alpha, marking the optimal estimated (alpha*, E*).
    """
    fig, ax = plt.subplots()
    ax.plot(alphas, E_alphas, linestyle='-', color='tab:blue', label='Exact Lanczos', alpha=0.7)
    if E_alphas_theo is not None:
        ax.plot(alphas, E_alphas_theo, linestyle='--', color='tab:green', label='Theoretical (Sampled Moments)')
    ax.plot(alpha_star, E_alpha_star, linestyle='None', marker='o', color='tab:orange', label='Estimated Minimum')
    ax.axhline(E_dmrg, color='k', linestyle=':', label='DMRG Energy')
    ax.axhline(E_exact, color='r', linestyle=':', label='Exact Energy')
    ax.set_ylim(E_exact - 0.01, E_dmrg + 1)
    ax.legend()
    fig.savefig(figs_filename, bbox_inches='tight')
    plt.close(fig)

def plot_Ealpha_vs_Ns(
        Nss: list[int],
        E_alphas: list[float],
        E_dmrg: float,
        E_exact: float,
        figs_filename: str
    ):
    """
    Plots the estimated optimal energy E_alpha against the number of samples N_s.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(Nss, E_alphas, marker='o', linestyle='-', color='tab:orange', label='Estimated E_alpha')
    ax.axhline(E_dmrg, color='k', linestyle=':', label='DMRG Energy')
    ax.axhline(E_exact, color='r', linestyle=':', label='Exact Energy')
    
    ax.set_title('Estimated Energy vs Number of Samples')
    ax.set_xlabel(r'Number of Samples ($N_s$)')
    ax.set_ylabel(r'Energy $E_{\alpha}$ (Ha)')
    ax.grid(True)
    ax.legend()
    
    fig.savefig(figs_filename, bbox_inches='tight')
    plt.close(fig)
