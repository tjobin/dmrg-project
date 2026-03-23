import matplotlib.pyplot as plt
import numpy as np
from utils import test_alphas


def plot_E_vs_chi(
        chi: list[int],
        E: list[float],
        E_alpha: list[float],
        E_exact: float | None = None,
        dim: int | list[int] = 1,
        filename: str = 'energy_vs_chi'
        ) -> None:
    """
    
    """
    plt.figure(figsize=(10, 6))
    plt.plot(chi, E, marker='o', linestyle='--', color='tab:blue', label='DMRG')
    plt.plot(chi, E_alpha, marker='s', linestyle='-.', color='tab:orange', label='DMRG + Lanczos')
    if E_exact is not None:
        plt.axhline(E_exact, color='k', linestyle='-', label='Exact')
    plt.title(f'Energy vs. Chi (L={dim})')
    plt.xlabel('Chi')
    plt.ylabel('Energy (Ha)')
    plt.grid()
    plt.legend()
    plt.savefig(f'figs/{filename}', bbox_inches='tight')

def plot_variance_vs_samples(
        vars_per_c: list,
        samples: list,
        cs: list,
        filename: str = 'variance_vs_samples'
    ):
    plt.figure()
    for i, vars in enumerate(vars_per_c):
        plt.plot(samples, vars, marker='o', linestyle='--', label=f'c={cs[i]}')
    plt.legend()
    plt.savefig(f'figs/{filename}.png', bbox_inches='tight')

def plot_Ealpha_vs_alpha(
        alphas,
        h1,
        h2,
        h3,
        psii,
        H_mpo
    ):

    E_alphas_est = (h1 + 2*alphas*h2 + alphas ** 2 * h3) / (1 + 2 * alphas * h1 + alphas ** 2 * h2)
    E_alphas = test_alphas(alphas, psii, H_mpo)
    plt.figure()
    plt.plot(alphas, E_alphas_est, linestyle='--', marker='o', color='tab:blue',label='Estimation')
    plt.plot(alphas, E_alphas, linestyle='-.', marker='s', color='tab:orange', label='Exact')
    plt.legend()
    plt.show()
