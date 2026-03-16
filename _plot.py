import matplotlib.pyplot as plt
import numpy as np

def plot_E_vs_chi(
        chi: list[int],
        E: list[float],
        E_alpha: list[float],
        E_exact: float,
        dim: int | list[int] = 1,
        filename: str = 'energy_vs_chi'
        ) -> None:
    """
    
    """
    plt.figure(figsize=(10, 6))
    plt.plot(chi, E, marker='o', linestyle='--', color='tab:blue', label='DMRG')
    plt.plot(chi, E_alpha, marker='s', linestyle='-.', color='tab:orange', label='DMRG + Lanczos')
    plt.axhline(E_exact, color='k', linestyle='-', label='Exact')
    plt.title(f'Energy vs. Chi (L={dim})')
    plt.xlabel('Chi')
    plt.ylabel('Energy (Ha)')
    plt.grid()
    plt.legend()
    plt.savefig(f'figs/{filename}', bbox_inches='tight')