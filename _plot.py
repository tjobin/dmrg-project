import matplotlib.pyplot as plt
import numpy as np
import os

def _get_robust_ylim(y_values, factor=3.0, pad=0.1):
    """Calculate robust y-limits ignoring extreme outliers."""
    y_vals = np.array(y_values)
    y_vals = y_vals[np.isfinite(y_vals)]
    if len(y_vals) == 0:
        return None, None
        
    q1, q3 = np.percentile(y_vals, [25, 75])
    iqr = q3 - q1
    
    # Fallback to pure min/max if IQR is exactly 0 
    if iqr == 0:
        ymin, ymax = np.min(y_vals), np.max(y_vals)
    else:
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        
        valid_y = y_vals[(y_vals >= lower_bound) & (y_vals <= upper_bound)]
        if len(valid_y) == 0:
            ymin, ymax = np.min(y_vals), np.max(y_vals)
        else:
            ymin, ymax = np.min(valid_y), np.max(valid_y)
            
    range_y = ymax - ymin
    if range_y == 0:
        range_y = abs(ymin) * 0.1 if ymin != 0 else 0.1
        
    return ymin - pad * range_y, ymax + pad * range_y

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
    
    y_sampled = (np.array(El_alpha) - E_exact + 1e-12) / (np.array(E_dmrg) - E_exact + 1e-12)
    ax.plot(chi, y_sampled, marker='o', linestyle='-.', color='tab:blue', label='DMRG + Lanczos (Sampled)')
    
    all_y = list(y_sampled)
    if El_exact is not None:
        y_exact = (np.array(El_exact) - E_exact + 1e-12) / (np.array(E_dmrg) - E_exact + 1e-12)
        ax.plot(chi, y_exact, marker='s', linestyle='-.', color='k', label='DMRG + Lanczos (Exact)')
        all_y.extend(y_exact)
        
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
    
    y_dmrg = np.array(E_dmrg) - E_exact
    y_sampled = np.array(El_alpha) - E_exact
    
    ax.plot(chi, y_dmrg, marker='s', linestyle='--', color='tab:blue', label='DMRG')
    ax.plot(chi, y_sampled, marker='o', linestyle='-.', color='tab:orange', label='DMRG + Lanczos (Sampled)')
    
    all_y = list(y_dmrg) + list(y_sampled)
    if El_exact is not None:
        y_exact = np.array(El_exact) - E_exact
        ax.plot(chi, y_exact, marker='s', linestyle='-.', color='k', label='DMRG + Lanczos (Exact)')
        all_y.extend(y_exact)
        
    ax.set_title(f'Energy vs. Chi (Lx={dim[0]}, Ly={dim[1]})')
    ax.set_xlabel('Chi')
    ax.set_ylabel(r'Energy diff  $\Delta E_l$ (Ha)')
    
    ymin, ymax = _get_robust_ylim(all_y)
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)
        
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
    all_y = []
    for i, vars in enumerate(vars_per_c):
        ax.plot(samples, vars, marker='o', linestyle='--', label=f'c={cs[i]}')
        all_y.extend(vars)
        
    ymin, ymax = _get_robust_ylim(all_y)
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)
        
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
    
    all_y = list(E_alphas)
    if E_alphas_theo is not None:
        ax.plot(alphas, E_alphas_theo, linestyle='--', color='tab:green', label='Theoretical (Sampled Moments)')
        all_y.extend(E_alphas_theo)
        
    ax.plot(alpha_star, E_alpha_star, linestyle='None', marker='o', color='tab:orange', label='Estimated Minimum')
    ax.axhline(E_dmrg, color='k', linestyle=':', label='DMRG Energy')
    ax.axhline(E_exact, color='r', linestyle=':', label='Exact Energy')
    
    all_y.extend([E_alpha_star, E_dmrg, E_exact])
    ymin, ymax = _get_robust_ylim(all_y)
    if ymin is not None and ymax is not None:
        ax.set_ylim(min(E_exact - 0.01, ymin), max(E_dmrg + 0.1, ymax))
    else:
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
    
    all_y = list(E_alphas) + [E_dmrg, E_exact]
    ymin, ymax = _get_robust_ylim(all_y)
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)
        
    ax.set_title('Estimated Energy vs Number of Samples')
    ax.set_xlabel(r'Number of Samples ($N_s$)')
    ax.set_ylabel(r'Energy $E_{\alpha}$ (Ha)')
    ax.grid(True)
    ax.legend()
    
    fig.savefig(figs_filename, bbox_inches='tight')
    plt.close(fig)

# Append to _plot.py
def plot_E_vs_time(
        chi_maxs: list[int],
        E_exact: float,
        E_dmrg: list[float],
        El_alpha: list[float],
        T_dmrg: list[float],
        T_total: list[float],
        figs_filename: str
    ):
    """
    Plots the variational energy against absolute computational time.
    Provides a direct comparison of scaling efficiency.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    
    y_dmrg = np.array(E_dmrg) - E_exact
    y_sampled = np.array(El_alpha) - E_exact

    # Plot DMRG scaling curve
    ax.plot(T_dmrg, y_dmrg, marker='s', linestyle='--', color='k', label='Standard DMRG')
    
    # Plot Lanczos scaling curve
    ax.plot(T_total, y_sampled, marker='o', linestyle='-', color='tab:orange', label='DMRG + Lanczos (Sampled)')
    
    # Annotate points with their bond dimension
    for i, chi in enumerate(chi_maxs):
        ax.annotate(f"$\\chi={chi}$", (T_dmrg[i], y_dmrg[i]), 
                     textcoords="offset points", xytext=(6, 4), ha='left', fontsize=9)
        ax.annotate(f"$\\chi={chi}$", (T_total[i], y_sampled[i]), 
                     textcoords="offset points", xytext=(6, -10), ha='left', fontsize=9)

    ax.set_xlabel("Total Wall Time (Hours)", fontsize=11)
    ax.set_ylabel(r"Energy diff $\Delta E$ (Ha)", fontsize=11)
    ax.set_title("Variational Energy Difference vs. Computational Cost", fontsize=12)
    
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(frameon=True, loc='upper right')
    
    fig.savefig(figs_filename, bbox_inches='tight')
    plt.close(fig)