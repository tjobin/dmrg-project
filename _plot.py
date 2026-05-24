import matplotlib.pyplot as plt
import numpy as np
import os
import json

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

import json
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_sampling_diagnostics(sampling_json_path: str, figs_filename: str, mad_threshold: float = 15.0):
    """
    Reads the raw sampling JSON generated by estimate_hamiltonian_moments
    and plots the distributions, correlations, and running averages to diagnose heavy tails.
    Displays Median Absolute Deviation (MAD) boundaries on the plots.
    """
    if not os.path.exists(sampling_json_path):
        print(f"File not found: {sampling_json_path}")
        return

    with open(sampling_json_path, 'r') as f:
        data = json.load(f)

    # Extract raw local moments
    h1_vals = np.array([float(d['h1']) for d in data.values()])
    h2_vals = np.array([float(d['h2']) for d in data.values()])
    h3_vals = np.array([float(d['h3']) for d in data.values()])
    
    N_s = len(h1_vals)
    samples = np.arange(1, N_s + 1)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # --- Calculate MAD bounds for h3 ---
    med_h3 = np.median(h3_vals)
    mad_h3 = np.median(np.abs(h3_vals - med_h3))
    
    # If the variance is strictly zero, default to a small epsilon to avoid plotting errors
    if mad_h3 < 1e-12:
        mad_h3 = 1e-12
        
    lower_bound = med_h3 - mad_threshold * mad_h3
    upper_bound = med_h3 + mad_threshold * mad_h3

    # --- Plot 1: Log-Scaled Histogram of h3 ---
    axs[0].hist(h3_vals, bins=100, color='tab:red', alpha=0.7, log=True)
    
    # Add Median and MAD threshold lines
    axs[0].axvline(med_h3, color='blue', linestyle=':', linewidth=2, label='Median')
    axs[0].axvline(lower_bound, color='k', linestyle='--', linewidth=1.5, label=f'MAD Lower ({mad_threshold}x)')
    axs[0].axvline(upper_bound, color='k', linestyle='--', linewidth=1.5, label=f'MAD Upper ({mad_threshold}x)')
    
    axs[0].set_title(r'Distribution of Local $h_3$ (Log Scale)')
    axs[0].set_xlabel(r'Local Third Moment $h_3(n)$')
    axs[0].set_ylabel('Frequency')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # --- Plot 2: Correlation of h1 vs h3 ---
    axs[1].scatter(h1_vals, h3_vals, alpha=0.3, s=10, color='tab:purple')
    axs[1].axvline(np.median(h1_vals), color='k', linestyle='--', label=r'Median $h_1$')
    
    # Add horizontal MAD bounds to see which outliers get clipped/removed
    axs[1].axhline(lower_bound, color='tab:red', linestyle='--', alpha=0.8, label=f'MAD Bounds')
    axs[1].axhline(upper_bound, color='tab:red', linestyle='--', alpha=0.8)
    
    axs[1].set_title(r'Correlation: Local $h_1$ vs Local $h_3$')
    axs[1].set_xlabel(r'Local Energy $h_1(n)$')
    axs[1].set_ylabel(r'Local Third Moment $h_3(n)$')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    # --- Plot 3: Cumulative Moving Average of h3 ---
    cum_h3 = np.cumsum(h3_vals) / samples
    axs[2].plot(samples, cum_h3, color='tab:blue', linewidth=1.5)
    axs[2].set_title(r'Cumulative Moving Average of $h_3$')
    axs[2].set_xlabel('Number of Samples')
    axs[2].set_ylabel(r'$\langle h_3 \rangle$ up to sample $n$')
    axs[2].grid(True, alpha=0.5)

    plt.tight_layout()
    
    os.makedirs(os.path.dirname(figs_filename), exist_ok=True)
    plt.savefig(figs_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved sampling diagnostics to {figs_filename}")