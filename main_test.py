from lanczos_method import optimize_lanczos_step
from Hmoments import evaluate_tfim_moments

def test_lanczos_optimization():
    # Parameters for the TFIM
    L = 100
    J = 1.0
    g = 1.0
    chi_max = 10

    # Step 1: Evaluate the moments <H>, <H^2>, and <H^3> using TeNPy
    h1, h2, h3 = evaluate_tfim_moments(L=L, J=J, g=g, chi_max=chi_max)

    print(f"Evaluated moments:")
    print(f"<H>   : {h1:.8f}")
    print(f"<H^2> : {h2:.8f}")
    print(f"<H^3> : {h3:.8f}")

    # Step 2: Optimize the Lanczos step using the evaluated moments
    E_opt, alpha_opt = optimize_lanczos_step(h1, h2, h3)

    print(f"\nLanczos optimization results:")
    print(f"Optimized Energy   : {E_opt:.8f}")
    print(f"Optimal alpha      : {alpha_opt:.8f}")

if __name__ == "__main__":
    test_lanczos_optimization()