# dmrg-project

---

### 📖 Overview

dmrg-project is a computational framework for simulating the $J_1-J_2$ Heisenberg model on a cylinder using tensor network techniques. It implements Density Matrix Renormalization Group (DMRG) to optimize Matrix Product States (MPS).

Furthermore, it enhances ground state energy approximations by applying a variational Lanczos step. The Lanczos step relies on optimizing a variational parameter $\alpha$ based on the first three moments of the Hamiltonian. These moments can be evaluated exactly or estimated via perfect MPS-sampling.

---

### 🛠 Installation

Prerequisites:

* Python $\ge 3.13$
* [TeNPy](https://tenpy.readthedocs.io/) (`physics-tenpy`) - Tensor network library for quantum many-body physics.
* [Hydra](https://hydra.cc/) (`hydra-core`) - Framework for elegantly configuring complex applications.
* `matplotlib`, `pytest`, `tqdm`

```bash
# Clone the repository
git clone [https://github.com/tjobin/dmrg-project.git](https://github.com/tjobin/dmrg-project.git)
cd dmrg-project
```

Install dependencies via pip using pyproject.toml or via uv using uv.lock
```bash
pip install .
```
or
```bash
uv sync
```


---

### 🚀 Usage 

#### Configuration

The project uses Hydra for configuration management. Parameters are set in conf/config.yaml:

-System: Adjust lattice dimensions (Lx, Ly) and coupling constants (j1, j2).

-DMRG: Define the list of maximum bond dimensions (chi_maxs) to sweep over.

-Lanczos: Set the number of samples (Nss) for Hamiltonian moment estimation and define the random seed.

#### Execution

Run the simulation using the main script:
- standard
```bash
    python main.py
```
- with uv
```bash
    uv run main.py
```
---
### 📊 Results

The simulation executes the following pipeline:

- Calculates or retrieves the exact ground state energy for the specified J1​-J2 cylinder geometry.

- Runs standard DMRG for the defined bond dimensions χmax and records the energy.

- Applies a sampled Lanczos step to the resulting MPS to estimate an optimized energy (Eα) and variational parameter (α).

- Generates performance plots comparing the absolute and relative energy errors (ΔE) against the bond dimension χ. These figures are saved in a dynamically generated directory corresponding to the system size (e.g., J1J2_5x5/).



