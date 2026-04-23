import logging
import json
import os
from tenpy.networks.mps import MPS
from tenpy.models.spins import SpinModel
from tenpy.algorithms import dmrg

logging.basicConfig(level=logging.INFO)

class j1j2_model:
    def __init__(self, Lx, Ly, j1, j2, bc_x, bc_y):
        self.Lx = Lx
        self.Ly = Ly
        self.j1 = j1
        self.j2 = j2
        self.bc_x = bc_x
        self.bc_y = bc_y
        self.model_params = {
            'lattice': 'Square',
            'Lx': self.Lx, 'Ly': self.Ly, 'S': 0.5,
            'conserve': 'Sz',
            'bc_MPS': 'finite',
            'bc_x': self.bc_x, 'bc_y': self.bc_y,
            'Jx': self.j1, 'Jy': self.j1, 'Jz': self.j1,
        }
        model = SpinModel(self.model_params)
        model.manually_call_init_H = True 
        for dist in [[1, 1], [1, -1]]:
            model.add_coupling(self.j2, 0, 'Sz', 0, 'Sz', dist)
            model.add_coupling(self.j2/2., 0, 'Sp', 0, 'Sm', dist)
            model.add_coupling(self.j2/2., 0, 'Sm', 0, 'Sp', dist)
        model.init_H_from_terms()
        self.model = model
        
    def get_mpo(self):
        return self.model.H_MPO
        
    def run(self, chi_max):
        # Dynamic filename based on parameters
        if self.bc_x == 'periodic' and self.bc_y == 'periodic':
            lattice = f'square_{self.Lx}x{self.Ly}_torus'
        elif self.bc_x == 'periodic' and self.bc_y == 'open':
            lattice = f'square_{self.Lx}x{self.Ly}_cylinder'
        filepath = f'log_dmrg/J1J2_{self.Lx}x{self.Ly}_{lattice}/cleaned/'
        filename = f"DMRG_chi={chi_max}.json"
        
        model = self.model

        # Optional: print the new MPO bond dimension to see the improvement
        # logging.info(f"MPO bond dimensions after compression: {model.H_MPO.chi}")
        # -----------------------------------------

        # 2. Initialize MPS
        n_sites = model.lat.N_sites
        psi = MPS.from_product_state(model.lat.mps_sites(), 
                                    (['up', 'down'] * (n_sites // 2 + 1))[:n_sites])

        # 3. DMRG Settings
        dmrg_params = {
            'mixer': True,
            'trunc_params': {'chi_max': chi_max, 'svd_min': 1.e-12},
            'max_E_err': 1.e-10,
            'max_trunc_err': 1.e-1,
            'max_sweeps': 64,
            'active_sites': 2,
            'combine': True,
        }

        # 4. Run
        info = dmrg.run(psi, model, dmrg_params)
        
        # 5. Compute Final Stats
        energy = info['E']
        variance = model.H_MPO.variance(psi)
        v_score = n_sites * variance / (energy**2)
        
        # 6. Extract and format sweep statistics
        stats = info.get('sweep_statistics', {})
        
        data_to_save = {
            "params": {
                "Lx": self.Lx, "Ly": self.Ly, "J1": self.j1, "J2": self.j2, "chi_max": chi_max, "n_sites": n_sites
            },
            "convergence": {
                "energies": [float(e) for e in stats.get('E', [])],
                "max_chi": [int(c) for c in stats.get('max_chi', [])],
                "trunc_err": [float(t) for t in stats.get('max_trunc_err', [])],
                "entropy": [float(s) for s in stats.get('max_S', [])]
            },
            "final_results": {
                "energy": float(energy),
                "v_score": float(v_score),
                "variance": float(variance),
                "sweeps_done": len(stats.get('E', []))
            }
        }

        # Save to file
        os.makedirs(filepath, exist_ok=True)
        with open(f'{filepath}{filename}', 'w') as f:
            json.dump(data_to_save, f, indent=4)

        print("\n" + "="*40)
        print(f"Results saved to: {filename}")
        print(f"Final Energy:     {energy:.12f}")
        print(f"Final V-score:    {v_score:.6e}")
        print("="*40)
        
        return energy, psi
