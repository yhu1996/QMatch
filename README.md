# QMatch
 
## Getting Started

QMatch is Python package for simulating fermionic Gaussian states and operations. 

- Tutorial.ipynb is a jupyter notebook giving a thorough introduction of the QMatch


## Installation

```bash
# Clone the repository
git clone https://github.com/yhu1996/QMatch.git

# Navigate to the project directory
cd QMatch-main

# install the package in "editable" or "development" mode
pip install -e . 

```

## Usage

```python
# Import the QM package

from QMatch import QMatch as QM

# Use the func function
QM.func()

```


## Features

- ground state correlation matrix from the Hamiltonian: **GroundStateCorrMtx(M: np.ndarray) -> np.ndarray**
- critical Ising ground state correlation matrix: **IsingGS_CorrMtx(L: int) -> np.ndarray**
- correlation matrix for $\rho \propto \mathbb{I}_L$: **IdCorrMtx(L: int) -> np.ndarray**
- ground state energy from the Hamiltonian: **ground_state_energy(M: np.ndarray) -> float**
- tracing: **reduced_CorrMtx(Grho: np.ndarray, siteL: int, siteR: int) -> np.ndarray**
- tensor product: **tensor_prod(G1: np.ndarray, G2: np.ndarray) -> : np.ndarray**
- von Neumann Entropy: **vn_entropy(Grho: np.ndarray) -> float**
- Conditional mutual information: **CMI(Grho: np.ndarray, La: int, Lb: int, Lc: int) -> float**
- Fidelity: **Fidelity(Grho: np.ndarray, Gsigma: np.ndarray) -> float**
- Gaussian channel: **Gaussian_channel(A: np.ndarray, B: np.ndarray, Grho: np.ndarray) -> : np.ndarray**
- Erasure channel: **erasure_channel(Grho: np.ndarray, L1: int, L2: int) -> np.ndarray**
- Petz recovery map: **Petz_map(Grho: np.ndarray, A_N: np.ndarray, B_N: np.ndarray, G_sigma: np.ndarray) -> np.ndarray**
- rotated Petz recovery map: **rotated_Petz_map(Grho: np.ndarray, t: float, A_N: np.ndarray, B_N: np.ndarray, G_sigma: np.ndarray) -> np.ndarray**
- Z measurement: **measure_Z(Grho: np.ndarray, spin: int) -> np.ndarray**
