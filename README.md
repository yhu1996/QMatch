# QMatch
 
## Getting Started

QMatch is a Python package for simulating fermionic Gaussian states and operations. 

- Tutorial.ipynb is a jupyter notebook giving a thorough introduction to the QMatch



## Usage
First, download the folder QMatch.

Then, for example, put your scripts in the same directory as the folder QMatch. 

```python
# Import the QMatch package
import sys
import os
sys.path.append(os.path.abspath("./"))
from QMatch import QMatch as QM

# Use the func function
QM.func()

```


## Features

- Correlation matrices:
  - get the ground state correlation matrix from the Hamiltonian
  - correlation matrix for the critical Ising ground state  
  - correlation matrix for $\rho \propto \mathbb{I}_L$
- Ground state energy from the Hamiltonian
- The partial trace over the subregion
- Tensor product
- von Neumann Entropy
- Conditional mutual information
- Fidelity
- Gaussian channel
- Erasure channel
- Petz recovery map
- Rotated Petz recovery map
- Gaussian measurements:
   - Z measurement
