# imports

import os
import sys

sys.path.append(os.path.abspath("../"))

from QMatch import QMatch as QM
from Ising_exact import *

# check function ground_state_energy(M)
for L in range(1,10):
    M = QM.Ising_Hamiltonian_M(L)
    E = LA.eigvalsh(Ising_H_def(L, g=1))
    assert abs(QM.ground_state_energy(M) - E[0]) < 1e-10