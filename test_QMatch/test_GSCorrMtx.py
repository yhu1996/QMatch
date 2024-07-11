# imports

import os
import sys

sys.path.append(os.path.abspath("../"))

from QMatch import QMatch as QM
import numpy as np

# check function GroundStateCorrMtx(M)
for L in range(1,20):
    M = QM.Ising_Hamiltonian_M(L)
    assert np.allclose(QM.IsingGS_CorrMtx(L), QM.GroundStateCorrMtx(M))