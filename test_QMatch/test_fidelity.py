# imports

import os
import sys

sys.path.append(os.path.abspath("../"))

from QMatch import QMatch as QM
from Ising_exact import *

# check function Fidelity(Grho, Gsigma)
def fidelity(rho, sigma):
    Lambda_rho, evector_rho = LA.eigh(rho)
    sqrt_rho = evector_rho * np.sqrt(np.absolute(Lambda_rho)) @ np.conjugate(evector_rho).transpose()
    mat = sqrt_rho @ sigma @ sqrt_rho 
    Lambda_mat = LA.eigvalsh(mat)
    fidelity = np.sum(np.sqrt(np.absolute(Lambda_mat)))
    return fidelity

# set up two test states
for L1 in range(2,7):
    for L2 in range(1,L1):
        Grho = QM.reduced_CorrMtx(QM.IsingGS_CorrMtx(L1), 0, L2)
        Gsigma = QM.IsingGS_CorrMtx(L2)
        F1 = QM.Fidelity(Grho, Gsigma)

        state1 = Ising_ground_state(L1)
        state2 = Ising_ground_state(L2)
        rho = get_rhoA_exact_diag(state1, L2, L1)
        sigma = get_rhoA_exact_diag(state2, L2, L2)
        F2 = fidelity(rho, sigma)

        assert abs(F1-F2) < 1e-4