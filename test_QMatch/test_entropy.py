# imports

import os
import sys

sys.path.append(os.path.abspath("../"))

from QMatch import QMatch as QM
from Ising_exact import *

# check function vn_entropy(Grho) via comparing with the exact diagonalization result
L = 6
Grho = QM.IsingGS_CorrMtx(L)
state = Ising_ground_state(L)

for La in range(L+1):
    rhoA = get_rhoA_exact_diag(state, La, L)
    GrhoA = QM.reduced_CorrMtx(Grho, 0, La)
    if abs(QM.vn_entropy(GrhoA) - SvN_exact_diag(rhoA)) > 1e-10:
        print("Error happens when La = %d"%La)


# check function CMI(Grho, La, Lb, Lc)
