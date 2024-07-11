# imports

import os
import sys

sys.path.append(os.path.abspath("../"))

from QMatch import QMatch as QM
from Ising_exact import *

# check pure state: S = 0
for L in range(1, 20):
    Grho = QM.IsingGS_CorrMtx(L)
    assert abs(QM.vn_entropy(Grho)) < 1e-10

# check function vn_entropy(Grho) via comparing with the exact diagonalization result
L = 8
Grho = QM.IsingGS_CorrMtx(L)
state = Ising_ground_state(L)

for La in range(L+1):
    rhoA = get_rhoA_exact_diag(state, La, L)
    GrhoA = QM.reduced_CorrMtx(Grho, 0, La)
    assert abs(QM.vn_entropy(GrhoA) - SvN_exact_diag(rhoA)) < 1e-8
