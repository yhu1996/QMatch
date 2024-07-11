# imports

import os
import sys

sys.path.append(os.path.abspath("../"))

from QMatch import QMatch as QM
from Ising_exact import *
from Uhlmann_Fidelity import *

# check function Petz_map(Grho, A_N, B_N, G_sigma) by check the Petz fidelity

def erasure_rotated_Petz(Grho_ABC, La, Lb, Lc, t):
    A_N, B_N = QM.erasure_channel_param(La+Lb, Lc)
    Gsigma = QM.tensor_prod(QM.reduced_CorrMtx(Grho_ABC, 0, La), QM.reduced_CorrMtx(Grho_ABC, La, La+Lb+Lc))
    G_X = QM.tensor_prod(QM.reduced_CorrMtx(Grho_ABC,0,La+Lb),QM.IdCorrMtx(Lc))
    G_tilde_rho = QM.rotated_Petz_map(G_X, t, A_N, B_N, Gsigma)
    return G_tilde_rho


L = 6
t_lst = [x/2 for x in range(-10, 11, 1)]
Grho = QM.IsingGS_CorrMtx(L)
state = Ising_ground_state(L)
for La in range(L):
    for Lb in range(L):
        for Lc in range(L):
            if L - (La + Lb + Lc) > 0:
                Grho_ABC = QM.reduced_CorrMtx(Grho, 0, La+Lb+Lc)
                for t in t_lst:
                    F1 = QM.Fidelity(Grho_ABC, erasure_rotated_Petz(Grho_ABC, La, Lb, Lc, t))
                    F2 = UFt_state(state, La, Lb, Lc, L, t)
                    assert abs(F1-F2) < 1e-6