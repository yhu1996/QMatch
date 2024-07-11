# imports

import os
import sys

sys.path.append(os.path.abspath("../"))

from QMatch import QMatch as QM
from Ising_exact import *
from ncon import *

# check function CMI(Grho, La, Lb, Lc) via comparing with the exact diagonalization result

def get_rho_eigh(state, siteL, siteR, L):
    state = state.reshape(2**siteL, 2**(siteR-siteL), 2**(L-siteR))
    rho_A = ncon([state, np.conjugate(state)],[[1,-1,2],[1,-2,2]])
    rhoA_S, rhoA_U = LA.eigh(rho_A)
    rhoA_S = np.absolute(rhoA_S)
    return rhoA_S, rhoA_U


def Ising_CMI_exact_diag(La, Lb, Lc, L):
    state = Ising_ground_state(L)
    Ld = L - La - Lb - Lc
    # prepare state and density matrices
    state = state.reshape(2**La, 2**Lb, 2**Lc, 2**Ld)
    # rho_bc 
    rhoBC_S, rhoBC_U = get_rho_eigh(state, La, La+Lb+Lc, L)
    # rho_b
    rhoB_S, rhoB_U = get_rho_eigh(state, La, La+Lb, L)
    # rho_abc
    rhoABC_S, rhoABC_U = get_rho_eigh(state, 0, La+Lb+Lc, L)
    # rho_ab
    rhoAB_S, rhoAB_U = get_rho_eigh(state, 0, La+Lb, L)
    # I(A:C|B)
    IACB_state = IACB_eigval(rhoAB_S, rhoBC_S, rhoABC_S, rhoB_S)
    return IACB_state

L = 8
Grho = QM.IsingGS_CorrMtx(L)
state = Ising_ground_state(L)

for La in range(L):
    for Lb in range(L):
        for Lc in range(L):
            if L - La - Lb - Lc > 0:
                G_rhoABC = QM.reduced_CorrMtx(Grho, 0, La+Lb+Lc)
                assert abs(QM.CMI(G_rhoABC, La, Lb, Lc) - Ising_CMI_exact_diag(La, Lb, Lc, L)) < 1e-8