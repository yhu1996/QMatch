import numpy as np
from ncon import *
from numpy import linalg as LA
from scipy import linalg as spLA

##################################################
### compute eigenval & U for a density matrix ###
##################################################

def safe_SVD(mat):
	try:
		U, S, Vh = spLA.svd(mat, full_matrices=False)
	except:
		U, S, Vh = spLA.svd(mat, full_matrices=False, lapack_driver = 'gesvd')
	return U, S, Vh

def get_rho_eigh(state, siteL, siteR, L):
    state = state.reshape(2**siteL, 2**(siteR-siteL), 2**(L-siteR))
    rho_A = ncon([state, np.conjugate(state)],[[1,-1,2],[1,-2,2]])
    rhoA_S, rhoA_U = LA.eigh(rho_A)
    rhoA_S = np.absolute(rhoA_S)
    return rhoA_S, rhoA_U

def SVD_rho(state):
    # assume rho = state * state^conjugate
    # state already reshaped into e.g. psi_{BC,AD} form
    U, S, Vh = safe_SVD(state)
    rho_S = S**2
    return rho_S, U

def get_rho_SVD(state, siteL, siteR, L):
    state = state.reshape(2**siteL, 2**(siteR-siteL), 2**(L-siteR))
    rhoA_S, rhoA_U = SVD_rho(np.transpose(state,(1,0,2)).reshape(2**(siteR-siteL),2**(L-siteR+siteL)))
    return rhoA_S, rhoA_U

def get_rho(state, siteL, siteR, L): 
    # goal: rho_A
    # region A starts from siteL to siteR: 2**siteL, 2**(siteR-siteL), 2**(L-siteR)
    region_A = siteR - siteL
    if region_A <= L/2:
        rhoA_S, rhoA_U = get_rho_eigh(state, siteL, siteR, L)
    else:
        rhoA_S, rhoA_U = get_rho_SVD(state, siteL, siteR, L)
    return rhoA_S, rhoA_U



#####################################################
### compute Uhlmann fidelity for rotated Petz map ###
#####################################################

def Petz_fidelity(state, rhoABC_S, rhoABC_U, rhoAB_S, rhoAB_U, rhoB_S, rhoB_U, rhoBC_S , rhoBC_U, La, Lb, Lc, L, t):
    BC_power_left = 1/2+1j*t/2
    B_power_left = -1/2-1j*t/2
    LambB_S = np.where(rhoB_S > 0.0000000001, rhoB_S**(B_power_left), 0)
    LambBC_S = np.where(rhoBC_S > 0.0000000001, rhoBC_S**(BC_power_left), 0)

    Ld = L - La - Lb - Lc
    state = state.reshape(2**La,2**Lb,2**Lc,2**Ld)
    
    if La + Lb >= Lc + Ld:
        rho_AB_pure = state.reshape(2**La,2**Lb,2**(Lc+Ld))
        size_CD = Lc + Ld
    else:
        rho_AB_pure = rhoAB_U * np.sqrt(rhoAB_S) @ np.conjugate(rhoAB_U).transpose()
        size_CD = La + Lb
        rho_AB_pure = rho_AB_pure.reshape(2**La,2**Lb,2**size_CD) 
    bottom_p1 = ncon([np.conjugate(rhoB_U), rho_AB_pure], [[1,-3],[-2,1,-1]])
    bottom = ncon([(rhoB_U * LambB_S),bottom_p1], [[-3,1],[-1,-2,1]])
    
    if La + Lb + Lc >= Ld:
        rhoABC_pure_conj = np.conjugate(state).reshape(2**La,2**(Lb+Lc),2**Ld)
        # size of D* = Ld
        size_D_star = Ld
    else:
        rhoABC_pure = rhoABC_U * np.sqrt(rhoABC_S) @ np.conjugate(rhoABC_U).transpose()
        rhoABC_pure_conj = np.conjugate(rhoABC_pure).reshape(2**La,2**(Lb+Lc),2**(La+Lb+Lc))
        # size of D* = La + Lb + Lc
        size_D_star = La + Lb + Lc
    top_p1 = ncon([rhoABC_pure_conj,(rhoBC_U * LambBC_S)],[[-2,1,-3],[1,-1]])
    top = ncon([top_p1,np.conjugate(rhoBC_U)],[[1,-2,-3],[-1,1]])
    top = top.reshape(2**Lb, 2**Lc, 2**La, 2**size_D_star)

    XdagY = ncon([bottom,top],[[-1,1,2],[2,-2,1,-3]])
    XdagY = XdagY.reshape(2**(size_CD+Lc),2**size_D_star)
    U, S, Vh = safe_SVD(XdagY)
    #S = spLA.svdvals(XdagY)
    Fidelity = np.sum(S)
    
    return Fidelity


def UFt_state(state, La, Lb, Lc, L, t):
    
    Ld = L - La - Lb - Lc
    # prepare state and density matrices
    state = state.reshape(2**La, 2**Lb, 2**Lc, 2**Ld)
    # rho_bc 
    rhoBC_S, rhoBC_U = get_rho(state, La, La+Lb+Lc, L)
    # rho_b
    rhoB_S, rhoB_U = get_rho(state, La, La+Lb, L)
    # rho_abc
    rhoABC_S, rhoABC_U = get_rho(state, 0, La+Lb+Lc, L)
    # rho_ab
    rhoAB_S, rhoAB_U = get_rho(state, 0, La+Lb, L)
    
    # compute fidelity
    f = Petz_fidelity(state, rhoABC_S, rhoABC_U, rhoAB_S, rhoAB_U, rhoB_S, rhoB_U, rhoBC_S , rhoBC_U, La, Lb, Lc, L, t)
    
    return f