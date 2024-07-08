import numpy as np
from numpy import linalg as LA
import scipy.linalg as scipyla
import random

###################################################################
###### Critical Ising Ground State via exact diagonalization ######
###################################################################

def Ising_H_def(L, g):
    mat = np.zeros([2**L,2**L])
    # Pauli Matrices
    sigma_x = np.array([[0,1],[1,0]])
    sigma_z = np.array([[1,0],[0,-1]])
    id2 = np.array([[1,0],[0,1]])
    for i in range(L):
        ## X_i X_i+1
        if i == 0 or i == L-1:
            XX = sigma_x
        else:
            XX = id2
        for j in range(1,L):
            if j == i or j == i+1:
                XX = np.kron(XX, sigma_x)
            else:
                XX = np.kron(XX, id2)  
        ## Z_i
        if i == 0:
            Z = sigma_z
        else: 
            Z = id2
        for j in range(1,L):
            if j == i:
                Z = np.kron(Z,sigma_z)
            else:
                Z = np.kron(Z,id2)

        mat -= (XX + g*Z)
    return mat

def SvN_exact_diag(rho):
    S = LA.eigvalsh(rho)
    log_rho = np.log2(S, out=np.zeros_like(S, dtype=np.float64), where=(S > 1e-10))
    entropy = - np.dot(S, log_rho)
    return entropy


def get_rhoA_exact_diag(state, L_A, L):
    if L_A <= L:
        state = state.reshape(2**L_A, 2**(L-L_A))
        rho_A = np.matmul(state, np.conjugate(state).transpose())
        return rho_A
    else:
        print('Error: the subregion size is bigger than the system size!')
        return 0

def Ising_SvN_exact_diag(L_A, L):
    H = Ising_H_def(L, 1)
    eigv, eigvec = LA.eigh(H)
    state = eigvec[:,0]
    rho_A = get_rhoA_exact_diag(state, L_A, L)
    SvN = SvN_exact_diag(rho_A)
    return SvN
