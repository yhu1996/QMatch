import numpy as np
from numpy import linalg as LA


###################################################################
###### Critical Ising Ground State via exact diagonalization ######
###################################################################

def Ising_H_def(L, g):
    mat = np.zeros([2**L,2**L])
    # Pauli Matrices
    sigma_x = np.array([[0,1],[1,0]])
    sigma_z = np.array([[1,0],[0,-1]])
    id2 = np.array([[1,0],[0,1]])
    if L == 1:
        return -id2-g*sigma_z
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


def SvN_eigval(S):
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
        return np.zeros([0,0])


def Ising_ground_state(L):
    H = Ising_H_def(L, 1)
    eigv, eigvec = LA.eigh(H)
    state = eigvec[:,0]
    return state


def Ising_SvN_exact_diag(L_A, L):
    state = Ising_ground_state(L)
    rho_A = get_rhoA_exact_diag(state, L_A, L)
    SvN = SvN_exact_diag(rho_A)
    return SvN



# compute I(A:C|B) = S_AB + S_BC - S_B - S_ABC
def IACB_eigval(rho_AB_eig, rho_BC_eig, rho_ABC_eig, rho_B_eig):

	S_AB = SvN_eigval(rho_AB_eig)
	S_BC = SvN_eigval(rho_BC_eig)
	S_B = SvN_eigval(rho_B_eig)
	S_ABC = SvN_eigval(rho_ABC_eig)

	return S_AB + S_BC - S_B - S_ABC