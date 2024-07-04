import numpy as np
import copy
from numpy import linalg as LA
from scipy import sparse as SP
import scipy.sparse.linalg as spLA
import scipy.linalg as scipyla
import random

def sgn(num):
    if num > 1e-10:
        s = 1
    elif num < -1e-10:
        s = -1
    else:
        s = 0
    return s


def sgn_lst(arr):
    sgn_arr = []
    for k in arr:
        sgn_arr.append(sgn(k))
    sgn_arr = np.array(sgn_arr)
    return sgn_arr


def square_mat_get_size(mat):
    mat = np.array(mat)
    if mat.shape[0] == mat.shape[1] and mat.shape[0]%2 == 0:
        L = mat.shape[0]//2
        return L
    else:
        print('Error: the size of the input matrix is not correct!')
        return 0


def antisym_decomp(mat):
    L = square_mat_get_size(mat)
    Lambda, O = scipyla.schur(mat)
    epsilon = np.zeros(L)
    for k in range(L):
        epsilon[k] = Lambda[2*k,2*k+1]
    return epsilon, O


def correlation_mat(M_ij):
    # M_ij is a 2L x 2L matrix
    # H = i/2 sum_{i,j} M_{ij} gamma_i gamma_j 
    
    # get the size of L and check it is legitimate
    L = square_mat_get_size(M_ij)
    
    # schur decompose M_ij 
    epsilon, O = antisym_decomp(M_ij)
    
    # two-point function tildeC = < i tildegamma_i tildegamma_j >_GS 
    tildeC = np.zeros([2*L, 2*L])
    for k in range(L):
        tildeC[2*k, 2*k+1] = - sgn(epsilon[k])
        tildeC[2*k+1, 2*k] = - tildeC[2*k, 2*k+1]
    
    # C_ij = sum_{k,l} O_{i,k} tildeC_{k,l} O^T_{l,j}
    C_ij = O @ tildeC @ O.transpose()
    
    # return the correlation function C_ij for the ground state of H
    return C_ij


# compute matrix M for the critical Ising Hamiltonian with the system size L
def Ising_Hamiltonian_mat(L):
    # H = i/2 sum_{i,j} t_{ij} gamma_i gamma_j 
    # H_Ising = i/2 sum_{k} [ gamma_k gamma_{k+1} - gamma_{k+1} gamma_k ] with periodic boundary condition gamma_{2L+1} = - gamma_1
    M_ij = np.zeros([2*L, 2*L])
    for k in range(2*L-1):
        M_ij[k,k+1] = 1
        M_ij[k+1,k] = -1
    M_ij[2*L-1,0] = -1
    M_ij[0,2*L-1] = 1
    return M_ij


# the analytical result for the correlation matrix of critical Ising ground state with the system size L
def Ising_correlation_mat(L):
    G_ij = np.zeros([2*L,2*L])
    for i in range(2*L):
        for j in range(i+1, 2*L, 2):
            G_ij[i,j] = 1/(np.sin(np.pi/(2*L)*(i-j))*L)
            G_ij[j,i] = - G_ij[i,j]
    return G_ij


def ground_state_energy(M_ij):
	# schur decompose M_ij 
    epsilon, O = antisym_decomp(M_ij)
    E_GS = -np.sum(epsilon * sgn_lst(epsilon))
    return E_GS


# get correlation matrix for reduced density matrix
def get_Grho_A(G_ij, siteL, siteR):
    # trace out complement of A, A start from siteL to siteR
    start = 2 * siteL
    new_length = 2 * (siteR - siteL)
    if new_length > 0:
        G_ij_cut = np.zeros([new_length, new_length])
        for i in range(new_length):
            for j in range(new_length):
                G_ij_cut[i,j] = G_ij[i+start, j+start]
        return G_ij_cut
    else:
        print('Warning: the resulting region is empty!')
        return 0


def tensor_prod(G1, G2):
    return scipyla.block_diag(G1,G2)


# compute vN_entropy given the correlation function
def vn_entropy(G_ij):
    epsilon_G, O = antisym_decomp(G_ij)
    S = 0
    for n in range(len(epsilon_G)):
        if abs(abs(epsilon_G[n])-1) > 1e-10:
            Sn = -(1+epsilon_G[n])/2 * np.log((1+epsilon_G[n])/2) - (1-epsilon_G[n])/2 * np.log((1-epsilon_G[n])/2)
            S += Sn
    return S


# compute CMI given the correlation function and the subsystem configuration
def Grassmann_rep_CMI(Grho, La, Lb, Lc):
    Grho_AB = get_Grho_A(Grho, 0, La+Lb)
    S_AB = vn_entropy(Grho_AB)
    Grho_BC = get_Grho_A(Grho, La, La+Lb+Lc)
    S_BC = vn_entropy(Grho_BC)
    Grho_B = get_Grho_A(Grho, La, La+Lb)
    S_B = vn_entropy(Grho_B)
    S_ABC = vn_entropy(Grho)
    return S_AB + S_BC - S_B - S_ABC


def Gaussian_channel(A, B, Grho):
    new_G = A + B @ Grho @ B.transpose()
    return new_G


def sqrt_mat(m):
    eigv, eig_U = LA.eigh(m)
    return eig_U * np.sqrt(np.absolute(eigv)) @ np.conjugate(eig_U).transpose()


def Petz_map(G_rho, A_N, B_N, G_sigma):
    G_Nsigma = Gaussian_channel(A_N, B_N, G_sigma)
    L = square_mat_get_size(G_sigma)
    I2n = np.identity(2*(L))
    B_p = sqrt_mat(I2n + G_sigma @ G_sigma) @ B_N.transpose() @ scipyla.inv(sqrt_mat(I2n + G_Nsigma @ G_Nsigma))
    A_p = G_sigma - B_p @ G_Nsigma @ B_p.transpose()
    return A_p, B_p




##################################################################
###### Critical Ising Ground State via exact diagonalization######
##################################################################

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
    log_rho = np.log(S, out=np.zeros_like(S, dtype=np.float64), where=(S > 1e-10))
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
