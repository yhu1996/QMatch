import numpy as np
from numpy import linalg as LA
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


def GroundStateCorrMtx(M):
    # M_ij is a 2L x 2L matrix
    # H = i/2 sum_{i,j} M_{ij} gamma_i gamma_j 
    
    # get the size of L and check it is legitimate
    L = square_mat_get_size(M)
    
    # schur decompose M_ij 
    M = np.array(M)
    epsilon, O = antisym_decomp(M)
    
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
def Ising_Hamiltonian_M(L):
    # H = i/2 sum_{i,j} t_{ij} gamma_i gamma_j 
    # H_Ising = i/2 sum_{k} [ gamma_k gamma_{k+1} - gamma_{k+1} gamma_k ] with periodic boundary condition gamma_{2L+1} = - gamma_1
    M_ij = np.zeros([2*L, 2*L])
    for k in range(2*L-1):
        M_ij[k,k+1] = 1
        M_ij[k+1,k] = -1
    M_ij[2*L-1,0] += -1
    M_ij[0,2*L-1] += 1
    return M_ij


# the analytical result for the correlation matrix of critical Ising ground state with the system size L
def IsingGS_CorrMtx(L):
    G_ij = np.zeros([2*L,2*L])
    for i in range(2*L):
        for j in range(i+1, 2*L, 2):
            G_ij[i,j] = 1/(np.sin(np.pi/(2*L)*(i-j))*L)
            G_ij[j,i] = - G_ij[i,j]
    return G_ij


def IdCorrMtx(L):
    return np.zeros([2*L,2*L])


def ground_state_energy(M):
	# schur decompose M_ij 
    epsilon, O = antisym_decomp(M)
    E_GS = -np.sum(epsilon * sgn_lst(epsilon))
    return E_GS


# get correlation matrix for reduced density matrix
def reduced_CorrMtx(Grho, siteL, siteR):
    # trace out complement of A, A start from (siteL+1)-th spin to siteR-th spin
    start = 2 * siteL
    new_length = 2 * (siteR - siteL)
    if new_length > 0:
        G_cut = np.zeros([new_length, new_length])
        for i in range(new_length):
            for j in range(new_length):
                G_cut[i,j] = Grho[i+start, j+start]
        return G_cut
    else:
        #print('Warning: the resulting region is empty!')
        return np.zeros([0,0])


def tensor_prod(G1, G2):
    return scipyla.block_diag(G1,G2)


# compute vN_entropy given the correlation function
def vn_entropy(Grho):
    epsilon_G, O = antisym_decomp(Grho)
    S = 0
    for n in range(len(epsilon_G)):
        if abs(abs(epsilon_G[n])-1) > 1e-10:
            Sn = -(1+epsilon_G[n])/2 * np.log2((1+epsilon_G[n])/2) - (1-epsilon_G[n])/2 * np.log2((1-epsilon_G[n])/2)
            S += Sn
    return S


# compute CMI given the correlation function and the subsystem configuration
def CMI(GrhoABC, La, Lb, Lc):
    Grho_AB = reduced_CorrMtx(GrhoABC, 0, La+Lb)
    S_AB = vn_entropy(Grho_AB)
    Grho_BC = reduced_CorrMtx(GrhoABC, La, La+Lb+Lc)
    S_BC = vn_entropy(Grho_BC)
    Grho_B = reduced_CorrMtx(GrhoABC, La, La+Lb)
    S_B = vn_entropy(Grho_B)
    S_ABC = vn_entropy(GrhoABC)
    return S_AB + S_BC - S_B - S_ABC


def Gaussian_channel(A, B, Grho):
    new_G = A + B @ Grho @ B.transpose()
    return new_G


def erasure_channel_param(L1, L2):
    # tr_region2 & then otimes I_region2
    A = np.zeros([2*(L1+L2),2*(L1+L2)])
    B = scipyla.block_diag(np.identity(2*L1),np.zeros([2*L2,2*L2]))
    return A, B


def erasure_channel(Grho, L1, L2):
	A, B = erasure_channel_param(L1, L2)
	G_new = Gaussian_channel(A, B, Grho)
	return G_new


def sqrt_mtx_h(m):
    # m is a hermitian matrix
    eigv, eig_U = LA.eigh(m)
    return eig_U * np.sqrt(np.absolute(eigv)) @ np.conjugate(eig_U).transpose()


# return parameters: matrices A and B for the Petz map
def Petz_map_param(A_N, B_N, G_sigma):
    G_Nsigma = Gaussian_channel(A_N, B_N, G_sigma)
    L = square_mat_get_size(G_sigma)
    I2n = np.identity(2*L)
    B_p = sqrt_mtx_h(I2n + G_sigma @ G_sigma) @ B_N.transpose() @ scipyla.inv(sqrt_mtx_h(I2n + G_Nsigma @ G_Nsigma))
    A_p = G_sigma - B_p @ G_Nsigma @ B_p.transpose()
    return A_p, B_p


# compute the new correlation matrix after implementing the Petz recovery map
# Grho has the same size as G_sigma
def Petz_map(Grho, A_N, B_N, G_sigma):
    A_p, B_p = Petz_map_param(A_N, B_N, G_sigma)
    G_petz = A_p + B_p @ Grho @ B_p.transpose()
    return G_petz


# compute the fidelity between rho & sigma
def Fidelity(Grho, Gsigma):
    # n = the size of the system
    n = square_mat_get_size(Grho)
    I2n = np.identity(2*n)
    tilde_G = Grho + sqrt_mtx_h(I2n + Grho @ Grho) @ scipyla.inv(scipyla.inv(Gsigma) - Grho) @ sqrt_mtx_h(I2n + Grho @ Grho)

    F = 1/ 2**(n/2) * (scipyla.det(I2n - Grho @ Gsigma))**(1/4) * (scipyla.det(I2n + sqrt_mtx_h(I2n + tilde_G @ tilde_G)))**(1/4)

    return F.real


def B_sigma_t_Block(eps, t):
    if abs(abs(eps)-1) < 1e-10:
        return np.identity(2)
    block = np.zeros([2,2])
    factor = (1+eps)/(1-eps)
    block[0,0] = np.real(factor**(1j*t/2))
    block[0,1] = -np.imag(factor**(1j*t/2))
    block[1,0] = - block[0,1]
    block[1,1] = block[0,0]
    return block

def B_sigma_t(G_sigma, t, size_n):
    eps, O = antisym_decomp(G_sigma)
    if size_n == 0:
        return np.zeros([0,0])
    block = B_sigma_t_Block(eps[0], t)
    for i in range(1,size_n):
        block = scipyla.block_diag(block,B_sigma_t_Block(eps[i], t))
    return O @ block @ O.transpose()


# return parameters: matrices A and B for rotated Petz map
def rotated_Petz_param(t, A_N, B_N, G_sigma):
    #STEP1: Petz map
    G_Nsigma = Gaussian_channel(A_N, B_N, G_sigma)
    L = square_mat_get_size(G_sigma)
    I2n = np.identity(2*L)
    B_p = sqrt_mtx_h(I2n + G_sigma @ G_sigma) @ B_N.transpose() @ scipyla.inv(sqrt_mtx_h(I2n + G_Nsigma @ G_Nsigma))
    A_p = G_sigma - B_p @ G_Nsigma @ B_p.transpose()
    #STEP2: rotated map 
    B_sigma = B_sigma_t(G_sigma, t, L)
    BN_sigma = B_sigma_t(G_Nsigma, -t, L)
    A_R = B_sigma @ A_p @ B_sigma.transpose()
    B_R = B_sigma @ B_p @ BN_sigma
    return A_R, B_R


# return the new correlation matrix after implementing a rotated Petz map
def rotated_Petz_map(Grho, t, A_N, B_N, G_sigma):
    A_R, B_R = rotated_Petz_param(t, A_N, B_N, G_sigma)
    G_rotated = A_R + B_R @ Grho @ B_R.transpose()
    return G_rotated


# measure Z at site = spin, return the new correlation matrix after measurement 
def measure_Z(Grho, spin):
    # site = spin in (0,...,L-1)
    L = square_mat_get_size(Grho)
    random_num = random.uniform(0, 1)
    prob_0 = 1/2 + 1/2 * Grho[2*spin, 2*spin+1]
    if random_num < prob_0:
        epsilon = 0
    else:
        epsilon = 1
    x = Grho[2*spin, 2*spin+1] * (-1)**(epsilon+1)
    new_Cij = np.zeros([2*L,2*L])
    for a in range(L):
        if a != spin:
            for b in range(L):
                if b != spin:
                    new_Cij[2*a,2*b] = Grho[2*a, 2*b] - (-1)**(epsilon+1)/(1-x) * (Grho[2*a, 2*spin] * Grho[2*spin+1, 2*b]- Grho[2*a, 2*spin+1] * Grho[2*spin, 2*b] )
                    new_Cij[2*a+1,2*b] = Grho[2*a+1, 2*b] - (-1)**(epsilon+1)/(1-x) * (Grho[2*a+1, 2*spin] * Grho[2*spin+1, 2*b]- Grho[2*a+1, 2*spin+1] * Grho[2*spin, 2*b] )
                    new_Cij[2*a,2*b+1] = Grho[2*a, 2*b+1] - (-1)**(epsilon+1)/(1-x) * (Grho[2*a, 2*spin] * Grho[2*spin+1, 2*b+1]- Grho[2*a, 2*spin+1] * Grho[2*spin, 2*b+1] )
                    new_Cij[2*a+1,2*b+1] = Grho[2*a+1, 2*b+1] - (-1)**(epsilon+1)/(1-x) * (Grho[2*a+1, 2*spin] * Grho[2*spin+1, 2*b+1]- Grho[2*a+1, 2*spin+1] * Grho[2*spin, 2*b+1] )
    new_Cij[2*spin, 2*spin+1] = (-1)**epsilon
    new_Cij[2*spin+1, 2*spin] = -(-1)**epsilon
    return new_Cij