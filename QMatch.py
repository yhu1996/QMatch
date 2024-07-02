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


def ground_state_energy(M_ij):
	# schur decompose M_ij 
    epsilon, O = antisym_decomp(M_ij)
    E_GS = -np.sum(epsilon * sgn_lst(epsilon))
    return E_GS



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
    	print('Error: the resulting region is empty!')
    	return 0


# compute vN_entropy given the correlation function
def vn_entropy(G_ij, siteL, siteR):
    epsilon_G, O = antisym_decomp(G_ij)
    S = 0
    for n in range(len(epsilon_G)):
        if abs(abs(epsilon_G[n])-1) > 1e-10:
            Sn = -(1+epsilon_G[n])/2 * np.log((1+epsilon_G[n])/2) - (1-epsilon_G[n])/2 * np.log((1-epsilon_G[n])/2)
            S += Sn
    return S



