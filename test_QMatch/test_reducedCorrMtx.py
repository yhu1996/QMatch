# imports

import os
import sys

sys.path.append(os.path.abspath("../"))

from QMatch import QMatch as QM
import numpy as np

# check function reduced_CorrMtx(Grho, siteL, siteR)

# set up a test matrix
L = 3
test_matrix = np.zeros([2*L,2*L])
for i in range(2*L):
    for j in range(2*L):
        test_matrix[i,j] = (i+1)*10 + (j+1)

# check when siteR > siteL
trim_mat1 = np.array([[11., 12.],[21.,22.]])
np.testing.assert_array_equal(trim_mat1, QM.reduced_CorrMtx(test_matrix, 0, 1))

trim_mat2 = np.array([[11., 12., 13., 14.],[21.,22.,23., 24.],[31., 32., 33., 34.],[41., 42., 43., 44.]])
np.testing.assert_array_equal(trim_mat2, QM.reduced_CorrMtx(test_matrix, 0, 2))

np.testing.assert_array_equal(test_matrix, QM.reduced_CorrMtx(test_matrix, 0, 3))

trim_mat3 = np.array([[33., 34.],[43.,44.]])
np.testing.assert_array_equal(trim_mat3, QM.reduced_CorrMtx(test_matrix, 1, 2))

trim_mat4 = np.array([[33., 34., 35., 36.],[43.,44.,45., 46.],[53., 54., 55., 56.],[63., 64., 65., 66.]])
np.testing.assert_array_equal(trim_mat4, QM.reduced_CorrMtx(test_matrix, 1, 3))

trim_mat5 = np.array([[55., 56.],[65.,66.]])
np.testing.assert_array_equal(trim_mat5, QM.reduced_CorrMtx(test_matrix, 2, 3))

# check when siteR <= siteL
for siteR in range(L):
    for siteL in range(siteR, L):
        np.testing.assert_array_equal(np.zeros([0,0]), QM.reduced_CorrMtx(test_matrix, siteL, siteR))