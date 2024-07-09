# imports

import os
import sys

sys.path.append(os.path.abspath("../"))

from QMatch import QMatch as QM
import numpy as np

# check function erasure_channel(Grho, L1, L2)
# erasure channel should be equivalent to tracing out the subregion L2 and then tensor product with the identity matrix

# set up a test matrix
L = 3
test_matrix = np.zeros([2*L,2*L])
for i in range(2*L):
    for j in range(2*L):
        test_matrix[i,j] = (i+1)*10 + (j+1)


for L1 in range(L):
    L2 = L - L1
    array1 = QM.tensor_prod(QM.reduced_CorrMtx(test_matrix, 0, L1), QM.IdCorrMtx(L2))
    array2 = QM.erasure_channel(test_matrix, L1, L2)
    np.testing.assert_array_equal(array1, array2)
