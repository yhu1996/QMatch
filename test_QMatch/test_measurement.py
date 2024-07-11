# imports

import os
import sys

sys.path.append(os.path.abspath("../"))

from QMatch import QMatch as QM

# check function measure_Z(Grho, spin)

# pure state -> pure state

for L in range(2, 20):
    Grho = QM.IsingGS_CorrMtx(L)
    for spin in range(L):
        newG = QM.measure_Z(Grho, spin)
        assert abs(QM.vn_entropy(newG)) < 1e-10