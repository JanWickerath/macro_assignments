"""
Compute reservation wage of a given instance of a McCallModel.

"""

import numpy as np
from mccall_bellman_iteration import *


def compute_reservation_wage(mcm):
    """
    Compute and return the reservation wage of a given model of a McCall growth
    model.

    """

    res_wage = np.inf
    V, U = solve_mccall_model(mcm)

    w_idx = 0
    for v_idx, value in enumerate(V):
        if value > U:
            res_wage = mcm.w_vec[v_idx]
            break

    return res_wage
