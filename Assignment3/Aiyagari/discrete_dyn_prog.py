"""Implementation of different algorithms to solve discrete dynamic programming
problems. Description of algorithms are mainly taken from the textbook by Heer
and Maussner (2009).

"""

import numpy as np


def value_iter_deter(state_grid, reward_fun, beta, tol=1e-8):
    """Docstring for deterministic value function iteration."""
    # Denote length of the state grid with n

    # Initialize first guess of value function to vector of zeros with length
    # n.

    # Initialize nXn array *util_mat* to compute utility levels for different
    # state-choice combinations.

    # Loop over rows of util_mat

        # Loop over cols of util_mat

            # Set util_mat(row,col)=reward_fun(state_grid(row),state_grid(col))

    # Initialize distance to infinity

    # While distance is larger then tolerance

        # Initialize vector of new values v_new of length n.

        # Loop over indices (idx) of v_new

            # Compute
            # _bin_search_opt(util_mat[:, idx] + beta * v[idx]
            # and assign the result to v_new(idx)

        # Compute the distance between v_new and v

        # If the distance is smaller then tolerance terminate and return v_new

        # Else start over again


    pass

def _bin_search_opt(fun, sup_grid):
    """Find the maximum of a strictly concave function *fun* defined over a
    grid of n points *sup_grid*. Return maximum element and index at which the
    maximum was found.

    """
    # Initialize minimum and maximum indices of the support grid.
    idx_min = 0
    idx_max = len(sup_grid) - 1

    # While i_max - i_min > 2
    while (idx_max - idx_min) > 2:
        # Select two indices i_l=floor((i_min + i_max)/2) and i_u=i_l + 1
        idx_l = np.floor((idx_min + idx_max) / 2)
        idx_u = idx_l + 1

        # if f(x_iu) > f(x_il) set i_min=i_l.
        # else set i_max=i_u
        if fun[int(idx_u)] > fun[int(idx_l)]:
            idx_min = idx_l
        else:
            idx_max = idx_u

    # Return the largest element from f(x_imin), f(x_imax)
    if fun[int(idx_min)] > fun[int(idx_min + 1)]:
        return fun[int(idx_min)], sup_grid[int(idx_min)]
    elif fun[int(idx_min + 1)] > fun[int(idx_max)]:
        return fun[int(idx_min + 1)], sup_grid[int(idx_min + 1)]
    else:
        return fun[int(idx_max)], sup_grid[int(idx_max)]
