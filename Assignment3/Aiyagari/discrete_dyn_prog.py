"""Implementation of different algorithms to solve discrete dynamic programming
problems. Description of algorithms are mainly taken from the textbook by Heer
and Maussner (2009).

"""

import numpy as np
import matplotlib.pyplot as plt

def value_iter_deter(state_grid, reward_fun, beta, prod=.5, tol=1e-8):
    """Docstring for deterministic value function iteration."""
    # Denote length of the state grid with n
    n = len(state_grid)
    # Initialize first guess of value function to vector of zeros with length
    # n.
    v = np.zeros(n)
    pol = np.zeros(n)

    # Initialize nXn array *util_mat* to compute utility levels for different
    # state-choice combinations.
    util_mat = np.array([[np.nan] * n] * n)

    # Loop over rows of util_mat
    for row_idx in range(n):
        # Loop over cols of util_mat
        for col_idx in range(n):
            # Calculate consumption if state is state[row] and choice is
            # state[col]
            cons = prod * state_grid[row_idx]**(.5) - state_grid[col_idx]

            # Set
            # util_mat(row,col)=reward_fun(state_grid(row),state_grid(col))
            util_mat[row_idx, col_idx] = reward_fun(cons)

    # Initialize distance to infinity
    dist = np.inf

    # While distance is larger then tolerance
    while dist > tol:

        # Initialize vector of new values v_new of length n.
        v_new = np.array([np.nan] * n)

        # Loop over indices (idx) of v_new
        for idx in range(n):

            # Compute
            # _bin_search_opt(util_mat[:, idx] + beta * v[idx]
            # and assign the result to v_new(idx)
            v_new[idx], pol[idx] = _bin_search_opt(util_mat[:, idx] + beta *
                                                   v[idx], k)

        # Compute the distance between v_new and v
        dist = np.linalg.norm(v_new - v)

        v = v_new

    return v_new, pol


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


k = np.linspace(1/100, 1/9, 10)
z = .5
beta = .5
def util(cons):
    if cons > 0:
        return np.log(cons)
    else:
        return -np.inf

value, pol = value_iter_deter(state_grid=k, reward_fun=util, beta=beta, prod=z)

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(k, value)
plt.title('Value function')

plt.subplot(1, 2, 2)
plt.plot(k, pol)
plt.title('policy function')

plt.show()
