"""
Set up a bewley model class to model the householod sector of an Aiyagari
style economy.

"""

import numpy as np
from scipy.stats import norm


class MyBewleyModel():
    """docstring for MyBewleyModel."""
    def __init__(self, r=.02):
        pass

    def _utility(self):
        pass

    def _solve(self):
        pass

    def _create_transition(self):
        pass

    def _compute_stat_dist(self):
        pass

    def _aggregates(self):
        pass


def tauchen(rho, sigma, n=2, sup_low=0.2, sup_up=1):
    """Implementation of the Tauchen algorithm to approximate an
    autoregressive process of order 1 by a Markov chain.
    Return the discrete vector of states and the transition probability
    matrix P of the Markov chain.
    """
    # 1. Choice of points
    # Set the first point equal to sup_low and the last point equal to
    # sup_up.
    # Set the grid for states as n linearly spaced points between the
    # largest and smallest points as defined above.
    # Calculate the distance between two successive points in the state
    # space and store this distance in variable dist.
    states, dist = np.linspace(sup_low, sup_up, n, retstep=True)

    # 2. Compute transition matrix
    # Initialize transition matrix of size nxn.
    transit = np.array([[np.nan] * n] * n)

    for idx in range(n):
        # Compute first column of the transition matrix
        # P(x_i, x_0) = F(x_0 - rho*x_i + d/2) for all i
        transit[idx, 0] = norm.cdf(states[0] - rho * states[idx] + dist/2)

        # Compute last column of the transition matrix
        # P(x_i, x_n-1) = 1 - F(x_n-1 - rho*x_i - d/2)
        transit[idx, n-1] = 1 - norm.cdf(
            states[n-1] - rho * states[idx] - dist / 2
        )

    # Compute remaining columns according to
    # P(x_i, x_j) = F(x_j - rho*x_i + d/2) - F(x_j - rho*x_i - d/2)
    if n > 2:
        for row_idx in range(0, n):
            for col_idx in range(1, n - 1):
                transit[row_idx, col_idx] = norm.cdf(
                    states[col_idx] - rho * states[row_idx] + dist/2
                ) - \
                norm.cdf(
                    states[col_idx] - rho * states[row_idx] - dist/2
                )

    # Return state vector and transition matrix
    return transit, states

def transistor(pol, transit):
    """Compute transition matrix of state space for stochastic shocks and
    assets.

    """
    pass

def value_iter(asset_grid, state_grid, state_trans):
    """Implement value function iteration. Return value function and policy
    function.

    """

    # Let n be the length of the asset grid and m the length of the state
    # grid. For m states define nX1 vectors v_j whose ith rows are determined
    # by v_j(i) = v(a_i, s_j) , j=1,...,m, i=1,...,n.

    # Create a vector of nX1 ones.

    # Define j nXn matrices R_j whose (i, h) elements are given by
    # R_j(i, h) = u((r + 1) a_i + w*s_j - a_h) , i=1,...,n , h=1,...,n

    # Define the Bellman operator T([v_1, v_2]) that maps a pair of nX1 vectors
    # [v_1, v_2] into a pair of nX1 vectors [tv_1, tv_2] such that:
    # tv_j(i) = max_h {R_j(i, h) + beta*sum_k(Pi_{jk}*v_k(h))}

    # Iterate over the bellman operator until convergence
    pass
