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

    def _tauchen(self, rho, sigma, n=2, sup_low=0.2, sup_up=1):
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
