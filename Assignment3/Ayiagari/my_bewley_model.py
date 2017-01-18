"""
Set up a bewley model class to model the householod sector of an Aiyagari
style economy.

"""

# import numpy as np


class MyBewleyModel():
    """docstring for MyBewleyModel."""
    def __init__(self, r):
        pass

    def _tauchen(self, rho, sigma, n=2, sup_low=0.2, sup_up=1):
        """Implementation of the Tauchen algorithm to approximate an
        autoregressive process of order 1 by a Markov chain.
        Return the discrete vector of states and the transition probability
        matrix P of the Markov chain.
        """
        # 1. Choice of points
        # Compute unconditional standard deviation of dependent variable

        # Compute the largest point as multiple m of this standard deviation

        # Set the first point as the negative of the largest point

        # Set the grid for states as n linearly spaced points between the
        # largest and smallest points calculated as above.

        # Calculate the distance between two successive points in the state
        # space and store this distance in variable d.

        # 2. Compute transition matrix
        # Initialize transition matrix of size nxn.

        # Compute first column of the transition matrix
        # P(x_i, x_0) = F(x_0 - rho*x_i + d/2) for all i

        # Compute last column of the transition matrix
        # P(x_i, x_n-1) = 1 - F(x_n-1 - rho*x_i - d/2)

        # Compute remaining columns according to
        # P(x_i, x_j) = F(x_j - rho*x_i + d/2) - F(x_j - rho*x_i - d/2)

        # Return state vector and transition matrix

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
