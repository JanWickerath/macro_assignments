"""
My implementation of a McCall model class.

"""

import numpy as np


# Definition of my McCall model class
class MyMcCallModel:
    """
    Model class that represents a McCall job search model. Stores the
    parameterization of the model and provides methods to solve the model and
    compute the reservation wage.

    """

    def __init__(self,
                 alpha=0,
                 beta=.98,
                 gamma=0.7,
                 b=6.0,
                 sigma=2.0,
                 wage_grid=None,
                 prob_grid=None):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.b = b
        self.sigma = sigma
        self.wage_grid, self.prob_grid = wage_grid, prob_grid

        # Add uniformly distributed wages in case wage_grid is not provided
        if wage_grid is None:
            n = 100
            self.wage_grid = np.linspace(0, 1, n)
            self.prob_grid = np.array([1/n] * n)
        else:
            self.wage_grid = wage_grid
            self.prob_grid = prob_grid


    def _utility(self, cons, sigma):
        """
        CRRA utility function.

        """
        if cons <= 0:
            return -10e6
        elif sigma == 1:
            return np.log(cons)
        else:
            return (cons**(1 - sigma) - 1) / (1 - sigma)

    def get_utility(self, cons):
        """
        Return the utility of the agent in the current instance of the model
        for a given consumption level.

        """
        return self._utility(cons, self.sigma)


    def _update_bellman(self, V_e, V_u):
        """
        Update the Bellman equations of the McCall job search model.

        """
        V_e_new = np.zeros(len(V_e))

        for w_idx, wage in enumerate(self.wage_grid):
            V_e_new[w_idx] = self.get_utility(self.wage_grid[w_idx]) + \
                             self.beta * ((1 - self.alpha) * V_e[w_idx] +
                                          self.alpha * V_u)

        V_u_new = self.get_utility(self.b) + \
                  self.beta * (1 - self.gamma) * V_u + \
                  self.beta * self.gamma * \
                  np.sum(np.maximum(V_e, V_u) * self.prob_grid)

        return V_e_new, V_u_new
