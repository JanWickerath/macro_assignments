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
