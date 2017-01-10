"""
My implementation of a McCall model class.

"""

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
