"""
Compute reservation wage of a given instance of a McCallModel.

"""


import numpy as np
from my_mccall_model import MyMcCallModel

def log_logistics_pdf(x, alpha, beta):
    """
    Return the pdf of a log logistical distribution with shape parameters
    alpha and beta at position x.

    """

    return ((beta / alpha) * (x / alpha)**(beta - 1)) / \
        (1 + (x / alpha)**beta)**2


# Create instance of a model with uniform wage distribution
alpha = 0                       # No risk of losing a job
beta = 0.98                     # Standard discount factor
gamma = 1                       # Get a job offer every period
b = 0.1                         # Unemployment benefits
util_spec = 'linear'            # utility u(y) = y
n = 100
wage_grid = np.linspace(0, 1, n)
prob_grid = np.array([1/n] * n)

model_unif = MyMcCallModel(
    alpha=alpha,
    beta=beta,
    gamma=gamma,
    b=b,
    wage_grid=wage_grid,
    prob_grid=prob_grid,
    util_spec=util_spec
)

w_res_unif = model_unif.compute_reservation_wage()
print(w_res_unif)


# Create an instance of the model with log-logistic wage distribution
alph_shape = 1
beta_shape = 20
wage_grid_log = np.linspace(0, 10, n)
prob_grid_log = log_logistics_pdf(wage_grid_log, alph_shape, beta_shape)
print(sum(wage_grid_log * prob_grid_log))
model_log = MyMcCallModel(
    wage_grid=wage_grid_log,
    prob_grid=prob_grid_log
)

w_res_log = model_log.compute_reservation_wage()
print(w_res_log)
