"""
Compute reservation wage of a given instance of a McCallModel.
"""


import numpy as np
import scipy.integrate as integrate
# from my_mccall_model import MyMcCallModel


def log_logistics_pdf(x, alpha, beta):
    """
    Return the pdf of a log logistical distribution with shape parameters
    alpha and beta at position x.

    """

    return ((beta / alpha) * (x / alpha)**(beta - 1)) / \
        (1 + (x / alpha)**beta)**2


def log_logistics_quant(x, alpha, beta):
    """
    Return the inverse of the cumulative distribution function of a
    log-logistic with shape parameters alpha and bet distribution at point x.

    """

    return alpha * (x / (1 - x))**(1 / beta)

def unif_pdf(x, a, b):
    return 1/(b - a)

# Different iterative procedure to calculate the reservation wage_grid
def e_max_w(pdf, pdfargs, w_bar, w_up=1, w_down=0):
    pr_wbar = integrate.quad(
        lambda w: pdf(w, *pdfargs), w_down, w_bar
    )[0]
    exp_w_bar = integrate.quad(
        lambda w: w * pdf(w, *pdfargs), w_bar, w_up
    )[0]
    return w_bar * pr_wbar + exp_w_bar

def update_rw(w0, pdf, pdfargs, b, beta, w_up=1, w_down=0):
    e_wage_offer = e_max_w(pdf, pdfargs, w0, w_up, w_down)
    w_new = b * (1 - beta) + beta * e_wage_offer
    return w_new


def iter_res_w(w0, pdf, pdfargs, b, beta, w_up=1, w_down=0,
               tol=1e-5, max_iter=200000):
    count = 0
    error = tol + 1

    while error > tol:  # and count > max_iter:
        w_new = update_rw(w0, pdf, pdfargs, b, beta)
        error = np.abs(w_new - w0)
        w0 = w_new
        count += 1

    return w0

# example
w0 = 0
b = .4
beta = .96
res_wage = iter_res_w(
    w0=w0, pdf=unif_pdf, pdfargs=[0, 1], b=b, beta=beta, w_up=10
)
print(res_wage)

res_wage_log = iter_res_w(
    w0=w0, pdf=log_logistics_pdf, pdfargs=[1, 2], b=b, beta=beta,
    w_up=10, w_down=0
)
print(res_wage_log)
