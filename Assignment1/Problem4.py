"""
Solution to Problem 4: Find optimal linear taxation scheme in a world with
under Greenwood, Hercowitz, Huffman preferences and wage heterogeneity among
households.

"""

# Import scientific libraries
from scipy.optimize import fsolve
import scipy.integrate as integrate
from scipy.stats import lognorm
import numpy as np


def hours(wage, tax, psi):
    return( ((1 - tax) * wage)**(1 / psi))

def cons(wage, tax, psi, trans):
    return( ((1 - tax) * wage)**((1 + psi) / psi) + trans)

def dell_u_tax(wage, cons, hours, psi, tax):
    numerator = wage * hours
    denom = cons - (hours**(1 + psi) / (1 + psi))
    bracks = ((1 - tax) * wage)**((1 - psi) / psi) - (1 + psi) / psi
    return(numerator / denom * bracks)

def dell_u_trans(cons, hours, psi):
    return(1 / (cons - hours**(1 + psi) / (1 + psi)))

def foc(gov_policies, psi, sig):
    tax = gov_policies[0]
    trans = gov_policies[1]
    part_a = integrate.quad(
        lambda w: dell_u_tax(w, cons(w, tax, psi, trans),
                             hours(w, tax, psi), psi, tax) * lognorm.pdf(
                                 w, s=sig, scale=np.exp(-sig**2 / 2)),
        0, 10                   # set integration borders
    )

    return part_a

print(foc([0.5, 0.5], 2, 0.5))


# class GovProblem():
#     """
#     Definition of the governments problem plus several methods.

#     """

    # def __init__(self):
    #     """
    #     Create an instance of the government with a given social welfare
    #     fucntion.
    #     """

    #     self.__swf__ = swf

    # def set_swf(self, swf):
    #     """
    #     Method to set the governments social welfare function.

    #     """
    #     self.__swf__ = swf

    # def foc(self, policies, wage_dist, psi):
    #     """
    #     First order condition of the government given the households optimal
    #     policy decisions, a wage distribution and parameter psi.

    #     """
    #     tax = policies[0]
    #     trans = policies[1]
    #     def hours(wage, tax, psi):
    #         return( ((1 - tax) * wage)**(1 / psi))
    #     def cons(wage, tax, psi, trans):
    #         return( ((1 - tax) * wage)**((1 + psi) / psi) + trans)
