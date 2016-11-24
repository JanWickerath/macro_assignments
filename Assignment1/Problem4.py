"""
Solution to Problem 4: Find optimal linear taxation scheme in a world with
under Greenwood, Hercowitz, Huffman preferences and wage heterogeneity among
households.

"""

# Import scientific libraries
from scipy.optimize import fsolve
import scipy.integrate as integrate
import numpy.random.lognormal as np.lognormal

class GovProblem():
    """
    Definition of the governments problem plus several methods.

    """

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

def hours(wage, tax, psi):
    return( ((1 - tax) * wage)**(1 / psi))
def cons(wage, tax, psi, trans):
    return( ((1 - tax) * wage)**((1 + psi) / psi) + trans)
