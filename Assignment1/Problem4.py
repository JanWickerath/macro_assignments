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
import matplotlib.pyplot as plt


# Define helper functions
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

def foc(gov_policies, psi, sig, start=0, end=10):
    # Initialize local variables for government policies for better
    # readability.
    tax = gov_policies[0]
    trans = gov_policies[1]
    result = []

    # Compute different parts of first FOC (respect tax rate) and combine
    part_a = integrate.quad(
        lambda w: dell_u_tax(w, cons(w, tax, psi, trans),
                             hours(w, tax, psi), psi, tax) * lognorm.pdf(
                                 w, s=sig, scale=np.exp(-sig**2 / 2)),
        0, 10                   # set integration borders
    )[0]
    part_b = integrate.quad(
        lambda w: w * hours(w, tax, psi), start, end
    )[0]

    # Compute first part of the second FOC (respect transfers)
    part_c = integrate.quad(
        lambda w: lognorm.pdf(w, s=sig, scale=np.exp(-sig**2 / 2)) *
        dell_u_trans(cons(w, tax, psi, trans), hours(w, tax, psi), psi),
        start, end
    )[0]

    # Store first foc in results vector
    result.append(part_a + part_c * part_b)

    # Compute budget constraint
    bud_const = trans - integrate.quad(
        lambda w: tax * w * hours(w, tax, psi), start, end
    )[0]
    result.append(bud_const)

    return result

print(foc([.5, .5], 2, 0.5))


## Fix exogenous parameters
psi = 2
sig_grid = [0.1, 0.3, 0.5]
x0 = [.5, 2]
n_obs = 10000

# Initialize policy as empty list
policy_grid = []
opt_tax = []
opt_trans = []
wage_grid = []
hours_grid = []
cons_grid = []

counter = 0
for sig in sig_grid:

    # Compute optimal redistribution scheme of the government
    policy_grid.append(fsolve(
        lambda policies: foc(policies, psi, sig, start=0, end=10),
        x0=x0
    ))
    opt_tax.append(policy_grid[counter][0])
    opt_trans.append(policy_grid[counter][1])

    ## Simulate distribution of wages and compute the distribution of
    ## consumption and hours worked given the optimal redistribution scheme
    ## calculated above
    wage_grid.append(
        lognorm.rvs(s=sig, scale=np.exp(- sig**2 / 2),
                                     size=n_obs)
    )
    print("Check the mean ", np.mean(wage_grid[counter]), " approx. 1???")

    hours_grid.append(hours(wage_grid[counter], opt_tax[counter], psi))
    cons_grid.append(cons(wage_grid[counter], opt_tax[counter], psi,
                          opt_trans[counter]))

    counter += 1


## Plot distributions of optimal wages, consumption and hours worked

fig = plt.figure()
plt.subplot(3, 3, 1)
plt.hist(wage_grid[0], bins=100)
plt.title("Wage distribution, sigma = .1")

plt.subplot(3, 3, 2)
plt.hist(hours_grid[0], bins=100)
plt.title("Hours worked distribution, sigma = .1")

plt.subplot(3, 3, 3)
plt.hist(cons_grid[0], bins=100)
plt.title("Consumption distribution, sigma = .1")

plt.subplot(3, 3, 4)
plt.hist(wage_grid[1], bins=100)
plt.title("Wage distribution, sigma = .3")

plt.subplot(3, 3, 5)
plt.hist(hours_grid[1], bins=100)
plt.title("Hours worked distribution, sigma = .3")

plt.subplot(3, 3, 6)
plt.hist(cons_grid[1], bins=100)
plt.title("Consumption distribution, sigma = .3")

plt.subplot(3, 3, 7)
plt.hist(wage_grid[2], bins=100)
plt.title("Wage distribution, sigma = .5")

plt.subplot(3, 3, 8)
plt.hist(hours_grid[2], bins=100)
plt.title("Hours worked distribution, sigma = .5")

plt.subplot(3, 3, 9)
plt.hist(cons_grid[2], bins=100)
plt.title("Consumption distribution, sigma = .5")

plt.tight_layout()

fig.savefig(
    'tex/figures/distribs.pdf'
)

## Plot optimal policy vs sigma
fig = plt.figure()
plt.subplot(1, 2, 1)
plt.plot(sig_grid, opt_tax)
plt.title("optimal tax rate")

plt.subplot(1, 2, 2)
plt.plot(sig_grid, opt_trans)
plt.title("Optimal transfers")

fig.savefig(
    'tex/figures/opt_policy.pdf'
)




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
