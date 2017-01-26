from my_bewley_model import MyBewleyModel, aggregate_demand
import numpy as np
import matplotlib.pyplot as plt

# Set parameters
r = .01
rho = .9
sigma = 2
assets = np.linspace(0, 30, 100)
beta = .95
n_stoch = 2
alpha = .5
delta = .05

# Create an instance of the Bewley model with given parameterization
bewley_small = MyBewleyModel(
    r, rho, sigma, assets, beta, n_stoch, alpha=alpha, delta=delta
)

# Solve the model and store results internally
bewley_small.solve_model()

# Do the same of a larger model
# Set parameters
n_stoch_large = 32

# Create an instance of the Bewley model with given parameterization
bewley_large = MyBewleyModel(
    r, rho, sigma, assets, beta, n_stoch_large, alpha=alpha, delta=delta
)

# Solve the model and store results internally
bewley_large.solve_model()



# Plot distribution of assets
fig = plt.figure()
plt.subplot(1, 2, 1)
plt.plot(assets, bewley_small.get_stat_assets())
plt.title('2 Stochastic states')

plt.subplot(1, 2, 2)
plt.plot(assets, bewley_large.get_stat_assets())
plt.title('32 Stochastic states')

plt.tight_layout()

fig.savefig('figures/asset_distribution.pdf')
