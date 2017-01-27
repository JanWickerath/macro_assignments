from my_bewley_model import MyBewleyModel, aggregate_demand
import numpy as np
import matplotlib.pyplot as plt


def plot_asset_dist():
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


# Calculate Asset supply and demand for different interest rates.
def supply_demand_plot():
    # Initialize interest rate grid supply and demand vectors
    r_grid = np.linspace(-0.04, 0.06, 100)
    supply = np.array([np.nan] * len(r_grid))
    demand = np.array([np.nan] * len(r_grid))

    # Initialize parameters
    rho = .9
    sigma = 2
    assets = np.linspace(0, 30, 100)
    beta = .95
    n_stoch = 2
    alpha = .5
    delta = .05

    # Loop over interest rate grid and fill supply and demand vectors
    # appropriately
    for idx in range(len(r_grid)):
        r = r_grid[idx]
        # Create an instance of the Bewley model with given
        # parameterization
        bewley = MyBewleyModel(
            r, rho, sigma, assets, beta, n_stoch, alpha=alpha, delta=delta
        )
        bewley.solve_model()

        supply[idx] = bewley.get_asset_supply()

        # Create aggregate demand
        labor_sup = bewley.get_avg_lab_sup()
        demand[idx] = aggregate_demand(r, labor_sup, alpha, delta)

    # Plot results
    fig = plt.figure()
    plt.plot(supply, r_grid, label='Supply')
    plt.plot(demand, r_grid, label='Demand')

    plt.legend()
    plt.xlim(xmax=30)
    plt.tight_layout()
    fig.savefig('figures/supply_demand_plot.pdf')
    plt.show()

# Solve the exercise.
plot_asset_dist()
supply_demand_plot()
