"""
Set up a bewley model class to model the householod sector of an Aiyagari
style economy.

"""

import numpy as np
from scipy.stats import norm


class MyBewleyModel():
    """docstring for MyBewleyModel."""
    def __init__(self, r, rho, sigma, assets, beta=.98,
                 n_stoch=2, nsup_low=.2, nsup_up=1, alpha=.5, delta=.05):
        self.r = r
        self.stoch_trans, self.stoch_states = tauchen(
            rho, sigma, n=n_stoch, sup_low=nsup_low, sup_up=nsup_up
        )
        self.state_dist = []
        self.avg_labor = None

        self.assets = assets
        self.beta = beta
        self.alpha = alpha
        self.delta = delta

        # Initialize vfi and pol
        self.val_fun = np.zeros([len(assets), n_stoch])
        self.pol_fun = np.zeros([len(assets), n_stoch])
        self.pol_idx = np.zeros([len(assets), n_stoch])

        self.state_trans = np.zeros([len(assets)*n_stoch, len(assets)*n_stoch])

    def _comp_stat_states(self):
        """
        Return the stationary distribution of stochastic states in this model
        instance.
        Compute the stationary distribution of the stochastic states computed
        by the eigenvector of the stochastic transition matrix
        *self.stoch_trans*.

        """
        eig_val, eig_vec = np.linalg.eig(np.transpose(self.stoch_trans))
        self.state_dist = eig_vec[:, 0]/sum(eig_vec[:, 0])

    def get_stat_states(self):
        return self.state_dist

    def _comp_avg_labor_sup(self):
        """Store and return the average labor supply of the economy."""

        self.avg_labor = sum(self.stoch_states * self.state_dist)
        self.wage = (1 - self.alpha) * (self.alpha / (self.r + self.delta))**(
            self.alpha/(1 - self.alpha))

    def get_avg_lab_sup(self):
        return self.avg_labor

    def _utility(self, cons):
        return np.log(cons)

    def _vfi(self, tol=1e-5):
        """Implement value function iteration approach to solve for the value
        function and the policy function of the household.

        """
        # Initialize v to n-by-m array where n is the length of the asset grid
        # and m the length of the stochastic grid.
        n = len(self.assets)
        m = len(self.stoch_states)
        v = np.zeros([n, m])

        # Compute consumption matrix and utility matrix
        a_prime, prod, a = np.meshgrid(
            self.assets,
            self.stoch_states,
            self.assets
        )
        cons_mat = self.wage * prod + (1 + self.r) * a - a_prime

        util_mat = self._utility(cons_mat)
        util_mat[cons_mat <= 0] = -np.inf

        # Iterate over value function
        dist = 1

        while dist > tol:
            # Compute the expected value
            ex_val = v @ np.transpose(self.stoch_trans)

            # Initialize v_new and policy function
            v_new = np.zeros([n, m])
            pol = np.zeros([n, m])
            pol_idx = np.zeros([n, m])

            for idx in range(m):
                v_new[:, idx] = np.amax(
                    util_mat[idx, :, :] +
                    self.beta * np.array([ex_val[:, idx]] * n), 1
                )
                pol[:, idx] = self.assets[
                    np.argmax(
                    util_mat[idx, :, :] +
                    self.beta * np.array([ex_val[:, idx]] * n), 1
                    )
                ]
                pol_idx[:, idx] = np.argmax(
                    util_mat[idx, :, :] +
                    self.beta * np.array([ex_val[:, idx]] * n), 1
                )

            dist = np.linalg.norm(v_new - v)

            v = np.copy(v_new)

        self.val_fun = v
        self.pol_fun = pol
        self.pol_idx = pol_idx

    def get_pol(self):
        return self.pol_fun, self.pol_idx

    def get_val(self):
        return self.val_fun

    def _create_transition(self):
        """Compute transition matrix of state space for stochastic shocks and
        assets.

        """
        # Initialize transition matrix of zeros of size n*m-by-n*m
        n = len(self.pol_idx[:, 0])
        m = len(self.pol_idx[0, :])

        # Loop over all rows in the trainsition matrix
        for row_idx in range(n*m):
            # Check the optimal policy. To do so go to the policy function at
            # row position row_idx//m (m denotes the number of stochastic
            # states and '//' is integer division) and column position
            # row_idx%m. Store the policy (in terms of the index in the choice
            # vector that returns the optimal choice) as opt_pol_idx.
            opt_pol_idx = self.pol_idx[row_idx//m, row_idx%m]
            # Fill the current row from position opt_col_idx*m to
            # (opt_col_idx*m + m - 1) with values from stoch_trans[row_idx%m,
            # :]
            self.state_trans[
                row_idx, (opt_pol_idx*m):(opt_pol_idx*m + m)
            ] = self.stoch_trans[row_idx%m, :]

    def get_transition(self):
        return self.state_trans

    def _compute_stat_dist(self):
        eig_val, eig_vec = np.linalg.eig(np.transpose(self.state_trans))
        # Select unit eigenvector and normalize so that it adds up to 1
        inter_stat = eig_vec[:, np.isclose(eig_val, 1)] / \
            sum(eig_vec[:, np.isclose(eig_val, 1)])
        # Reshape vector to a matrix, so that sum over the columns will give
        # the stationary asset distribution.
        inter_stat = inter_stat.reshape((len(self.assets),
                                         len(self.stoch_states)))
        # Sum up over columns
        self.stat_assets = np.sum(inter_stat, axis=1)

    def get_stat_assets(self):
        return self.stat_assets

    def _aggregates(self):
        self.asset_supply = sum(self.stat_assets * self.assets)

    def get_asset_supply(self):
        return self.asset_supply

    def solve_model(self):
        self._comp_stat_states()
        self._comp_avg_labor_sup()
        self._vfi()
        self._create_transition()
        self._compute_stat_dist()
        self._aggregates()


def tauchen(rho, sigma, n=2, sup_low=0.2, sup_up=1):
    """Implementation of the Tauchen algorithm to approximate an
    autoregressive process of order 1 by a Markov chain.
    Return the discrete vector of states and the transition probability
    matrix P of the Markov chain.
    """
    # 1. Choice of points
    # Set the first point equal to sup_low and the last point equal to
    # sup_up.
    # Set the grid for states as n linearly spaced points between the
    # largest and smallest points as defined above.
    # Calculate the distance between two successive points in the state
    # space and store this distance in variable dist.
    states, dist = np.linspace(sup_low, sup_up, n, retstep=True)

    # 2. Compute transition matrix
    # Initialize transition matrix of size nxn.
    transit = np.array([[np.nan] * n] * n)

    for idx in range(n):
        # Compute first column of the transition matrix
        # P(x_i, x_0) = F(x_0 - rho*x_i + d/2) for all i
        transit[idx, 0] = norm.cdf(states[0] - rho * states[idx] + dist/2,
                                   scale=sigma)

        # Compute last column of the transition matrix
        # P(x_i, x_n-1) = 1 - F(x_n-1 - rho*x_i - d/2)
        transit[idx, n-1] = 1 - norm.cdf(
            states[n-1] - rho * states[idx] - dist / 2,
            scale=sigma
        )

    # Compute remaining columns according to
    # P(x_i, x_j) = F(x_j - rho*x_i + d/2) - F(x_j - rho*x_i - d/2)
    if n > 2:
        for row_idx in range(0, n):
            for col_idx in range(1, n - 1):
                transit[row_idx, col_idx] = norm.cdf(
                    states[col_idx] - rho * states[row_idx] + dist/2,
                    scale=sigma
                ) - \
                norm.cdf(
                    states[col_idx] - rho * states[row_idx] - dist/2,
                    scale=sigma
                )

    # Return state vector and transition matrix
    return transit, states

def transistor(pol_idx, transit):
    """Compute transition matrix of state space for stochastic shocks and
    assets.

    """
    # Initialize transition matrix of zeros of size n*m-by-n*m
    n = len(pol_idx[:, 0])
    m = len(pol_idx[0, :])
    state_trans = np.zeros([n*m, n*m])

    # Loop over all rows in the trainsition matrix
    for row_idx in range(n*m):
        # Check the optimal policy. To do so go to the policy function at row
        # position row_idx//m (m denotes the number of stochastic states and
        # '//' is integer division) and column position row_idx%m. Store the
        # policy (in terms of the index in the choice vector that returns the
        # optimal choice) as opt_pol_idx.
        opt_pol_idx = pol_idx[row_idx//m, row_idx%m]
        # Fill the current row from position opt_col_idx*m to (opt_col_idx*m +
        # m - 1) with values from stoch_trans[row_idx%m, :]
        state_trans[
            row_idx, (opt_pol_idx*m):(opt_pol_idx*m + m)
        ] = transit[row_idx%m, :]

    return state_trans

def aggregate_demand(r, labor, alpha, delta):
    return labor * (alpha / (r + delta))**(1/(1 - alpha))
