import unittest
import numpy as np
from quantecon.distributions import BetaBinomial
import my_mccall_model as mcm
from mccall_bellman_iteration import solve_mccall_model, McCallModel


class TestMyMcCall(unittest.TestCase):

    def test_utility_zero_cons(self):
        model = mcm.MyMcCallModel()
        self.assertEqual(model.get_utility(0), -10e6)

    def test_utility_log_case(self):
        model = mcm.MyMcCallModel(sigma=1.0)
        self.assertEqual(model.get_utility(1), 0)

    def test_utility_sigma2(self):
        model = mcm.MyMcCallModel()
        self.assertEqual(model.get_utility(1), 0)

    def test_solve_model_benchmark_quantecon(self):
        n = 60
        wage_grid = np.linspace(10, 20, n)
        a, b = 600, 400
        dist = BetaBinomial(n-1, a, b)
        prob_grid = dist.pdf()

        model = mcm.MyMcCallModel(wage_grid=wage_grid, prob_grid=prob_grid)
        model_sol = model.solve()
        true_mod = McCallModel()
        true_sol = solve_mccall_model(true_mod)

        np.testing.assert_array_almost_equal(
            model_sol[0],
            true_sol[0]
        )
        self.assertAlmostEqual(
            model_sol[1],
            true_sol[1]
        )

    def test_solve_model_uniform_wages(self):
        n = 100
        wage_grid = np.linspace(10, 20, n)
        prob_grid = np.array([1 / n] * n)
        model = mcm.MyMcCallModel(
            wage_grid=wage_grid, prob_grid=prob_grid
        )
        model_sol = model.solve()

        true_mod = McCallModel(
            w_vec=wage_grid, p_vec=prob_grid
        )
        true_sol = solve_mccall_model(true_mod)

        np.testing.assert_array_almost_equal(
            model_sol[0],
            true_sol[0]
        )
        self.assertAlmostEqual(
            model_sol[1],
            true_sol[1]
        )

    def test_compute_reservation_wage(self):
        n = 100
        wage_grid = np.linspace(0, 1, n)
        prob_grid = np.array([1 / n] * n)
        alpha = 0
        beta = .98
        gamma = 1
        b = .1
        util_spec = 'linear'
        model = mcm.MyMcCallModel(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            b=b,
            wage_grid=wage_grid,
            prob_grid=prob_grid,
            util_spec=util_spec
        )
        res_wage = model.compute_reservation_wage()
        self.assertAlmostEqual(res_wage, .827661731378)


if __name__ == '__main__':
    unittest.main()
