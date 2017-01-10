import unittest
import numpy as np
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
        model = mcm.MyMcCallModel()
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



if __name__ == '__main__':
    unittest.main()
