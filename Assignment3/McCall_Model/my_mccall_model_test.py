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
        true_mod = McCallModel(
            w_vec=np.linspace(0, 1, 100), p_vec=np.array([.01] * 100)
        )
        # self.assertEqual(
        #     model.solve(),
        #     solve_mccall_model(true_mod)
        # )
        np.testing.assert_array_almost_equal

if __name__ == '__main__':
    unittest.main()
