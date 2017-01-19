import unittest
import numpy as np
from my_bewley_model import MyBewleyModel, tauchen


class TestMyBewleyModel(unittest.TestCase):

    def test_tauchen_len2(self):
        transit = tauchen(rho=.5, sigma=1)[0]
        true_sol = np.array([[.6915, .3085], [.5398, .4602]])
        np.testing.assert_array_almost_equal(
            transit, true_sol, decimal=3
        )

    def test_tauchen_len3_rowsums(self):
        transit = tauchen(rho=.5, sigma=1, n=3)[0]
        row_sums = []
        for idx in range(len(transit)):
            row_sums.append(sum(transit[idx]))
        self.assertEqual(row_sums, [1, 1, 1])

    def test_stationary_stoch_states(self):
        bewley = MyBewleyModel(
            r=.02, rho=.5, sigma=1, n_stoch=2, nsup_low=.2, nsup_up=1
        )
        stationary = bewley.get_stat_states()
        true_sol = np.array([.5, .5])
        np.testing.assert_array_almost_equal(
            stationary, true_sol, decimal=3
        )


if __name__ == '__main__':
    unittest.main()
