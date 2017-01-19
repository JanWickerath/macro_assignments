import unittest
import numpy as np
from my_bewley_model import tauchen


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


if __name__ == '__main__':
    unittest.main()
