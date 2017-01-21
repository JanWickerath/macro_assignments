"""Test suite for discrete dynamic programming algorithms."""

import unittest
import numpy as np
import discrete_dyn_prog as ddp


class TestMyBewleyModel(unittest.TestCase):

    def test_bin_search_opt_linear_fun_inc(self):
        x = np.linspace(0, 10, 100)
        def test_fun(x):
            return 2*x
        fmax = 20
        f_imax = 10
        self.assertEqual(
            ddp._bin_search_opt(test_fun, x), (fmax, f_imax)
        )

    def test_bin_search_opt_linear_fun_dec(self):
        x = np.linspace(0, 10, 100)
        def test_fun(x):
            return -2*x
        fmax = 0
        f_imax = 0
        self.assertEqual(
            ddp._bin_search_opt(test_fun, x), (fmax, f_imax)
        )

    def test_bin_search_opt_quadr_fun(self):
        x = np.linspace(0, 10, 101)
        def test_fun(x):
            return -(x - 2)**2

        fmax = 0
        f_imax = 2
        self.assertEqual(
            ddp._bin_search_opt(test_fun, x), (fmax, f_imax)
        )


if __name__ == '__main__':
    unittest.main()
