import unittest
import numpy as np
import matplotlib.pyplot as plt
from my_bewley_model import MyBewleyModel, tauchen, transistor


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
            r=.02, rho=.5, sigma=1, assets=np.array([0, 1, 2])
        )
        # bewley.solve_model()
        bewley._comp_stat_states()
        stationary = bewley.get_stat_states()
        true_sol = np.array([.5, .5])
        np.testing.assert_array_almost_equal(
            stationary, true_sol, decimal=3
        )

    def test_avg_labor_supply(self):
        bewley = MyBewleyModel(
            r=.02, rho=.5, sigma=1, assets=np.array([0, 1, 2])
        )
        bewley._comp_stat_states()
        bewley._comp_avg_labor_sup()
        self.assertAlmostEqual(bewley.get_avg_lab_sup(), .6)

    # def test_solve_visually(self):
    #     assets=np.linspace(0, 30, 100)
    #     bewley = MyBewleyModel(
    #         r=.02, rho=.5, sigma=1, assets=assets
    #     )
    #     v, pol = bewley._vfi()
    #
    #     plt.figure()
    #     plt.subplot(1, 2, 1)
    #     plt.plot(assets, v)
    #     plt.title('Value function')
    #
    #     plt.subplot(1, 2, 2)
    #     plt.plot(assets, pol)
    #     plt.title('Policy function')
    #
    #     plt.show()

    def test_transistor_3states_2shocks(self):
        policy = np.array([
            [0, 1],
            [1, 1],
            [1, 2]
        ])
        stoch_trans = np.array([
            [.7, .3],
            [.4, .6]
        ])
        true_sol = np.array([
            [.7, .3, 0, 0, 0, 0],
            [0, 0, .4, .6, 0, 0],
            [0, 0, .7, .3, 0, 0],
            [0, 0, .4, .6, 0, 0],
            [0, 0, .7, .3, 0, 0],
            [0, 0, 0, 0, .4, .6]
        ])
        trans = transistor(policy, stoch_trans)
        np.testing.assert_array_equal(
            trans, true_sol
        )

    def test_create_transition(self):
        bewley = MyBewleyModel(
            r=.02, rho=.5, sigma=1, assets=np.array([0, 1, 2])
        )
        bewley.pol_idx = np.array([
            [0, 1],
            [1, 1],
            [1, 2]
        ])
        bewley.stoch_trans = np.array([
            [.7, .3],
            [.4, .6]
        ])
        true_sol = np.array([
            [.7, .3, 0, 0, 0, 0],
            [0, 0, .4, .6, 0, 0],
            [0, 0, .7, .3, 0, 0],
            [0, 0, .4, .6, 0, 0],
            [0, 0, .7, .3, 0, 0],
            [0, 0, 0, 0, .4, .6]
        ])
        bewley._create_transition()
        np.testing.assert_array_equal(
            bewley.get_transition(), true_sol
        )

    def test_compute_stat_dist(self):
        bewley = MyBewleyModel(
            r=.02, rho=.5, sigma=1, assets=np.array([0, 1, 2])
        )
        bewley.pol_idx = np.array([
            [0, 1],
            [1, 1],
            [1, 2]
        ])
        bewley.stoch_trans = np.array([
            [.7, .3],
            [.4, .6]
        ])
        bewley._create_transition()
        bewley._compute_stat_dist()
        np.testing.assert_array_equal(
            bewley.get_stat_assets(), np.array([0, 1., 0])
        )

    def test_create_transition_large(self):
        bewley = MyBewleyModel(
            r=.02, rho=.5, sigma=1, assets=np.linspace(0, 30, 100), n_stoch=32
        )
        bewley._vfi()
        bewley._create_transition()
        print(np.sum(bewley.get_transition(), axis=1))


if __name__ == '__main__':
    unittest.main()
