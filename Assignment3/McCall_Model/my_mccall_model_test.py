import unittest
import my_mccall_model as mcm


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


if __name__ == '__main__':
    unittest.main()
