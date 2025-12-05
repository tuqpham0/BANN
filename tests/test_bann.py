import unittest
import bann
import numpy as np

"""
def k_search(
    np.ndarray[double, ndim=2] data,
    np.ndarray[double, ndim=2] query,
    int K = 1, double EPS = 0, str div = 'kl') -> np.ndarray:

def bhaus(
    np.ndarray[double, ndim=2] data,
    np.ndarray[double, ndim=2] query,
    double EPS = 0, str div = 'kl') -> double:
"""

class Test_bann(unittest.TestCase):
    def setUp(self):
        self.data = np.array([[.1], [.6]])
        self.query = np.array([[.3]])

        self.dim_data = np.array(
            [[0.4296287,  0.54582694, 0.02454436],
                [0.6776321,  0.18751646, 0.13485144],
                [0.0974914,  0.46193274, 0.44057586],
                [0.63578628, 0.26316086, 0.10105286],
                [0.22945766, 0.13604054, 0.6345018 ],
                [0.63722053, 0.18824259, 0.17453688],
                [0.50396997, 0.31401086, 0.18201917],
                [0.49469846, 0.29576555, 0.20953599],
                [0.40426814, 0.17074074, 0.42499112],
                [0.10986894, 0.54545651, 0.34467455],
                [0.31036794, 0.22884235, 0.46078971],
                [0.23513868, 0.30733254, 0.45752877],
                [0.3248107,  0.46709761, 0.20809169],
                [0.0474218,  0.27828197, 0.67429623],
                [0.00363314, 0.74667265, 0.24969421],
                [0.282168,   0.36771445, 0.35011755],
                [0.24839865, 0.09348735, 0.658114  ],
                [0.16890292, 0.76215108, 0.068946  ],
                [0.37772481, 0.09690062, 0.52537457],
                [0.52245922, 0.04822038, 0.42932041],
                [0.56000997, 0.28056901, 0.15942103],
                [0.43237275, 0.33424503, 0.23338222],
                [0.13167681, 0.11194199, 0.7563812 ],
                [0.50245531, 0.29345491, 0.20408978],
                [0.04321371, 0.30805602, 0.64873027],
                [0.26807204, 0.39285044, 0.33907752],
                [0.13126271, 0.50189492, 0.36684237],
                [0.66105254, 0.16892868, 0.17001878],
                [0.06683453, 0.42714443, 0.50602103],
                [0.33279741, 0.24865218, 0.41855041],
                [0.40386816, 0.35425416, 0.24187769],
                [0.39062708, 0.1732431,  0.43612982],
                [0.68608459, 0.13843772, 0.17547768],
                [0.05874155, 0.50327138, 0.43798707],
                [0.33924437, 0.26573512, 0.39502051],
                [0.098073, 0.54351756, 0.35840944],
                [0.6761732, 0.0376295, 0.28619731],
                [0.43036298, 0.52050126, 0.04913575],
                [0.71745816, 0.23047905, 0.05206279],
                [0.32845324, 0.21766456, 0.4538822 ]]
        )
        self.dim_query = np.array(
            [[0.3251311,  0.33769861, 0.33717029],
                [0.47371353, 0.01794854, 0.50833792],
                [0.47749099, 0.03905358, 0.48345542],
                [0.42256473, 0.20426618, 0.37316909],
                [0.08729165, 0.423712,   0.48899635],
                [0.20434417, 0.36045547, 0.43520036],
                [0.09446384, 0.43386852, 0.47166764],
                [0.20749814, 0.22021566, 0.5722862 ],
                [0.4715353,  0.16871541, 0.35974929],
                [0.50778605, 0.19006153, 0.30215242]]
        )

    
    def test_nn_out_dims(self):
        print("Testing output dimensions...")
        for k in range(1, 100):
            self.assertTrue(bann.k_search(np.random.rand(200, 15), np.random.rand(10, 15), k, 0, 'kl').shape == (10,k))

    def test_nn_basics(self):
        print("Testing basic nearest neighbor searches...")
        # Query 1 point into a 2 point set for the divergences.
        self.assertTrue(np.array_equal(bann.k_search(self.data, self.query, 1, 0, 'kl'), np.array([[1]])))
        self.assertTrue(np.array_equal(bann.k_search(self.data, self.query, 1, 0, 'dkl'), np.array([[0]])))

        self.assertTrue(np.array_equal(bann.k_search(self.data, self.query, 1, 0, 'is'), np.array([[1]])))
        self.assertTrue(np.array_equal(bann.k_search(self.data, self.query, 1, 0, 'dis'), np.array([[1]])))
        self.assertTrue(np.array_equal(bann.k_search(self.data, self.query, 1, 0, 'se'), np.array([[0]])))
    
    def test_knn(self):
        print("Testing k-nearest neighbor searches in higher dimensions...")
        # Checks if the k-nearest neighbor searches are correct
        self.assertTrue(np.array_equal(
            bann.k_search(self.dim_data, self.dim_query, 3, 0, 'kl'),
            np.array([[15, 25, 34],
                [19, 18, 36],
                [19, 18,  8],
                [ 8, 31, 34],
                [28,  2, 33],
                [11, 15, 25],
                [ 2, 28, 33],
                [ 4, 11, 10],
                [ 8, 31, 39],
                [ 8,  7, 31]])))
        self.assertTrue(np.array_equal(
            bann.k_search(self.dim_data, self.dim_query, 3, 0, 'dkl'),
            np.array([[15, 25, 34],
                [19, 18, 36],
                [19, 18, 36],
                [ 8, 31, 29],
                [28,  2, 33],
                [11, 25, 15],
                [ 2, 28, 33],
                [ 4, 11, 10],
                [ 8, 31, 39],
                [ 8,  7, 31]])))
        self.assertTrue(np.array_equal(
            bann.k_search(self.dim_data, self.dim_query, 3, 0, 'is'),
            np.array([[15, 25, 34],
                [19, 36, 18],
                [19, 36, 18],
                [ 8, 31, 39],
                [ 2, 28, 35],
                [11, 25, 15],
                [ 2, 35, 28],
                [11, 10, 39],
                [ 8, 31, 39],
                [ 8, 31,  7]])))
        self.assertTrue(np.array_equal(
            bann.k_search(self.dim_data, self.dim_query, 3, 0, 'dis'),
            np.array([[15, 25, 34],
                [36, 19, 18],
                [19, 36, 18],
                [ 8, 31, 39],
                [ 2, 28, 35],
                [11, 25, 15],
                [ 2, 28, 35],
                [11,  4, 10],
                [ 8, 31, 39],
                [ 8, 31,  5]])))
        self.assertTrue(np.array_equal(
            bann.k_search(self.dim_data, self.dim_query, 3, 0, 'se'),
            np.array([[15, 25, 34],
                [19, 18,  8],
                [19, 18,  8],
                [ 8, 31, 34],
                [28,  2, 33],
                [11, 15, 25],
                [ 2, 28, 33],
                [ 4, 11, 10],
                [ 8, 31, 19],
                [ 7, 23,  8]])))


    def test_bh_basics(self):
        print("Testing basic Bregman--Hausdorff divergence computations...")
        # Query two 1-point sets for Bregman--Hausdorff divergences.
        # Should return divergence between the two points.
        self.assertTrue(np.isclose(bann.bhaus(self.data, self.query, 0, 'kl'), 0.09013877113318905))
        self.assertTrue(np.isclose(bann.bhaus(self.data, self.query, 0, 'dkl'), 0.09205584583201637))

        self.assertTrue(np.isclose(bann.bhaus(self.data, self.query, 0, 'is'), 0.3068528194400544))
        self.assertTrue(np.isclose(bann.bhaus(self.data, self.query, 0, 'dis'), 0.1931471805599454))
        self.assertTrue(np.isclose(bann.bhaus(self.data, self.query, 0, 'se'), 0.039999999999999994))

    def test_bh(self):
        print("Testing Bregman--Hausdorff divergences in higher dimensions...")
        # Checks if the Bregman Hausdorff divergences are correct
        self.assertTrue(np.isclose(bann.bhaus(self.dim_data, self.dim_query, 0, 'kl'), 0.03451059186103342))
        self.assertTrue(np.isclose(bann.bhaus(self.dim_data, self.dim_query, 0, 'dkl'), 0.03306561649869702))
        self.assertTrue(np.isclose(bann.bhaus(self.dim_data, self.dim_query, 0, 'is'), 0.5652548546527374))
        self.assertTrue(np.isclose(bann.bhaus(self.dim_data, self.dim_query, 0, 'dis'), 0.38024526638997314))
        self.assertTrue(np.isclose(bann.bhaus(self.dim_data, self.dim_query, 0, 'se'), 0.019922427962113392))

    def tests_errors(self):
        print("Testing error handling...")
        # Check if the Exceptions in bann.pyx throw properly

        ## number of nearest neighbours K ValueError
        with self.assertRaises(ValueError):
            bann.k_search(self.data, self.query, 0, 0, None)
        ## Divergence choice TypeError
        with self.assertRaises(TypeError):
            bann.k_search(self.data, self.query, 1, 0, 123)
        ## ValueErrors for invalid divergence choices
        with self.assertRaises(ValueError):
            bann.k_search(self.data, self.query, 1, 0, 'not_a_divergence')

        ## Convert to np.ndarray TypeErrors
        with self.assertRaises(TypeError):
            bann.k_search("not an ndarray", self.query, 1, 0, 'kl')
        with self.assertRaises(TypeError):
            bann.k_search(self.data, "not an ndarray", 1, 0, 'kl')
        ## Dimension of Data and Query ValueErrors
        with self.assertRaises(ValueError):
            bann.k_search(np.array([.1, .6]), self.query, 1, 0, 'kl')
        with self.assertRaises(ValueError):
            bann.k_search(self.data, np.array([.3, .4]), 1, 0, 'kl')

if __name__ == '__main__':
    unittest.main()