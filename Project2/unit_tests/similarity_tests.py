import unittest
import numpy as np
from similarity import (
    vector_norm,
    cosine_similarity,
    pearson_correlation,
    filter_common
)
from scipy.stats import pearsonr


class SimilarityTests(unittest.TestCase):
    vector_norm_tests = (
        (
            np.array([1, 0]), 1
        ),
        (
            np.array([0, 0, 0]), 0
        ),
        (
            np.array([1, -1, 1, -1]), 2
        )
    )

    def test_vector_norm(self):
        for v, expected in self.vector_norm_tests:
            result = vector_norm(v)
            self.assertEqual(result, expected)

    cosine_similarity_tests = (
        (
            (np.array([9, 3, 0, 0, 5]), np.array([10, 3, 8, 0, 5])),
            0.9989
        ),
        (
            (np.array([1, 0, -1]), np.array([-1, 0, 1])),
            -1
        ),
        (
            (np.array([1, 1, 0, 0]), np.array([0, 0, 1, -1])),
            0
        )
    )

    def test_cosine_similarity(self):
        for vecs, expected in self.cosine_similarity_tests:
            result = cosine_similarity(vecs[0], vecs[1])
            self.assertAlmostEqual(result, expected, places=3)

    pearson_correlation_tests = (
        (
            (np.array([4, 4, 1, 4, 3]), np.array([5, 4, 2, 0, 3])),
            0.91287093
        ),
        (
            (np.array([5, 4, 2, 0, 3]), np.array([4, 4, 1, 4, 3])),
            0.91287093
        ),
        (
            (np.array([1, 0, -1]), np.array([-1, 0, 1])),
            -1.0
        ),
        (
            (np.array([1, 1, 0, 0]), np.array([0, 0, 1, -1])),
            0
        )
    )

    def test_pearson_correlation(self):
        for vecs, expected in self.pearson_correlation_tests:
            p = pearsonr(vecs[0], vecs[1])
            print(p, sum(p))
            result = pearson_correlation(vecs[0], vecs[1])
            self.assertAlmostEqual(result, expected, places=5)

    filter_common_tests = (
        (
            ([0, 1, 2, 4, 5], [1, 2, 3, 0, 4]),
            ([1, 2, 5], [2, 3, 4])
        ),
    )

    def test_filter_common(self):
        for vecs, expected in self.filter_common_tests:
            result = filter_common(vecs[0], vecs[1])
            self.assertTrue((result[0] == expected[0]).all())
            self.assertTrue((result[1] == expected[1]).all())

#    def test_speed(self):
#        start = time.time()
#        for i in range(10000):
#            self.test_cosine_similarity()
#
#        end = time.time()
#        elapsed = end - start
#        self.assertGreaterEqual(0.5, elapsed)


if __name__ == '__main__':
    unittest.main()
