import unittest

import numpy as np

import nmsbind


def get_exact_cosine(row, data, N=10):
    scores = data.dot(row) / np.linalg.norm(data, axis=-1)
    best = np.argpartition(scores, -N)[-N:]
    return sorted(zip(best, scores[best] / np.linalg.norm(row)), key=lambda x: -x[1])


def get_hitrate(ground_truth, ids):
    return len(set(i for i, _ in ground_truth).intersection(ids))


class NMSBindTest(unittest.TestCase):
    def testCosine(self):
        np.random.seed(23)
        data = np.random.randn(1000, 10).astype(np.float32)

        index = nmsbind.init(data=data, method='sw-graph', space='cosinesimil')
        index.createIndex()

        row = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1.])
        ids, distances = index.knnQuery(row, k=10)
        self.assertTrue(get_hitrate(get_exact_cosine(row, data), ids) >= 9)


if __name__ == "__main__":
    unittest.main()
