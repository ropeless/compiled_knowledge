from ck.pgm import PGM
from ck.pgm_compiler.support.clusters import Clusters, ve_fixed
from tests.helpers.unittest_fixture import Fixture, test_main


class TestClusters(Fixture):
    def test_empty_clusters(self):
        pgm: PGM = PGM()
        clusters = Clusters(pgm)

        self.assertIs(clusters.pgm, pgm)
        self.assertArrayEqual(clusters.eliminated, [])
        self.assertEqual(clusters.uneliminated, set())
        self.assertArrayEqual(clusters.clusters, [])

        self.assertEqual(clusters.max_cluster_size(), 0)
        self.assertEqual(clusters.max_cluster_weighted_size(pgm.rv_log_sizes), 0)

        with self.assertRaises(Exception):
            # Nothing to eliminate
            clusters.eliminate(0)

    def test_elimination(self):
        # Make a Bayesian network:
        #
        #  x0   x1
        #   \  / \
        #    x2  x3
        #    |
        #   x4
        pgm: PGM = PGM()
        x = [pgm.new_rv(f'x{i}', 2) for i in [0, 1, 2, 3, 4]]
        pgm.new_factor(x[0])
        pgm.new_factor(x[1])
        pgm.new_factor(x[2], x[0], x[1])
        pgm.new_factor(x[3], x[1])
        pgm.new_factor(x[4], x[2])

        clusters = Clusters(pgm)

        self.assertIs(clusters.pgm, pgm)

        self.assertArrayEqual(clusters.eliminated, [])
        self.assertEqual(clusters.uneliminated, {xi.idx for xi in x})

        self.assertEqual(clusters.connections(x[0].idx), {x[1].idx, x[2].idx})
        self.assertEqual(clusters.connections(x[1].idx), {x[0].idx, x[2].idx, x[3].idx})
        self.assertEqual(clusters.connections(x[2].idx), {x[0].idx, x[1].idx, x[4].idx})
        self.assertEqual(clusters.connections(x[3].idx), {x[1].idx})
        self.assertEqual(clusters.connections(x[4].idx), {x[2].idx})

        self.assertEqual(clusters.degree(x[0].idx), 2)
        self.assertEqual(clusters.degree(x[1].idx), 3)
        self.assertEqual(clusters.degree(x[2].idx), 3)
        self.assertEqual(clusters.degree(x[3].idx), 1)
        self.assertEqual(clusters.degree(x[4].idx), 1)

        self.assertEqual(clusters.fill(x[0].idx), 0)
        self.assertEqual(clusters.fill(x[1].idx), 2)
        self.assertEqual(clusters.fill(x[2].idx), 2)
        self.assertEqual(clusters.fill(x[3].idx), 0)
        self.assertEqual(clusters.fill(x[4].idx), 0)

        self.assertEqual(clusters.weighted_degree(x[0].idx), 2)
        self.assertEqual(clusters.weighted_degree(x[1].idx), 3)
        self.assertEqual(clusters.weighted_degree(x[2].idx), 3)
        self.assertEqual(clusters.weighted_degree(x[3].idx), 1)
        self.assertEqual(clusters.weighted_degree(x[4].idx), 1)

        self.assertEqual(clusters.weighted_fill(x[0].idx), 0)
        self.assertEqual(clusters.weighted_fill(x[1].idx), 2)
        self.assertEqual(clusters.weighted_fill(x[2].idx), 2)
        self.assertEqual(clusters.weighted_fill(x[3].idx), 0)
        self.assertEqual(clusters.weighted_fill(x[4].idx), 0)

        self.assertEqual(clusters.traditional_weighted_fill(x[0].idx), 0)
        self.assertEqual(clusters.traditional_weighted_fill(x[1].idx), 0.5)
        self.assertEqual(clusters.traditional_weighted_fill(x[2].idx), 0.5)
        self.assertEqual(clusters.traditional_weighted_fill(x[3].idx), 0)
        self.assertEqual(clusters.traditional_weighted_fill(x[4].idx), 0)

        clusters.eliminate(x[4].idx)

        self.assertArrayEqual(clusters.eliminated, [x[4].idx])
        self.assertEqual(clusters.uneliminated, {x[0].idx, x[1].idx, x[2].idx, x[3].idx})

        self.assertEqual(clusters.connections(x[0].idx), {x[1].idx, x[2].idx})
        self.assertEqual(clusters.connections(x[1].idx), {x[0].idx, x[2].idx, x[3].idx})
        self.assertEqual(clusters.connections(x[2].idx), {x[0].idx, x[1].idx})
        self.assertEqual(clusters.connections(x[3].idx), {x[1].idx})

        self.assertEqual(clusters.degree(x[0].idx), 2)
        self.assertEqual(clusters.degree(x[1].idx), 3)
        self.assertEqual(clusters.degree(x[2].idx), 2)
        self.assertEqual(clusters.degree(x[3].idx), 1)

        self.assertEqual(clusters.fill(x[0].idx), 0)
        self.assertEqual(clusters.fill(x[1].idx), 2)
        self.assertEqual(clusters.fill(x[2].idx), 0)
        self.assertEqual(clusters.fill(x[3].idx), 0)

        clusters.eliminate(x[3].idx)

        self.assertArrayEqual(clusters.eliminated, [x[4].idx, x[3].idx])
        self.assertEqual(clusters.uneliminated, {x[0].idx, x[1].idx, x[2].idx})

        self.assertEqual(clusters.connections(x[0].idx), {x[1].idx, x[2].idx})
        self.assertEqual(clusters.connections(x[1].idx), {x[0].idx, x[2].idx})
        self.assertEqual(clusters.connections(x[2].idx), {x[0].idx, x[1].idx})

        self.assertEqual(clusters.degree(x[0].idx), 2)
        self.assertEqual(clusters.degree(x[1].idx), 2)
        self.assertEqual(clusters.degree(x[2].idx), 2)

        self.assertEqual(clusters.fill(x[0].idx), 0)
        self.assertEqual(clusters.fill(x[1].idx), 0)
        self.assertEqual(clusters.fill(x[2].idx), 0)

        clusters.eliminate(x[0].idx)

        self.assertArrayEqual(clusters.eliminated, [x[4].idx, x[3].idx, x[0].idx])
        self.assertEqual(clusters.uneliminated, {x[1].idx, x[2].idx})

        self.assertEqual(clusters.connections(x[1].idx), {x[2].idx})
        self.assertEqual(clusters.connections(x[2].idx), {x[1].idx})

        self.assertEqual(clusters.degree(x[1].idx), 1)
        self.assertEqual(clusters.degree(x[2].idx), 1)

        self.assertEqual(clusters.fill(x[1].idx), 0)
        self.assertEqual(clusters.fill(x[2].idx), 0)

        clusters.eliminate(x[1].idx)

        self.assertArrayEqual(clusters.eliminated, [x[4].idx, x[3].idx, x[0].idx, x[1].idx])
        self.assertEqual(clusters.uneliminated, {x[2].idx})

        self.assertEqual(clusters.connections(x[2].idx), set())

        self.assertEqual(clusters.degree(x[2].idx), 0)

        self.assertEqual(clusters.fill(x[2].idx), 0)

        clusters.eliminate(x[2].idx)

        self.assertArrayEqual(clusters.eliminated, [x[4].idx, x[3].idx, x[0].idx, x[1].idx, x[2].idx])
        self.assertEqual(clusters.uneliminated, set())

        # Confirm we got the expected clusters
        self.assertEqual(
            clusters.clusters,
            [
                {x[0].idx, x[1].idx, x[2].idx},
                {x[1].idx, x[3].idx},
                {x[2].idx, x[4].idx},
            ],
        )
        self.assertEqual(clusters.max_cluster_size(), 3)
        self.assertEqual(clusters.max_cluster_weighted_size(pgm.rv_log_sizes), 3)

    def test_ve_fixed(self):
        # Make a Bayesian network:
        #
        #  x0   x1
        #   \  / \
        #    x2  x3
        #    |
        #   x4
        pgm: PGM = PGM()
        x = [pgm.new_rv(f'x{i}', 2) for i in [0, 1, 2, 3, 4]]
        pgm.new_factor(x[0])
        pgm.new_factor(x[1])
        pgm.new_factor(x[2], x[0], x[1])
        pgm.new_factor(x[3], x[1])
        pgm.new_factor(x[4], x[2])

        clusters = Clusters(pgm)

        ve_fixed(clusters, [4, 3, 0, 1, 2])

        self.assertArrayEqual(clusters.eliminated, [x[4].idx, x[3].idx, x[0].idx, x[1].idx, x[2].idx])
        self.assertEqual(clusters.uneliminated, set())

        # Confirm we got the expected clusters
        self.assertEqual(
            clusters.clusters,
            [
                {0, 1, 2},
                {1, 3},
                {2, 4},
            ],
        )
        self.assertEqual(clusters.max_cluster_size(), 3)
        self.assertEqual(clusters.max_cluster_weighted_size(pgm.rv_log_sizes), 3)

    def test_maximal_clusters_only_false(self):
        # Make a Bayesian network:
        #
        #  x0   x1
        #   \  / \
        #    x2  x3
        #    |
        #   x4
        pgm: PGM = PGM()
        x = [pgm.new_rv(f'x{i}', 2) for i in [0, 1, 2, 3, 4]]
        pgm.new_factor(x[0])
        pgm.new_factor(x[1])
        pgm.new_factor(x[2], x[0], x[1])
        pgm.new_factor(x[3], x[1])
        pgm.new_factor(x[4], x[2])

        clusters = Clusters(pgm, maximal_clusters_only=False)

        ve_fixed(clusters, [4, 3, 0, 1, 2])

        self.assertArrayEqual(clusters.eliminated, [x[4].idx, x[3].idx, x[0].idx, x[1].idx, x[2].idx])
        self.assertEqual(clusters.uneliminated, set())

        # Confirm we got the expected clusters
        self.assertEqual(
            clusters.clusters,
            [
                {0, 1, 2},
                {1, 2},
                {2},
                {1, 3},
                {2, 4},
            ],
        )
        self.assertEqual(clusters.max_cluster_size(), 3)
        self.assertEqual(clusters.max_cluster_weighted_size(pgm.rv_log_sizes), 3)


if __name__ == '__main__':
    test_main()
