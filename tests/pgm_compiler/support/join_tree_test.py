from ck.pgm import PGM
from ck.pgm_compiler.support.clusters import Clusters
from ck.pgm_compiler.support.join_tree import JoinTree, clusters_to_join_tree
from tests.helpers.unittest_fixture import Fixture, test_main


class TestJoinTree(Fixture):

    def test_empty(self):
        pgm = PGM()
        clusters = Clusters(pgm)

        self.assertArrayEqual(clusters.clusters, [])
        self.assertArrayEqual(clusters.eliminated, [])

        join_tree: JoinTree = clusters_to_join_tree(clusters)

        self.assertIs(join_tree.pgm, pgm)
        self.assertEqual(join_tree.cluster, set())
        self.assertEqual(join_tree.children, [])
        self.assertEqual(join_tree.factors, [])
        self.assertEqual(join_tree.separator, set())

    def test_one_cluster(self):
        pgm = PGM()
        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 2)
        c = pgm.new_rv('c', 2)
        fab = pgm.new_factor(a, b)
        fbc = pgm.new_factor(b, c)
        fac = pgm.new_factor(a, c)

        clusters = Clusters(pgm)
        clusters.eliminate(0)
        clusters.eliminate(1)
        clusters.eliminate(2)

        self.assertArraySetEqual(clusters.eliminated, [0, 1, 2])
        self.assertEqual(len(clusters.clusters), 1)

        cluster = clusters.clusters[0]
        self.assertArraySetEqual(cluster, [0, 1, 2])

        join_tree: JoinTree = clusters_to_join_tree(clusters)

        self.assertIs(join_tree.pgm, pgm)
        self.assertEqual(join_tree.cluster, {0, 1, 2})
        self.assertEqual(join_tree.children, [])
        self.assertEqual(len(join_tree.factors), 3)
        self.assertEqual(join_tree.separator, set())

        join_tree_factor_ids = {id(factor) for factor in join_tree.factors}
        self.assertEqual(join_tree_factor_ids, {id(fab), id(fbc), id(fac)})

    def test_two_clusters(self):
        pgm = PGM()
        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 2)
        c = pgm.new_rv('c', 2)
        fab = pgm.new_factor(a, b)
        fbc = pgm.new_factor(b, c)

        clusters = Clusters(pgm)
        clusters.eliminate(0)
        clusters.eliminate(1)
        clusters.eliminate(2)

        self.assertArraySetEqual(clusters.eliminated, [0, 1, 2])
        self.assertEqual(len(clusters.clusters), 2)

        cluster = clusters.clusters[0]
        self.assertArraySetEqual(cluster, [0, 1])

        cluster = clusters.clusters[1]
        self.assertArraySetEqual(cluster, [1, 2])

        join_tree: JoinTree = clusters_to_join_tree(clusters)

        self.assertIs(join_tree.pgm, pgm)
        self.assertEqual(join_tree.cluster, {0, 1})
        self.assertEqual(len(join_tree.children), 1)
        self.assertEqual(len(join_tree.factors), 1)
        self.assertEqual(join_tree.separator, set())

        child: JoinTree = join_tree.children[0]

        self.assertIs(child.pgm, pgm)
        self.assertEqual(child.cluster, {1, 2})
        self.assertEqual(child.children, [])
        self.assertEqual(len(child.factors), 1)
        self.assertEqual(child.separator, {1})

        self.assertEqual({id(factor) for factor in join_tree.factors}, {id(fab)})
        self.assertEqual({id(factor) for factor in child.factors}, {id(fbc)})


if __name__ == '__main__':
    test_main()
