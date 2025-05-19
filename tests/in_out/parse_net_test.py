import unittest

from ck.in_out.parse_net import *

BN3 = """
net{}
node x { states = ("0" "1"); }
node y { states = ("0" "1" "2"); }
node z { states = ("0" "1" "2" "3"); }
potential (x) { data = (0.4 0.6); }
potential (y|x) {
    data = (
        (0.1 0.2 0.7)
        (0.2 0.3 0.5)
    );
}
potential (z|y x) {
    data = (
        ((0.01 0.12 0.23 0.64)
         (0.02 0.13 0.24 0.61))

        ((0.11 0.22 0.63 0.04)
         (0.12 0.23 0.64 0.01))

        ((0.21 0.62 0.03 0.14)
         (0.22 0.63 0.04 0.11))
    );
}
"""


class Test_net_parser_simple(unittest.TestCase):

    def test_empty(self):
        pgm = read_network('net{}')

        self.assertEqual(0, pgm.number_of_rvs)
        self.assertEqual(0, pgm.number_of_factors)

    def test_comments(self):
        pgm = read_network("""
        net
        {
            name1 = % this is a comment
            "a string value";

            name2 = "percent in a string % ";
        }
        node x { states = ("0" "1"); }
        node y { states = ("0" "1" "2"); }
        potential (x) { data = (0.4 0.6); }
        potential (y|x) {
            data = (
                (0.1 0.2 0.7)   % and more comments
                (0.2 0.3 0.5)
            );
        }
        """)

        self.assertEqual(2, pgm.number_of_rvs)
        self.assertEqual(2, pgm.number_of_factors)

    def test_one_node(self):
        pgm = read_network("""
        net{}
        node x { states = (1 2 3); }
        """)

        self.assertEqual(1, pgm.number_of_rvs)
        self.assertEqual(0, pgm.number_of_factors)

    def test_one_node_with_potential(self):
        pgm = read_network("""
        net{}
        node x { states = (1 2 3); }
        potential (x) { data = (0.1 0.2 0.7); }
        """)

        self.assertEqual(1, pgm.number_of_rvs)
        self.assertEqual(1, pgm.number_of_factors)

    def test_two_nodes_with_potential(self):
        pgm = read_network("""
        net{}
        node x { states = ("0" "1"); }
        node y { states = ("0" "1" "2"); }
        potential (x) { data = (0.4 0.6); }
        potential (y|x) {
            data = (
                (0.1 0.2 0.7)
                (0.2 0.3 0.5)
            );
        }
        """)

        self.assertEqual(2, pgm.number_of_rvs)
        self.assertEqual(2, pgm.number_of_factors)

        f_x = pgm.factors[0].function
        self.assertEqual(f_x[0], 0.4)
        self.assertEqual(f_x[1], 0.6)

        f_yx = pgm.factors[1].function
        self.assertEqual(f_yx[0, 0], 0.1)
        self.assertEqual(f_yx[1, 0], 0.2)
        self.assertEqual(f_yx[2, 0], 0.7)
        self.assertEqual(f_yx[0, 1], 0.2)
        self.assertEqual(f_yx[1, 1], 0.3)
        self.assertEqual(f_yx[2, 1], 0.5)

    def test_three_nodes_with_potential(self):
        pgm = read_network(BN3)
        self._assert_pgm_is_BN3(pgm)

    def test_pgm_dense(self):
        pgm = read_network(BN3, network_builder=PGM_NetworkBuilder(PGM_NetworkBuilder.add_function_dense))
        self._assert_pgm_is_BN3(pgm)

    def test_pgm_sparse(self):
        pgm = read_network(BN3, network_builder=PGM_NetworkBuilder(PGM_NetworkBuilder.add_function_sparse))
        self._assert_pgm_is_BN3(pgm)

    def test_pgm_compact(self):
        pgm = read_network(BN3, network_builder=PGM_NetworkBuilder(PGM_NetworkBuilder.add_function_compact))
        self._assert_pgm_is_BN3(pgm)

    def test_pgm_cpt(self):
        pgm = read_network(BN3, network_builder=PGM_NetworkBuilder(PGM_NetworkBuilder.add_function_cpt))
        self._assert_pgm_is_BN3(pgm)

    def _assert_pgm_is_BN3(self, pgm):
        self.assertEqual(3, pgm.number_of_rvs)
        self.assertEqual(3, pgm.number_of_factors)

        f_x = pgm.factors[0].function
        self.assertEqual(f_x[0], 0.4)
        self.assertEqual(f_x[1], 0.6)

        f_yx = pgm.factors[1].function
        self.assertEqual(f_yx[0, 0], 0.1)
        self.assertEqual(f_yx[1, 0], 0.2)
        self.assertEqual(f_yx[2, 0], 0.7)
        self.assertEqual(f_yx[0, 1], 0.2)
        self.assertEqual(f_yx[1, 1], 0.3)
        self.assertEqual(f_yx[2, 1], 0.5)

        f_zyx = pgm.factors[2].function
        self.assertEqual(f_zyx[0, 0, 0], 0.01)
        self.assertEqual(f_zyx[1, 0, 0], 0.12)
        self.assertEqual(f_zyx[2, 0, 0], 0.23)
        self.assertEqual(f_zyx[3, 0, 0], 0.64)

        self.assertEqual(f_zyx[0, 0, 1], 0.02)
        self.assertEqual(f_zyx[1, 0, 1], 0.13)
        self.assertEqual(f_zyx[2, 0, 1], 0.24)
        self.assertEqual(f_zyx[3, 0, 1], 0.61)

        self.assertEqual(f_zyx[0, 1, 0], 0.11)
        self.assertEqual(f_zyx[1, 1, 0], 0.22)
        self.assertEqual(f_zyx[2, 1, 0], 0.63)
        self.assertEqual(f_zyx[3, 1, 0], 0.04)

        self.assertEqual(f_zyx[0, 1, 1], 0.12)
        self.assertEqual(f_zyx[1, 1, 1], 0.23)
        self.assertEqual(f_zyx[2, 1, 1], 0.64)
        self.assertEqual(f_zyx[3, 1, 1], 0.01)

        self.assertEqual(f_zyx[0, 2, 0], 0.21)
        self.assertEqual(f_zyx[1, 2, 0], 0.62)
        self.assertEqual(f_zyx[2, 2, 0], 0.03)
        self.assertEqual(f_zyx[3, 2, 0], 0.14)

        self.assertEqual(f_zyx[0, 2, 1], 0.22)
        self.assertEqual(f_zyx[1, 2, 1], 0.63)
        self.assertEqual(f_zyx[2, 2, 1], 0.04)
        self.assertEqual(f_zyx[3, 2, 1], 0.11)


if __name__ == '__main__':
    unittest.main()
