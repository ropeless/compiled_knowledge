import unittest
from itertools import count

from ck.in_out.render_net import render_bayesian_network
from ck.in_out.parse_net import read_network
from ck.pgm import PGM
from io import StringIO


class Test_net_renderer(unittest.TestCase):

    def test_round_trip_random_variables(self):
        escapes = ('\\', '\n', '\r', '\v', '\f', '\a', '\b', '\t')

        pgm = PGM()
        pgm.new_rv("name_0", 2)
        pgm.new_rv("name x", ("x1", "x2"))
        pgm.new_rv("name x", ("x 1", "x 2"))
        pgm.new_rv("name x", (1, 2))
        pgm.new_rv("x,\"';", (1, 2, ";\"'"))
        pgm.new_rv('"', ('"', "'"))
        pgm.new_rv('escapes', escapes)

        out = StringIO()
        render_bayesian_network(pgm, out, check_structure_bayesian=False)

        out.seek(0)
        new_pgm = read_network(out)
        rvs = new_pgm.rvs

        self.assertEqual(len(rvs), pgm.number_of_rvs)

        self.assertEqual(rvs[0].name, "name_0")
        self.assertEqual(len(rvs[0]), 2)
        self.assertEqual(rvs[0].states[0], "0")
        self.assertEqual(rvs[0].states[1], "1")

        self.assertEqual(rvs[1].name, "name x")
        self.assertEqual(len(rvs[1]), 2)
        self.assertEqual(rvs[1].states[0], "x1")
        self.assertEqual(rvs[1].states[1], "x2")

        self.assertEqual(rvs[2].name, "name x")
        self.assertEqual(len(rvs[2]), 2)
        self.assertEqual(rvs[2].states[0], "x 1")
        self.assertEqual(rvs[2].states[1], "x 2")

        self.assertEqual(rvs[3].name, "name x")
        self.assertEqual(len(rvs[3]), 2)
        self.assertEqual(rvs[3].states[0], "1")
        self.assertEqual(rvs[3].states[1], "2")

        self.assertEqual(rvs[4].name, "x,\"';")
        self.assertEqual(len(rvs[4]), 3)
        self.assertEqual(rvs[4].states[0], "1")
        self.assertEqual(rvs[4].states[1], "2")
        self.assertEqual(rvs[4].states[2], ";\"'")

        self.assertEqual(rvs[5].name, '"')
        self.assertEqual(len(rvs[5]), 2)
        self.assertEqual(rvs[5].states[0], '"')
        self.assertEqual(rvs[5].states[1], "'")

        self.assertEqual(rvs[6].name, 'escapes')
        self.assertEqual(len(rvs[6]), len(escapes))
        for i, state in enumerate(escapes):
            self.assertEqual(rvs[6].states[i], state)

    def test_factors(self):
        pgm = PGM()
        x = pgm.new_rv('x', 2)
        y = pgm.new_rv('y', 2)

        count_iter = iter(count(1))

        pot_x = pgm.new_factor(x).set_dense().set_iter(count_iter).normalise_cpt()
        pot_y = pgm.new_factor(y, x).set_dense().set_iter(count_iter).normalise_cpt()

        out = StringIO()
        render_bayesian_network(pgm, out)

        out.seek(0)
        new_pgm = read_network(out)

        self.assertEqual(2, new_pgm.number_of_rvs)
        self.assertEqual(2, new_pgm.number_of_factors)
        self.assertEqual(2, new_pgm.number_of_functions)

        new_fx = new_pgm.factors[0]
        new_fy = new_pgm.factors[1]
        if len(new_pgm.factors[0].rvs) != 1:
            new_fx, new_fy = new_fy, new_fx

        new_pot_x = new_fx.function
        new_pot_y = new_fy.function

        for key, value in pot_x.items():
            self.assertEqual(value, new_pot_x[key])

        for key, value in pot_y.items():
            self.assertEqual(value, new_pot_y[key])


if __name__ == '__main__':
    unittest.main()
