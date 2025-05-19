from ck.pgm import PGM
from tests.helpers.unittest_fixture import Fixture, test_main


class ExampleTests(Fixture):
    """
    Unit tests for the `ck.example` package.
    These test confirm that example PGMs construct without error.
    """

    def _check(self, pgm):
        self.assertTrue(isinstance(pgm, PGM))
        self.assertTrue(pgm.name.startswith(pgm.__class__.__name__))

    def test_all_examples(self):
        from ck.example import ALL_EXAMPLES
        for name, pgm_class in ALL_EXAMPLES.items():
            pgm = pgm_class()
            self.assertTrue(pgm.name.startswith(name))
            self._check(pgm)

    def test_alarm(self):
        from ck.example.alarm import Alarm
        self._check(Alarm())

    def test_bow_tie(self):
        from ck.example.bow_tie import BowTie
        self._check(BowTie())

    def test_cancer(self):
        from ck.example.cancer import Cancer
        self._check(Cancer())

    def test_chain(self):
        from ck.example.chain import Chain
        self._check(Chain())

    def test_clique(self):
        from ck.example.clique import Clique
        self._check(Clique())

    def test_cnf_pgm(self):
        from ck.example.cnf_pgm import CNF_PGM
        self._check(CNF_PGM())

    def test_diamond_square(self):
        from ck.example.diamond_square import DiamondSquare
        self._check(DiamondSquare())

    def test_empty(self):
        from ck.example.empty import Empty
        self._check(Empty())

    def test_hailfinder(self):
        from ck.example.hailfinder import Hailfinder
        self._check(Hailfinder())

    def test_hepar2(self):
        from ck.example.hepar2 import Hepar2
        self._check(Hepar2())

    def test_insurance(self):
        from ck.example.insurance import Insurance
        self._check(Insurance())

    def test_loop(self):
        from ck.example.loop import Loop
        self._check(Loop())

    def test_rectangle(self):
        from ck.example.rectangle import Rectangle
        self._check(Rectangle())

    def test_run(self):
        from ck.example.run import Run
        self._check(Run())

    def test_sprinkler(self):
        from ck.example.sprinkler import Sprinkler
        self._check(Sprinkler())

    def test_star(self):
        from ck.example.star import Star
        self._check(Star())

    def test_stress(self):
        from ck.example.stress import Stress
        self._check(Stress())

    def test_student(self):
        from ck.example.student import Student
        self._check(Student())

    def test_triangle_square(self):
        from ck.example.triangle_square import TriangleSquare
        self._check(TriangleSquare())

    def test_truss(self):
        from ck.example.truss import Truss
        self._check(Truss())


if __name__ == '__main__':
    test_main()
