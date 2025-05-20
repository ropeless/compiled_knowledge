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
        from ck.example import Alarm
        self._check(Alarm())

    def test_asia(self):
        from ck.example import Asia
        self._check(Asia())

    def test_binary_clique(self):
        from ck.example import BinaryClique
        self._check(BinaryClique())

    def test_bow_tie(self):
        from ck.example import BowTie
        self._check(BowTie())

    def test_cancer(self):
        from ck.example import Cancer
        self._check(Cancer())

    def test_chain(self):
        from ck.example import Chain
        self._check(Chain())

    def test_child(self):
        from ck.example import Child
        self._check(Child())

    def test_clique(self):
        from ck.example import Clique
        self._check(Clique())

    def test_cnf_pgm(self):
        from ck.example import CNF_PGM
        self._check(CNF_PGM())

    def test_diamond_square(self):
        from ck.example import DiamondSquare
        self._check(DiamondSquare())

    def test_earthquake(self):
        from ck.example import Earthquake
        self._check(Earthquake())

    def test_empty(self):
        from ck.example import Empty
        self._check(Empty())

    def test_hailfinder(self):
        from ck.example import Hailfinder
        self._check(Hailfinder())

    def test_hepar2(self):
        from ck.example import Hepar2
        self._check(Hepar2())

    def test_insurance(self):
        from ck.example import Insurance
        self._check(Insurance())

    def test_loop(self):
        from ck.example import Loop
        self._check(Loop())

    def test_mildew(self):
        from ck.example import Mildew
        self._check(Mildew())

    def test_munin(self):
        from ck.example import Munin
        self._check(Munin())

    def test_pathfinder(self):
        from ck.example import Pathfinder
        self._check(Pathfinder())

    def test_rain(self):
        from ck.example import Rain
        self._check(Rain())

    def test_rectangle(self):
        from ck.example import Rectangle
        self._check(Rectangle())

    def test_run(self):
        from ck.example import Run
        self._check(Run())

    def test_sachs(self):
        from ck.example import Sachs
        self._check(Sachs())

    def test_sprinkler(self):
        from ck.example import Sprinkler
        self._check(Sprinkler())

    def test_star(self):
        from ck.example import Star
        self._check(Star())

    def test_stress(self):
        from ck.example import Stress
        self._check(Stress())

    def test_student(self):
        from ck.example import Student
        self._check(Student())

    def test_survey(self):
        from ck.example import Survey
        self._check(Survey())

    def test_triangle_square(self):
        from ck.example import TriangleSquare
        self._check(TriangleSquare())

    def test_truss(self):
        from ck.example import Truss
        self._check(Truss())


if __name__ == '__main__':
    test_main()
