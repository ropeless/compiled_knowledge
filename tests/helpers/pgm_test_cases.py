from abc import ABC, abstractmethod
from random import random

from ck.pgm import PGM


class PGMTestCases(ABC):
    """
    This is a test case mix-in for running a variety of PGMs through a check function.
    """

    @abstractmethod
    def check_pgm(self, pgm: PGM) -> None:
        ...

    def test_empty(self):
        pgm = PGM()
        self.check_pgm(pgm)

    def test_stress(self):
        # Many different kinds of random variables, factors, and potential functions.
        pgm = PGM('test_stress PGM')

        no_name = pgm.new_rv('', 2)
        x1 = pgm.new_rv('x1', 2)
        x2 = pgm.new_rv('x2', ['one', 'two'])
        x3 = pgm.new_rv('x3', ['one', 'two', 'non-ascii 😊'])
        x4 = pgm.new_rv('x4', [1.1, 1.23, None, 'string'])
        x5 = pgm.new_rv('x5', 3)
        A = pgm.new_rv('A', 2)
        B = pgm.new_rv('B', 2)
        C = pgm.new_rv('C', 2)
        D = pgm.new_rv('D', 2)
        E = pgm.new_rv('E', [True, False])

        pgm.new_factor(no_name).set_zero()

        pgm.new_factor(x1).set_dense().set_flat(1, 2)

        pgm.new_factor(x1, x2).set_clause(1, 0)

        pgm.new_factor(x2, x3).set_cpt().set_all(
            (0.1, 0.9),
            (0.2, 0.8),
            (0.3, 0.7),
        )

        f_3_4 = pgm.new_factor(x3, x4).set_sparse()
        f_3_4[0, 0] = 11
        f_3_4[1, 1] = 11
        f_3_4[2, 2] = 22

        f_4_5 = pgm.new_factor(x4, x5).set_compact()
        f_4_5[0, 0] = 11
        f_4_5[1, 1] = 11
        f_4_5[2, 2] = 22

        f_ab = pgm.new_factor(A, B)
        f_ab.set_dense().set_flat(2, 3, 5, 7)

        pgm.new_factor(A, C).set_sparse()[0, 1] = 11
        pgm.new_factor(A, D).set_sparse()[1, 0] = 13
        pgm.new_factor(B, C).set_clause(0, 1)
        pgm.new_factor(B, D).set_cpt().set_all([0.2, 0.8], [0.3, 0.7])

        pgm.new_factor(A, E)  # leave with no potential function

        pgm.new_factor(B, E).function = f_ab.function  # shared function

        # Two factors on the same random variables
        pgm.new_factor(C, E).set_sparse()
        pgm.new_factor(C, E).set_sparse()

        self.check_pgm(pgm)

    def test_one_var_one_factor(self) -> None:
        pgm = PGM()
        x = pgm.new_rv('x', 2)
        pgm.new_factor(x).set_dense().set_flat(0.1, 0.9)

        self.check_pgm(pgm)

    def test_two_var_one_factor(self) -> None:
        pgm = PGM()
        x = pgm.new_rv('x', 2)
        y = pgm.new_rv('y', 3)
        pgm.new_factor(x, y).set_dense().set_flat(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)

        self.check_pgm(pgm)

    def test_shared_function(self):
        pgm = PGM()
        x1 = pgm.new_rv('x1', 2)
        x2 = pgm.new_rv('x2', 3)
        x3 = pgm.new_rv('x3', 2)
        x4 = pgm.new_rv('x4', 3)

        f = pgm.new_factor(x1, x2).set_dense().set_stream(random)
        pgm.new_factor(x3, x4).function = f

        self.check_pgm(pgm)

    def test_non_ascii_strings(self):
        pgm = PGM('a name')
        pgm.new_rv('Facility', (
            # St Vincent's
            b'\x53\x74\x20\x56\x69\x6E\x63\x65\x6E\x74\x92\x73\x20'.decode('cp1252'),
            # Sydney Children's
            b'\x53\x79\x64\x6E\x65\x79\x20\x43\x68\x69\x6C\x64\x72\x65\x6E\x92\x73'.decode('cp1252')
        ))

        self.check_pgm(pgm)
