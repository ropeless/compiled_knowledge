import random as _random

from ck.pgm import PGM
from ck.utils.random_extras import random_permutation as _random_permutation


class CNF_PGM(PGM):
    """
    This is a PGM that uses a factor graph to represent a
    CNF propositional theory.

    A CNF_PGM factor graph has random clause factors.
    Random variables are named x0, x1, x2, ... .
    """

    def __init__(
            self,
            num_vars: int = 30,
            num_clauses: int = 200,
            min_clause_size: int = 3,
            max_clause_size: int = 5,
            random_seed: int = 129837697,
    ):
        params = (num_vars, num_clauses, min_clause_size, max_clause_size)
        super().__init__(f'{self.__class__.__name__}({",".join(str(param) for param in params)})')

        random = _random.Random(random_seed)

        for i in range(num_vars):
            self.new_rv(f'x{i}', 2)

        pgm_rvs = self.rvs
        for i in range(num_clauses):
            size = random.randint(min_clause_size, max_clause_size)
            perm = _random_permutation(num_vars, random)
            rvs = tuple(pgm_rvs[j] for j in perm[:size])
            states = [random.randint(0, 1) for _ in range(size)]

            self.new_factor(*rvs).set_clause(*states)
