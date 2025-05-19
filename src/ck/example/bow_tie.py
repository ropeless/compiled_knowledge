import random as _random

from ck.pgm import PGM


class BowTie(PGM):
    """
    This PGM is the 'BowTie' factor graph.

    A BowTie is a factor graph with five random variables (x1, ..., x5).
    One factor connects: x1, x2, x3.
    Another factor connects: x1, x4, x5.
    """

    def __init__(
            self,
            states_per_var: int = 2,
            random_seed: int = 123456,
    ):
        params = (states_per_var,)
        super().__init__(f'{self.__class__.__name__}({",".join(str(param) for param in params)})')

        random_stream = _random.Random(random_seed).random
        rand_iter = iter(random_stream, None)

        x1 = self.new_rv('x1', states_per_var)
        x2 = self.new_rv('x2', states_per_var)
        x3 = self.new_rv('x3', states_per_var)
        x4 = self.new_rv('x4', states_per_var)
        x5 = self.new_rv('x5', states_per_var)

        self.new_factor(x1, x2, x3).set_dense().set_iter(rand_iter)
        self.new_factor(x1, x4, x5).set_dense().set_iter(rand_iter)
