import random as _random

from ck.pgm import PGM
from ck.utils.iter_extras import pairs as _pairs


class BinaryClique(PGM):
    """
    This PGM is a factor graph with binary factors and a fully connected set of variables
    """

    def __init__(
            self,
            vars_per_clique: int = 5,
            states_per_var: int = 2,
            include_unaries: bool = False,
            random_seed: int = 123456,
    ):
        params = (vars_per_clique, states_per_var, include_unaries)
        super().__init__(f'{self.__class__.__name__}({",".join(str(param) for param in params)})')

        random_stream = _random.Random(random_seed).random

        rvs = [self.new_rv(f'x{i}', states_per_var) for i in range(vars_per_clique)]

        for x_i, x_j in _pairs(rvs):
            self.new_factor(x_i, x_j).set_dense().set_stream(random_stream)

        if include_unaries:
            if include_unaries:
                for rv in rvs:
                    self.new_factor(rv).set_dense().set_stream(random_stream)
