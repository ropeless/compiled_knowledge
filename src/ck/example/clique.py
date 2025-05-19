import random as _random

from ck.pgm import PGM


class Clique(PGM):
    """
    This PGM is the 'Clique' factor graph.

    The Clique factor graph has one factor that connects to all random variables.
    Random variables are named x0, x1, x2, ... .
    If include_unaries then, also includes one unary factor per random variable.
    """

    def __init__(
            self,
            vars_per_clique: int = 17,
            states_per_var: int = 2,
            include_unaries: bool = True,
            random_seed: int = 123456,
    ):
        params = (vars_per_clique, states_per_var, include_unaries)
        super().__init__(f'{self.__class__.__name__}({",".join(str(param) for param in params)})')

        random_stream = _random.Random(random_seed).random

        rvs = [self.new_rv(f'x{i}', states_per_var) for i in range(vars_per_clique)]

        self.new_factor(*rvs).set_dense().set_stream(random_stream)

        if include_unaries:
            for rv in rvs:
                self.new_factor(rv).set_dense().set_stream(random_stream)
