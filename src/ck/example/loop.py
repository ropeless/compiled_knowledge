import math
import random as _random

from ck.pgm import PGM


class Loop(PGM):
    """
    This PGM is the 'Loop' factor graph.

    The Loop factor graph consists of a chain of random variables where
    adjacent random variables in the chain are connected by a binary factor.
    Random variables are named x0, x1, x2, ... .
    If include_unaries then, also includes one unary factor per random variable.
    """

    def __init__(
            self,
            vars_per_loop: int = 20,
            states_per_var: int = 20,
            include_unaries: int = True,
            random_seed: int = 123456,
    ):
        params = (vars_per_loop, states_per_var, include_unaries)
        super().__init__(f'{self.__class__.__name__}({",".join(str(param) for param in params)})')

        scale = 1 + math.log2(vars_per_loop)
        random_stream = _random.Random(random_seed).random
        binary_iter = map(lambda x: x / scale, iter(random_stream, None))
        unary_iter = iter(random_stream, None)

        rvs = [self.new_rv(f'x{i}', states_per_var) for i in range(vars_per_loop)]

        for i in range(1, len(rvs)):
            self.new_factor(rvs[i], rvs[i - 1]).set_dense().set_iter(binary_iter)
        self.new_factor(rvs[-1], rvs[0]).set_dense().set_iter(binary_iter)

        if include_unaries:
            for rv in rvs:
                self.new_factor(rv).set_dense().set_iter(unary_iter)
