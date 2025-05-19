import random as _random

from ck.pgm import PGM


class Star(PGM):
    """
    This PGM is the 'Star' factor graph.

    The Star factor graph is where the first random variable, x0, is
    connected to each of the other random variables, x1, x2, ..., via a binary factor.
    If include_unaries then, also includes one unary factor per random variable.
    """

    def __init__(
            self,
            num_of_arms: int = 3,
            length_of_arms: int = 2,
            states_per_var: int = 2,
            include_unaries: bool = True,
            random_seed: int = 123456,
    ):
        params = (num_of_arms, length_of_arms, states_per_var, include_unaries)
        super().__init__(f'{self.__class__.__name__}({",".join(str(param) for param in params)})')

        random_stream = _random.Random(random_seed).random
        rand_iter = iter(random_stream, None)

        x0 = self.new_rv('x0', states_per_var)
        arms = [
            [self.new_rv(f'x{arm}_{i}', states_per_var) for i in range(length_of_arms)]
            for arm in range(num_of_arms)
        ]

        for arm in arms:
            self.new_factor(x0, arm[0]).set_dense().set_iter(rand_iter)
            for i in range(1, len(arm)):
                self.new_factor(arm[i - 1], arm[i]).set_dense().set_iter(rand_iter)

        if include_unaries:
            self.new_factor(x0).set_dense().set_iter(rand_iter)
            for arm in arms:
                for rv in arm:
                    self.new_factor(rv).set_dense().set_iter(rand_iter)
