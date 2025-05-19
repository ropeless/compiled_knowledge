import random as _random

from ck.pgm import PGM


class TriangleSquare(PGM):
    r"""
    This PGM is the 'TriangleSquare' factor graph.

    The TriangleSquare is a factor graph with six random variables (a, b, c, ..., f).
    Binary factors are between pairs of random variables crating the pattern:
                  b -- d
                / |    | \
               a  |    |  f
                \ |    | /
                  c -- e
    If include_unaries then, also includes one unary factor per random variable.
    """

    def __init__(
            self,
            states_per_var=2,
            include_unaries=True,
            random_seed=123456
    ):
        params = (states_per_var, include_unaries)
        super().__init__(f'{self.__class__.__name__}({",".join(str(param) for param in params)})')

        random_stream = _random.Random(random_seed).random
        rand_iter = iter(random_stream, None)

        a = self.new_rv('a', states_per_var)
        b = self.new_rv('b', states_per_var)
        c = self.new_rv('c', states_per_var)
        d = self.new_rv('d', states_per_var)
        e = self.new_rv('e', states_per_var)
        f = self.new_rv('f', states_per_var)

        self.new_factor(a, b).set_dense().set_iter(rand_iter)
        self.new_factor(a, c).set_dense().set_iter(rand_iter)
        self.new_factor(b, c).set_dense().set_iter(rand_iter)
        self.new_factor(b, d).set_dense().set_iter(rand_iter)
        self.new_factor(c, e).set_dense().set_iter(rand_iter)
        self.new_factor(d, e).set_dense().set_iter(rand_iter)
        self.new_factor(d, f).set_dense().set_iter(rand_iter)
        self.new_factor(e, f).set_dense().set_iter(rand_iter)

        if include_unaries:
            self.new_factor(a).set_dense().set_iter(rand_iter)
            self.new_factor(b).set_dense().set_iter(rand_iter)
            self.new_factor(c).set_dense().set_iter(rand_iter)
            self.new_factor(d).set_dense().set_iter(rand_iter)
            self.new_factor(e).set_dense().set_iter(rand_iter)
            self.new_factor(f).set_dense().set_iter(rand_iter)
