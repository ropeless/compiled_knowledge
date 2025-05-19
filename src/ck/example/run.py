import random as _random

from ck.pgm import PGM


class Run(PGM):
    """
    This PGM is the 'Run' Bayesian network.

    The Run Bayesian network is a sequence of random variables, x0, x1, x2, ... .
    The parents of each random variable are all the random variable
    earlier in the sequence.
    """

    def __init__(
            self,
            vars_per_run: int = 4,
            states_per_var: int = 2,
            sparsity: int = 0,
            random_seed: int = 123456,
    ):
        params = (vars_per_run, states_per_var, sparsity)
        super().__init__(f'{self.__class__.__name__}({",".join(str(param) for param in params)})')

        random_stream = _random.Random(random_seed).random
        for i in range(vars_per_run):
            self.new_rv(f'x{vars_per_run - i}', states_per_var)
            factor = self.new_factor(*tuple(reversed(self.rvs)))
            cpt = factor.set_cpt()
            cpt.set_random(random_stream, sparsity)
