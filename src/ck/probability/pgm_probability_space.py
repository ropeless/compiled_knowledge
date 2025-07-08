from typing import Sequence, Tuple

from ck.pgm import RandomVariable, Indicator, PGM
from ck.probability.probability_space import ProbabilitySpace, Condition, check_condition


class PGMProbabilitySpace(ProbabilitySpace):
    def __init__(self, pgm: PGM):
        """
        Enable probabilistic queries directly on a PGM.
        Note that this is not necessarily an efficient approach to calculating probabilities and statistics.

        Args:
            pgm: The PGM to query.

        Warning:
            The resulting probability space assumes that the value of the partition
            function, `z`, remains constant for the lifetime of the PGM. If the value
            changes, then probabilities and statistics will be incorrect.
        """
        self._pgm = pgm
        self._z = None

    @property
    def rvs(self) -> Sequence[RandomVariable]:
        return self._pgm.rvs

    def wmc(self, *condition: Condition) -> float:
        condition: Tuple[Indicator, ...] = check_condition(condition)
        return self._pgm.value_product_indicators(*condition)

    @property
    def z(self) -> float:
        if self._z is None:
            self._z = self._pgm.value_product_indicators()
        return self._z
