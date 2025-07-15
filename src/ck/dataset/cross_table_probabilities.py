from typing import Sequence, Tuple, Dict

from ck.dataset.cross_table import CrossTable, Instance
from ck.pgm import RandomVariable, Indicator
from ck.probability.probability_space import ProbabilitySpace, Condition, check_condition


class CrossTableProbabilitySpace(ProbabilitySpace):
    def __init__(self, cross_table: CrossTable):
        """
        Enable probabilistic queries over a sample from a sample space.
        Note that this is not necessarily an efficient approach to calculating probabilities and statistics.

        Args:
            cross_table: a CrossTable to adapt to a ProbabilitySpace.
        """
        self._cross_table: CrossTable = cross_table
        self._rv_idx_to_sample_idx: Dict[int, int] = {
            rv.idx: i
            for i, rv in enumerate(cross_table.rvs)
        }

    @property
    def rvs(self) -> Sequence[RandomVariable]:
        return self._cross_table.rvs

    def wmc(self, *condition: Condition) -> float:
        condition: Tuple[Indicator, ...] = check_condition(condition)
        rvs: Sequence[RandomVariable] = self._cross_table.rvs

        checks = [set() for _ in rvs]
        for ind in condition:
            checks[self._rv_idx_to_sample_idx[ind.rv_idx]].add(ind.state_idx)
        for i in range(len(checks)):
            if len(checks[i]) > 0:
                checks[i] = set(range(len(rvs[i]))).difference(checks[i])

        def satisfied(item: Tuple[Instance, float]) -> float:
            """
            Return the weight of the instance, if the instance satisfies
            the condition, else return 0.
            """
            instance, weight = item
            if any((state in check) for state, check in zip(instance, checks)):
                return 0
            else:
                return weight

        return sum(map(satisfied, self._cross_table.items()))

    @property
    def z(self) -> float:
        return self._cross_table.total_weight()
