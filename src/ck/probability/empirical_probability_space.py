from typing import Sequence, Iterable, Tuple, Dict, List

from ck.pgm import RandomVariable, Indicator, Instance
from ck.probability.probability_space import ProbabilitySpace, Condition, check_condition


class EmpiricalProbabilitySpace(ProbabilitySpace):
    def __init__(self, rvs: Sequence[RandomVariable], samples: Iterable[Instance]):
        """
        Enable probabilistic queries over a sample from a sample space.
        Note that this is not necessarily an efficient approach to calculating probabilities and statistics.

        Assumes:
            len(sample) == len(rvs), for each sample in samples.
            0 <= sample[i] < len(rvs[i]), for each sample in samples, for i in range(len(rvs)).

        Args:
            rvs: The random variables.
            samples: instances (state indexes) that are samples from the given rvs.
        """
        self._rvs: Sequence[RandomVariable] = tuple(rvs)
        self._samples: List[Instance] = list(samples)
        self._rv_idx_to_sample_idx: Dict[int, int] = {
            rv.idx: i
            for i, rv in enumerate(self._rvs)
        }

    @property
    def rvs(self) -> Sequence[RandomVariable]:
        return self._rvs

    def wmc(self, *condition: Condition) -> float:
        condition: Tuple[Indicator, ...] = check_condition(condition)

        checks = [set() for _ in self._rvs]
        for ind in condition:
            checks[self._rv_idx_to_sample_idx[ind.rv_idx]].add(ind.state_idx)
        for i in range(len(checks)):
            if len(checks[i]) > 0:
                checks[i] = set(range(len(self._rvs[i]))).difference(checks[i])

        def satisfied(instance: Instance) -> bool:
            return not any((state in check) for state, check in zip(instance, checks))

        return sum(1 for _ in filter(satisfied, self._samples))

    @property
    def z(self) -> float:
        return len(self._samples)

