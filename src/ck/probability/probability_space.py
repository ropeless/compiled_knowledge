"""
An abstract class for object providing probabilities.
"""
import math
from abc import ABC, abstractmethod
from itertools import chain
from typing import Sequence, Tuple, Iterable, Callable

import numpy as np

from ck.pgm import Indicator, RandomVariable, rv_instances_as_indicators, number_of_states, rv_instances, Instance
from ck.utils.iter_extras import combos as _combos
from ck.utils.map_set import MapSet
from ck.utils.np_extras import dtype_for_number_of_states, NDArrayFloat64, DTypeStates, NDArrayNumeric

# Type defining a condition.
Condition = None | Indicator | Iterable[Indicator]


class ProbabilitySpace(ABC):
    """
    An abstract mixin class for a class providing probabilities over a state space defined by random variables.
    Each possible world of the state space is referred to as an 'instance'.
    """
    __slots__ = ()

    @property
    @abstractmethod
    def rvs(self) -> Sequence[RandomVariable]:
        """
        Return the random variables that define the state space.
        Each random variable, rv, has a length len(rv) which
        is the number of states, and rv[i] is the 'indicator' for
        the ith state of the random variable. Indicators must
        be unique across all rvs as rv[i] indicates the
        condition 'rv == i'.
        """

    @abstractmethod
    def wmc(self, *condition: Condition) -> float:
        """
        Return the weight of instances matching the given condition.

        If multiple indicators of the same random variable appear in
        the parameter 'indicators' then they are interpreted as
        a disjunction, otherwise indicators are interpreted as
        a conjunction. E.g.:  X=0, Y=1, Y=3  means X=0 and (Y=1 or Y=3)

        Args:
            condition: zero or more indicators that specify a condition.
        """

    @property
    @abstractmethod
    def z(self) -> float:
        """
        Return the summed weight of all instances.
        This is equivalent to self.wmc(), with no arguments.
        """

    def probability(self, *indicators: Indicator, condition: Condition = ()) -> float:
        """
        Return the joint probability of the given indicators,
        conditioned on any conditions, and
        marginalised over any unmentioned random variables.

        If multiple indicators of the same random variable appear in
        the parameters 'indicators' or 'condition' then they are interpreted as
        a disjunction, otherwise indicators are interpreted as
        a conjunction. E.g.:  X=0, Y=1, Y=3  means X=0 and (Y=1 or Y=3).

        Args:
            indicators: Indicators that specify which set of instances to compute probability.
            condition: Indicators that specify conditions for a conditional probability.
        Returns:
            the probability of the given indicators, conditioned on the given conditions.
        """
        condition: Tuple[Indicator, ...] = check_condition(condition)

        if len(condition) == 0:
            z = self.z
            if z <= 0:
                return np.nan
        else:
            z = self.wmc(*condition)
            if z <= 0:
                return np.nan

            # Combine the indicators with the condition
            # If a variable is mentioned in both the indicators and condition, then
            # we need to take the intersection, and check for contradictions.
            # If a variable is mentioned in the condition but not indicators, then
            # the rv condition needs to be added to the indicators.
            indicator_groups: MapSet[int, Indicator] = _group_indicators(indicators)
            condition_groups: MapSet[int, Indicator] = _group_indicators(condition)

            for rv_idx, indicators in condition_groups.items():
                indicator_group = indicator_groups.get(rv_idx)
                if indicator_group is None:
                    indicator_groups.add_all(rv_idx, indicators)
                else:
                    indicator_group.intersection_update(indicators)
                    if len(indicator_group) == 0:
                        # A contradiction between the indicators and conditions
                        return 0.0

            # Collect all the indicators from the updated indicator_groups
            indicators = chain(*indicator_groups.values())

        return self.wmc(*indicators) / z

    def marginal_distribution(self, *rvs: RandomVariable, condition: Condition = ()) -> NDArrayNumeric:
        """
        What is the marginal probability distribution over the states of the given random variables.
        Assumes that no indicators of rv in rvs appear in the conditions (if supplied).

        When multiple rvs are supplied, the order of instantiations is as per
        `rv_instances_as_indicators(*rvs)`.

        If multiple indicators of the same random variable appear in
        the parameter 'condition' then they are interpreted as
        a disjunction, otherwise indicators are interpreted as
        a conjunction. E.g.:  X=0, Y=1, Y=3  means X=0 and (Y=1 or Y=3).

        This is not an efficient implementation as it will call self.probability(...)
        for each possible state of the given random variable. If efficient marginal
        probability calculations are required, consider using a different method.

        Warning:
            If the probability of each state of rv (given the condition) is
            zero, then the marginal distribution is il-defined and the returned probabilities will
            all be NAN.

        Args:
            rvs: Random variables to compute the marginal distribution over.
            condition: Indicators that specify conditions for conditional probability.
            
        Returns:
            marginal probability distribution as an array co-indexed with `rv_instances_as_indicators(*rvs)`.
        """
        condition = check_condition(condition)

        # We have to be careful of the situation where indicators of rvs appear in condition.
        # If an RV has at least 1 indicator in condition then it must match it to have non-zero probability.
        wmc = self._get_wmc_for_marginals(rvs, condition)

        result: NDArrayFloat64 = np.fromiter(
            (wmc(indicators) for indicators in rv_instances_as_indicators(*rvs)),
            count=number_of_states(*rvs),
            dtype=np.float64
        )
        _normalise_marginal(result)
        return result

    def map(self, *rvs: RandomVariable, condition: Condition = ()) -> Tuple[float, Instance]:
        """
        Determine the maximum apriori probability (MAP).

        If there are tied solutions, one solution is returned, which
        is selected arbitrarily.

        If multiple indicators of the same random variable appear in
        the parameter 'condition' then they are interpreted as
        a disjunction, otherwise indicators are interpreted as
        a conjunction. E.g.: X=0, Y=1, Y=3  means X=0 and (Y=1 or Y=3)

        Warning:
            This is not an efficient implementation as it will call `self.wmc`
            for each possible state of the given random variables. If efficient MAP
            probability calculations are required, consider using a different method.

        Args:
            rvs: random variables to find the MAP over.
            condition: any conditioning indicators.

        Returns:
            (probability, instance) where
            probability: is the MAP probability
            instance: is the MAP state (co-indexed with the given rvs).
        """
        condition: Sequence[Indicator] = check_condition(condition)

        rv_indexes = set(rv.idx for rv in rvs)
        assert len(rv_indexes) == len(rvs), 'duplicated random variables not allowed'

        # Group conditioning indicators by random variable.
        conditions_by_rvs = _group_states(condition)

        # See if any MAP random variable is also conditioned.
        # Reduce the state space of any conditioned MAP rv.
        loop_rvs = []
        reduced_space = False
        for rv in rvs:
            states = conditions_by_rvs.get(rv.idx)
            if states is None:
                loop_rvs.append(rv)
            else:
                loop_rvs.append([rv[i] for i in sorted(states)])
                reduced_space = True

        # If the random variables we are looping over does not have any conditions
        # then it is expected to be faster by using computed marginal probabilities.
        if not reduced_space:
            prs = self.marginal_distribution(*rvs, condition=condition)
            best_probability = float('-inf')
            best_states = None
            for probability, inst in zip(prs, rv_instances(*rvs)):
                if probability > best_probability:
                    best_probability = probability
                    best_states = inst
            return best_probability, best_states

        else:
            # Remove any condition indicators with rv in rvs.
            new_conditions = tuple(ind for ind in condition if ind.rv_idx not in rv_indexes)

            # Loop over the state space of the 'loop' rvs
            best_probability = float('-inf')
            best_states = None
            indicators: Tuple[Indicator, ...]
            for indicators in _combos(loop_rvs):
                probability = self.wmc(*(indicators + new_conditions))
                if probability > best_probability:
                    best_probability = probability
                    best_states = tuple(ind.state_idx for ind in indicators)
            condition_probability = self.wmc(*condition)
            return best_probability / condition_probability, best_states

    def correlation(self, indicator1: Indicator, indicator2: Indicator, condition: Condition = ()) -> float:
        """
        What is the correlation between the two given indicators, r(indicator1, indicator2).
        
        Args:
            indicator1: a first random variable and its state.
            indicator2: a second random variable and its state.
            condition: any conditioning indicators.
            
        Returns:
            correlation between the two given indicators.
        """
        condition = check_condition(condition)

        p1 = self.probability(indicator1, condition=condition)
        p2 = self.probability(indicator2, condition=condition)
        p12 = self._joint_probability(indicator1, indicator2, condition=condition)
        d = p1 * (1.0 - p1) * p2 * (1.0 - p2)
        if d == 0.0:
            # As any marginal probability approaches zero, correlation approaches zero
            return 0.0
        else:
            return (p12 - p1 * p2) / math.sqrt(d)

    def entropy(self, rv: RandomVariable, condition: Condition = ()) -> float:
        """
        Calculate the entropy of the given random variable, H(rv).
        
        Args:
            rv: random variable to calculate the entropy for.
            condition: any conditioning indicators.
            
        Returns:
            entropy of the given random variable.
        """
        condition = check_condition(condition)
        e = 0.0
        for ind in rv:
            p = self.probability(ind, condition=condition)
            if p > 0.0:
                e -= p * math.log2(p)
        return e

    def conditional_entropy(self, rv1: RandomVariable, rv2: RandomVariable, condition: Condition = ()) -> float:
        """
        Calculate the conditional entropy, H(rv1 | rv2).
        
        Args:
            rv1: random variable to calculate the entropy for.
            rv2: the conditioning random variable for entropy calculation.
            condition: any conditioning indicators to restrict the state space.
            
        Returns:
            entropy of rv1, conditioned on rv2.
        """
        condition = check_condition(condition)
        e = 0.0
        for ind1 in rv1:
            for ind2 in rv2:
                p = self._joint_probability(ind1, ind2, condition=condition)
                if p > 0.0:
                    # if p > 0 then p2 > 0, as p <= p2
                    p2 = self.probability(ind2, condition=condition)
                    e -= p * math.log2(p / p2)
        return e

    def joint_entropy(self, rv1: RandomVariable, rv2: RandomVariable, condition: Condition = ()) -> float:
        """
        Calculate the joint entropy of the two random variables, H(rv1; rv2).
        
        Args:
            rv1: a first random variable to calculate joint entropy.
            rv2: a second random variable to calculate joint entropy.
            condition: any conditioning indicators to restrict the state space.
        Returns:
            joint entropy of the given random variables.
        """
        condition = check_condition(condition)
        e = 0.0
        for ind1 in rv1:
            for ind2 in rv2:
                p = self._joint_probability(ind1, ind2, condition=condition)
                if p > 0.0:
                    e -= p * math.log2(p)
        return e

    def mutual_information(self, rv1: RandomVariable, rv2: RandomVariable, condition: Condition = ()) -> float:
        """
        Calculate the mutual information between two random variables, I(rv1; rv2).

        Args:
            rv1: a first random variable
            rv2: a second random variable
            condition: indicators to specify a condition restricting the state space.
        Returns:
            mutual_information(rv1, rv2) / denominator
        """
        condition = check_condition(condition)
        p1s = self.marginal_distribution(rv1, condition=condition)
        p2s = self.marginal_distribution(rv2, condition=condition)
        info = 0.0
        for ind1, p1 in zip(rv1, p1s):
            for ind2, p2 in zip(rv2, p2s):
                p12 = self._joint_probability(ind1, ind2, condition=condition)
                if p12 > 0.0:
                    info += p12 * math.log2(p12 / p1 / p2)
        return info

    def total_correlation(self, rv1: RandomVariable, rv2: RandomVariable, condition: Condition = ()) -> float:
        """
        Calculate the 'total correlation' measure.
        total_correlation = I(rv1; rv2) / min(H(rv1), H(rv2)).
        This is a normalised mutual information between two random variables.
        0 => no mutual information.
        1 => perfect mutual information.

        Args:
            rv1: a first random variable
            rv2: a second random variable
            condition: indicators to specify a condition restricting the state space.
        Returns:
            total correlation between the given random variables.
        """
        condition = check_condition(condition)
        denominator = min(self.entropy(rv1), self.entropy(rv2, condition=condition))
        return self._normalised_mutual_information(rv1, rv2, denominator, condition=condition)

    def uncertainty(self, rv1: RandomVariable, rv2: RandomVariable, condition: Condition = ()) -> float:
        """
        Calculate the 'uncertainty' measure, C, between two random variables
        C(rv1, rv2) = I(rv1; rv2) / H(rv2)
        This is a normalised mutual information between two random variables.
        Note that it is not a symmetric measure; in general C(rv1, rv2) does not equal C(rv2, rv1).
        0 => no mutual information.
        1 => perfect mutual information.

        Args:
            rv1: a first random variable
            rv2: a second random variable
            condition: indicators to specify a condition restricting the state space.
        Returns:
            uncertainty between the given random variables.
        """
        condition = check_condition(condition)
        denominator = self.entropy(rv2, condition=condition)
        return self._normalised_mutual_information(rv1, rv2, denominator, condition=condition)

    def symmetric_uncertainty(self, rv1: RandomVariable, rv2: RandomVariable, condition: Condition = ()) -> float:
        """
        Calculate the 'symmetric uncertainty' measure.
        symmetric_uncertainty = 2 * I(rv1, rv2) / (H(rv1) + H(rv2)).
        This is the harmonic mean of the two uncertainty coefficients,
        C(rv1, rv2) = I(rv1; rv2) / H(rv2) and C(rv2, rv1) = I(rv1; rv2) / H(rv1).
        This is a normalised mutual information between two random variables.
        0 => no mutual information.
        1 => perfect mutual information.

        Args:
            rv1: a first random variable
            rv2: a second random variable
            condition: indicators to specify a condition restricting the state space.
        Returns:
            symmetric uncertainty between the given random variables.
        """
        condition = check_condition(condition)
        denominator = self.entropy(rv1) + self.entropy(rv2, condition=condition)
        return 2.0 * self._normalised_mutual_information(rv1, rv2, denominator, condition=condition)

    def iqr(self, rv1: RandomVariable, rv2: RandomVariable, condition: Condition = ()) -> float:
        """
        Calculate the Information Quality Ratio (IQR).
        IQR = I(rv1; rv2) / H(rv1; rv2).
        Also known as 'dual total correlation'.
        This is a normalised mutual information between two random variables.
        0 => no mutual information.
        1 => perfect mutual information.

        Args:
            rv1: a first random variable
            rv2: a second random variable
            condition: indicators to specify a condition restricting the state space.
        Returns:
            Information Quality Ratio between the given random variables.
        """
        condition = check_condition(condition)
        denominator = self.joint_entropy(rv1, rv2, condition=condition)
        return self._normalised_mutual_information(rv1, rv2, denominator, condition=condition)

    def covariant_normalised_mutual_information(self, rv1: RandomVariable, rv2: RandomVariable,
                                                condition: Condition = ()) -> float:
        """
        Calculate the covariant normalised mutual information
        = I(rv1; rv2) / sqrt(H(rv1) * H(rv2)).
        This is a normalised mutual information between two random variables.
        0 => no mutual information.
        1 => perfect mutual information.

        Args:
            rv1: a first random variable
            rv2: a second random variable
            condition: indicators to specify a condition restricting the state space.
        Returns:
            covariant normalised mutual information between the given random variables.
        """
        condition = check_condition(condition)
        denominator = math.sqrt(self.entropy(rv1, condition=condition) * self.entropy(rv2, condition=condition))
        return self._normalised_mutual_information(rv1, rv2, denominator, condition=condition)

    def _normalised_mutual_information(
            self,
            rv1: RandomVariable,
            rv2: RandomVariable,
            denominator: float,
            condition: Tuple[Indicator, ...],
    ) -> float:
        """
        Helper function for normalised mutual information calculations.

        Args:
            rv1: a first random variable
            rv2: a second random variable
            denominator: the normalisation factor
            condition: indicators to specify a condition restricting the state space.
        Returns:
            mutual_information(rv1, rv2) / denominator
        """
        if denominator == 0.0:
            return 0.0
        else:
            return self.mutual_information(rv1, rv2, condition) / denominator

    def _joint_probability(
            self,
            indicator1: Indicator,
            indicator2: Indicator,
            condition: Tuple[Indicator, ...],
    ) -> float:
        """
        Helper function to correctly calculate a joint probability even if the two indicators
        are from the same random variable.

        If the indicators are from the different random variables then
        probability(indicator1 and indicator2 | condition).

        If the indicators are from the same random variable then
        probability(indicator1 or indicator2 | condition).

        Args:
            indicator1: a first Indicator.
            indicator2: a second Indicator
            condition: indicators to specify a condition restricting the state space.
        Returns:
            joint probability of the two indicators, given the condition.
        """
        if indicator1 == indicator2:
            # Ensure correct behaviour, same random variable and same states
            return self.probability(indicator1, condition=condition)
        elif indicator1.rv_idx == indicator2.rv_idx:
            # Efficiency shortcut, same random variable but different states
            return 0.0
        else:
            # General case, two different random variables
            return self.probability(indicator1, indicator2, condition=condition)

    def _get_wmc_for_marginals(
            self,
            rvs: Sequence[RandomVariable],
            condition: Tuple[Indicator, ...],
    ) -> Callable[[Sequence[Indicator]], float]:
        """
        Return a wmc function that is suitable for calculating marginal distributions.

        This implementation is careful of the situation where indicators of rvs appear in condition.
        If an RV has at least 1 indicator in condition then it must match it to have non-zero probability.

        Args:
            rvs: random variables to calculate marginal distributions for.
            condition: indicators to specify a condition restricting the state space.
        Returns:
            A function from a condition, specified as a sequence of indicators, to a weighted model count.
        """
        if len(condition) > 0:
            check_sets = []
            overlap_detected = False
            cond_set = set(condition)
            for rv in rvs:
                in_condition = set()
                for ind in rv:
                    if ind in cond_set:
                        in_condition.add(ind)
                        cond_set.discard(ind)
                        overlap_detected = True
                if len(in_condition) == 0:
                    in_condition.update(rv)
                check_sets.append(in_condition)

            if overlap_detected:
                __wmc__condition = tuple(cond_set)

                def wmc(indicators: Sequence[Indicator]) -> float:
                    for indicator, check_set in zip(indicators, check_sets):
                        if indicator not in check_set:
                            return 0.0
                    full_condition = tuple(indicators) + __wmc__condition
                    return self.wmc(*full_condition)
            else:
                __wmc__condition = tuple(condition)

                def wmc(indicators: Sequence[Indicator]) -> float:
                    full_condition = tuple(indicators) + __wmc__condition
                    return self.wmc(*full_condition)
        else:
            def wmc(indicators: Sequence[Indicator]) -> float:
                return self.wmc(*indicators)

        return wmc


def check_condition(condition: Condition) -> Tuple[Indicator, ...]:
    """
    Make the best effort to interpret the given condition.

    Args:
        condition: a relaxed specification of a condition.
    Returns:
        a formal specification of the condition as a tuple of indicators with no duplicates.
    """
    if condition is None:
        return ()
    elif isinstance(condition, Indicator):
        return (condition,)
    else:
        return tuple(set(condition))


def dtype_for_state_indexes(rvs: Iterable[RandomVariable]) -> DTypeStates:
    """
    Infer a numpy dtype to hold any state index from any given random variable.

    Args:
        rvs: some random variables.
    Returns:
        a numpy dtype.
    """
    return dtype_for_number_of_states(max((len(rv) for rv in rvs), default=0))


def _group_indicators(indicators: Iterable[Indicator]) -> MapSet[int, Indicator]:
    """
    Group the given indicators by rv_idx.

    Args:
        indicators: the indicators to group.

    Returns:
        A mapping from rv_idx to set of indicators.
    """
    groups: MapSet[int, Indicator] = MapSet()
    for indicator in indicators:
        groups.add(indicator.rv_idx, indicator)
    return groups


def _group_states(indicators: Iterable[Indicator]) -> MapSet[int, int]:
    """
    Group the given indicator states by rv_idx.

    Args:
        indicators: the indicators to group.

    Returns:
        A mapping from rv_idx to set of state indexes.
    """
    groups: MapSet[int, int] = MapSet()
    for indicator in indicators:
        groups.add(indicator.rv_idx, indicator.state_idx)
    return groups


def _normalise_marginal(distribution: NDArrayFloat64) -> None:
    """
    Update the values in the given distribution to
    properly represent a marginal distribution.

    The update is made in-place.

    Args:
        a 1D numpy array of likelihoods.
    """
    total = np.sum(distribution)
    if total <= 0:
        distribution[:] = np.nan
    elif total != 1:
        distribution /= total
