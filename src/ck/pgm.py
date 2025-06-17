"""
For more documentation on this module, refer to the Jupyter notebook docs/4_PGM_advanced.ipynb.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import repeat as _repeat
from typing import Sequence, Tuple, Dict, Optional, overload, Set, Iterable, List, Union, Callable, \
    Collection, Any, Iterator

import numpy as np

from ck.utils.iter_extras import (
    combos_ranges as _combos_ranges, multiply as _multiply, combos as _combos
)
from ck.utils.np_extras import NDArrayFloat64, NDArrayUInt8

# What types are permitted as random variable states
State = Union[int, str, bool, float, None]

# An instance (of a sequence of random variables) is a tuple of integers
# that are state indexes, co-indexed with a known sequence of random variables.
Instance = Sequence[int]

# A key identifies an instance.
# A single integer is treated as an instance with one dimension.
Key = Union[Sequence[int], int]

# The shape of a sequence of random variables (e.g., a PGM, Factor or PotentialFunction).
Shape = Sequence[int]

DEFAULT_TOLERANCE: float = 0.000001  # For checking CPT sums.


class PGM:
    """
    A probabilistic graphical model (PGM) represents a joint probability distribution over
    a set of random variables. Specifically, a PGM is a factor graph with discrete random variables.

    Add a random variable to a PGM, pgm, using `rv = pgm.new_rv(...)`.

    Add a factor to the PGM, pgm, using `factor = pgm.new_factor(...)`.

    A PGM may be given a human-readable name.
    """

    def __init__(self, name: Optional[str] = None):
        """
        Create an empty PGM.

        Args:
            name: an optional name for the PGM. If not provided, a default name will be
                created using `default_pgm_name`.
        """
        self._name: str = name if name is not None else default_pgm_name(self)
        self._rvs: Tuple[RandomVariable, ...] = ()
        self._shape: Shape = ()
        self._indicators: Tuple[Indicator, ...] = ()
        self._factors: Tuple[Factor, ...] = ()

    @property
    def name(self) -> str:
        """
        Returns:
            The name of the PGM.
        """
        return self._name

    @property
    def number_of_rvs(self) -> int:
        """
        Returns:
            How many random variables are defined in this PGM.
        """
        return len(self._rvs)

    @property
    def shape(self) -> Shape:
        """
        Returns:
            a sequence of the lengths of `self.rvs`.
        """
        return self._shape

    @property
    def number_of_indicators(self) -> int:
        """
        Returns:
            How many indicators are defined in this PGM, i.e., `sum(len(rv) for rv in self.rvs)`.
        """
        return len(self._indicators)

    @property
    def number_of_states(self) -> int:
        """
        Returns:
            What is the size of the state space, i.e., `multiply(len(rv) for rv in self.rvs)`.
        """
        return number_of_states(*self._rvs)

    @property
    def number_of_factors(self) -> int:
        """
        Returns:
            How many factors are defined in this PGM.
        """
        return len(self._factors)

    @property
    def number_of_functions(self) -> int:
        """
        Returns:
            How many potential functions are defined in this PGM, including zero potential functions.
        """
        return sum(1 for _ in self.functions)

    @property
    def number_of_non_zero_functions(self) -> int:
        """
        Returns:
            How many potential functions are defined in this PGM, excluding zero potential functions.
        """
        return sum(1 for _ in self.non_zero_functions)

    @property
    def rvs(self) -> Sequence[RandomVariable]:
        """
        Returns:
            All the random variables, in `idx` order, which is the same as creation order.

        Ensures:
            `self.rvs[rv.idx] = rv`
        """
        return self._rvs

    @property
    def rv_log_sizes(self) -> Sequence[float]:
        """
        Returns:
            [log2(len(rv)) for rv in self.rvs]
        """
        return [math.log2(len(rv)) for rv in self.rvs]

    @property
    def indicators(self) -> Sequence[Indicator]:
        """
        Returns:
            All the random variable indicators.

        Ensures:
            the indicators of a random variable are adjacent,
            the indicators of a random variable are in state index order,
            the random variables are in the same order as `self.rvs`.
        """
        return self._indicators

    @property
    def factors(self) -> Sequence[Factor]:
        """
        Returns:
            All the factors, in `idx` order, which is the same as creation order.

        Ensures:
            `self.factors[factor.idx] = factor`
        """
        return self._factors

    @property
    def functions(self) -> Iterable[PotentialFunction]:
        """
        Iterate over all in-use potential functions of this PGM, including
        zero potential functions.

        Returns:
            An Iterable over all potential functions (including zero potential functions).
        """
        seen: Set[int] = set()
        for factor in self._factors:
            function = factor.function
            if id(function) not in seen:
                seen.add(id(function))
                yield function

    @property
    def non_zero_functions(self) -> Iterable[PotentialFunction]:
        """
        Iterate over all in-use potential functions of this PGM, excluding
        zero potential functions.

        Returns:
            An Iterable over all potential functions (excluding zero potential functions).
        """
        seen: Set[int] = set()
        for factor in self._factors:
            function = factor.function
            if not (isinstance(function, ZeroPotentialFunction) or id(function) in seen):
                seen.add(id(function))
                yield function

    def new_rv(self, name: str, states: Union[int, Sequence[State]]) -> RandomVariable:
        """
        Add a new random variable to this PGM.

        The returned random variable will have an `idx` equal to the value of
        `self.number_of_rvs` just prior to adding the new random variable.

        Assumes:
            Provided states contain no duplicates.

        Args:
            name: a name for the random variable.
            states: either an integer number of states or a sequence of state values. If a
                single integer, `n`, is provided then the states will be 0, 1, ..., n-1.
                If a sequence of states are provided then the states must be unique.

        Returns:
            a RandomVariable object belonging to this PGM.
        """
        return RandomVariable(self, name, states)

    def new_factor(self, *rvs: RandomVariable) -> Factor:
        """
        Add a new factor to this PGM where the factor connects
        the given random variables.

        The returned factor will have a ZeroPotentialFunction as its potential function.
        The potential function may be changed by calling methods on the returned factor.

        The returned factor will have an `idx` equal to the value of
        `self.number_of_factors` just prior to adding the new factor.

        Assumes:
            The given random variables all belong to this PGM.
            The random variables contain no duplicates.

        Args:
            *rvs: the random variables.

        Returns:
            a Factor object belonging to this PGM.
        """
        return Factor(self, *rvs)

    def new_factor_implies(
            self,
            rv_1: RandomVariable,
            state_idxs_1: int | Collection[int],
            rv_2: RandomVariable,
            state_idxs_2: int | Collection[int],
    ) -> Factor:
        """
        Add a sparse 0/1 factor to this PGM representing:
            rv_1 in state_idxs_1  ==>  rv_2 in states_2.
        That is:
            factor[s1, s2] = 1, if s1 not in state_idxs_1 or s2 in states_2;
                           = 0, otherwise.

        Args:
            rv_1: The first random variable.
            state_idxs_1: state idxs of the first random variable.
            rv_2: The second random variable.
            state_idxs_2: state idxs of the second random variable.

        Returns:
            a Factor object belonging to this PGM, with a configured sparse potential function.
        """
        if isinstance(state_idxs_1, int):
            state_idxs_1 = (state_idxs_1,)
        if isinstance(state_idxs_2, int):
            state_idxs_2 = (state_idxs_2,)

        factor = self.new_factor(rv_1, rv_2)
        f = factor.set_sparse()
        for i_1 in rv_1.state_range():
            if i_1 not in state_idxs_1:
                for i_2 in rv_2.state_range():
                    f[i_1, i_2] = 1
            else:
                for i_2 in rv_2.state_range():
                    if i_2 in state_idxs_2:
                        f[i_1, i_2] = 1
        return factor

    def new_factor_equiv(
            self,
            rv_1: RandomVariable,
            state_idxs_1: int | Collection[int],
            rv_2: RandomVariable,
            state_idxs_2: int | Collection[int],
    ) -> Factor:
        """
        Add a sparse 0/1 factor to this PGM representing:
            rv_1 in state_idxs_1  <==>  rv_2 in state_idxs_2.
        That is:
            factor[s1, s2] = 1, if s1 in state_idxs_1 == s2 in state_idxs_2;
                           = 0, otherwise.

        Args:
            rv_1: The first random variable.
            state_idxs_1: state idxs of the first random variable.
            rv_2: The second random variable.
            state_idxs_2: state idxs of the second random variable.

        Returns:
            a Factor object belonging to this PGM, with a configured sparse potential function.
        """
        if isinstance(state_idxs_1, int):
            state_idxs_1 = (state_idxs_1,)
        if isinstance(state_idxs_2, int):
            state_idxs_2 = (state_idxs_2,)

        factor = self.new_factor(rv_1, rv_2)
        f = factor.set_sparse()
        for i_1 in rv_1.state_range():
            in_1 = i_1 in state_idxs_1
            for i_2 in rv_2.state_range():
                in_2 = i_2 in state_idxs_2
                if in_1 == in_2:
                    f[i_1, i_2] = 1
        return factor

    def new_factor_functional(
            self,
            function: Callable[[...], int],
            result_rv: RandomVariable,
            *input_rvs: RandomVariable
    ) -> Factor:
        """
        Add a sparse 0/1 factor to this PGM representing:
            result_rv ==  function(*rvs).
        That is:
            factor[result_s, *input_s] = 1, if result_s == function(*input_s);
                                       = 0, otherwise.
        Args:
            function: a function from state indexes of the input random variables to a state index
                of the result random variable. The function should take the same number of arguments
                as `input_rvs` and return a state index for `result_rv`.
            result_rv: the random variable defining result values.
            *input_rvs: the random variables defining input values.

        Returns:
            a Factor object belonging to this PGM, with a configured sparse potential function.
        """
        factor = self.new_factor(result_rv, *input_rvs)
        f = factor.set_sparse()
        for input_s in _combos([list(rv.state_range()) for rv in input_rvs]):
            result_s = function(*input_s)
            f[(result_s,) + input_s] = 1
        return factor

    def indicator_pair(self, indicator: Indicator) -> Tuple[RandomVariable, State]:
        """
        Convert the given indicator to its RandomVariable and State value.

        Args:
            indicator: the indicator to convert.

        Returns:
            (rv, state) where
            rv: is the random variable of the indicator.
            state: is the random variable state of the indicator.
        """
        rv = self._rvs[indicator.rv_idx]
        state = rv.states[indicator.state_idx]
        return rv, state

    def indicator_str(self, *indicators: Indicator, sep: str = '=', delim: str = ', ') -> str:
        """
        Render indicators as a string.

        For example:
            pgm = PGM()
            a = pgm.new_rv('A', ('x', 'y', 'z'))
            b = pgm.new_rv('B', (3, 5))
            print(pgm.indicator_str(a[0], b[1], a[2]))
        will print:
            A=x, B=5, A=z

        Args:
            *indicators: the indicators to render.
            sep: the separator to use between the random variable and its state.
            delim: the delimiter to used when rendering multiple indicators.

        Returns:
            a string representation of the given indicators.
        """
        return delim.join(
            f'{_clean_str(rv)}{sep}{_clean_str(state)}'
            for rv, state in (
                self.indicator_pair(indicator)
                for indicator in indicators
            )
        )

    def condition_str(self, *indicators: Indicator) -> str:
        """
        Render indicators as a string, grouping indicators by random variable.

        For example:
            pgm = PGM()
            a = pgm.new_rv('A', ('x', 'y', 'z'))
            b = pgm.new_rv('B', (3, 5))
            print(pgm.condition_str(a[0], b[1], a[2]))
        will print:
            A in {x, z}, B=5

        Args:
            *indicators: the indicators to render.
        Return:
            a string representation of the given indicators, as a condition.
        """
        indicators: List[Indicator] = sorted(indicators, reverse=True)
        cur_rv: Set[Indicator] = set()
        cur_idx: int = -1  # rv_idx of the rv we are currently working on, -1 means not yet started.
        cur_str: str = ''  # accumulated result string
        while len(indicators) > 0:
            this_ind = indicators.pop()
            if this_ind.rv_idx != cur_idx:
                if cur_idx >= 0:
                    cur_str = self._condition_str_rv(cur_str, cur_rv)
                cur_rv = set()
                cur_idx = this_ind.rv_idx
            cur_rv.add(this_ind)
        if cur_idx >= 0:
            cur_str = self._condition_str_rv(cur_str, cur_rv)
        return cur_str

    def instance_str(
            self,
            instance: Instance,
            rvs: Optional[Sequence[RandomVariable]] = None,
            sep: str = '=',
            delim: str = ', ',
    ) -> str:
        """
        Render an instance as a string.

        The result looks something like 'X=x, Y=y, Z=z' where X, Y, and X are
        random variables and x, y, and z are the states represented by the
        given instance.

        Args:
            instance: the instance to render.
            rvs: the random variables that the instance refers to. If rvs is None, then `self.rvs` is used.
            sep: the separator to use between the random variable and its state.
            delim: the delimiter to used when rendering multiple indicators.

        Returns:
            a string representation of the indicators implied by the given instance.
        """
        if rvs is None:
            rvs = self.rvs
        assert len(instance) == len(rvs)
        return self.indicator_str(
            *[rv[state] for rv, state in zip(rvs, instance)],
            sep=sep,
            delim=delim
        )

    def state_str(
            self,
            instance: Instance,
            rvs: Optional[Sequence[RandomVariable]] = None,
            delim: str = ', ',
    ) -> str:
        """
        Render the states of an instance.

        The result looks something like 'x, y, z' where x, y, and z are
        the states of the random variables represented by the given instance.

        Args:
            instance: the instance to render.
            rvs: the random variables that the instance refers to. If rvs is None, then `self.rvs` is used.
            delim: the delimiter to used when rendering multiple indicators.

        Returns:
            a string representation of the states implied by the given instance.
        """
        if rvs is None:
            rvs = self.rvs
        assert len(instance) == len(rvs)
        return delim.join(str(rv.states[i]) for rv, i in zip(rvs, instance))

    def instances(self, flip: bool = False) -> Iterable[Instance]:
        """
        Iterate over all possible instances of this PGM, in natural index
        order (i.e., last random variable changing most quickly).

        Args:
            flip: if true, then first random variable changes most quickly.

        Returns:
            an iteration over tuples, each tuple holds random variable state indexes
            co-indexed with this PGM's random variables, `self.rvs`.
        """
        return _combos_ranges(tuple(len(rv) for rv in self._rvs), flip=not flip)

    def instances_as_indicators(self, flip: bool = False) -> Iterable[Sequence[Indicator]]:
        """
        Iterate over all possible instances of this PGM, in natural index
        order (i.e., last random variable changing most quickly).

        Args:
            flip: if true, then first random variable changes most quickly.

        Returns:
            an iteration over tuples, each tuples holds random variable indicators
            co-indexed with this PGM's random variables, `self.rvs`.
        """
        for inst in self.instances(flip=flip):
            yield self.state_idxs_to_indicators(inst)

    def state_idxs_to_indicators(self, instance: Sequence[int]) -> Sequence[Indicator]:
        """
        Given an instance (list of random variable state indexes), co-indexed with the PGM's
        random variables, `self.rvs`, return the corresponding indicators.

        Assumes:
            The instance has the same length as `self.rvs`.
            The instance is co-indexed with `self.rvs`.

        Args:
            instance: the instance to convert to indicators.

        Returns:
            a tuple of indicators, co-indexed with `self.rvs`.
        """
        return tuple(rv[state] for rv, state in zip(self._rvs, instance))

    def factor_values(self, key: Key) -> Iterable[float]:
        """
        For a given instance key, each factor defines a single value. This method
        returns those values.

        Args:
            key: the key defining an instance of this PGM.

        Returns:
            an iterator over factor values, co-indexed with the factors of this PGM.
        """
        instance: Instance = check_key(self._shape, key)
        assert len(instance) == len(self._rvs)
        for factor in self._factors:
            states: Sequence[int] = tuple(instance[rv.idx] for rv in factor.rvs)
            value: float = factor.function[states]
            yield value

    @property
    def is_structure_bayesian(self) -> bool:
        """
        Does the PGM structure correspond to a Bayesian network, where
        each factor is taken to be a CPT and the first random variable of factor
        is taken to be the child.

        This method does not check the factor parameters to confirm they correspond
        to valid CPTs.

        Return:
            True only if:
                the number of factors equals the number of random variables,
                each random variable appears exactly once as the first random variable of a factor,
                there are no directed loops created by the factors.
        """

        # One factor per random variable.
        if self.number_of_factors != self.number_of_rvs:
            return False

        # Each random variable is a child.
        # Map each random variable to the factor it is a child of
        child_to_factor: Dict[int, Factor] = {
            factor.rvs[0].idx: factor
            for factor in self._factors
        }
        if len(child_to_factor) != self.number_of_rvs:
            return False

        # Factors form a DAG
        states: NDArrayUInt8 = np.zeros(self.number_of_factors, dtype=np.uint8)
        for factor in self._factors:
            if self._has_cycle(factor, child_to_factor, states):
                return False

        # All tests passed
        return True

    def factors_are_cpts(self, tolerance: float = DEFAULT_TOLERANCE) -> bool:
        """
        Are all factor potential functions set with parameters values
        conforming to Conditional Probability Tables.

        Assumes:
            tolerance is non-negative.

        Args:
            tolerance: a tolerance when testing if values are equal to zero or one.

        Returns:
            True only if every potential function conforms to being a valid CPT.
        """
        return all(function.is_cpt(tolerance) for function in self.functions)

    def check_is_bayesian_network(self, tolerance: float = DEFAULT_TOLERANCE) -> bool:
        """
        Is this PGM a Bayesian network.

        Assumes:
            tolerance is non-negative.

        Args:
            tolerance: a tolerance when testing if values are equal to zero or one.

        Returns:
            `is_structure_bayesian and check_factors_are_cpts(tolerance)`.
        """
        return self.is_structure_bayesian and self.factors_are_cpts(tolerance)

    def value_product(self, key: Key) -> float:
        """
        For a given instance key, each factor defines a single value. This method
        returns the product of those values.

        Args:
            key: the key defining an instance of this PGM.

        Returns:
            the product of factor values.
        """
        return _multiply(self.factor_values(key))

    def value_product_indicators(self, *indicators: Indicator) -> float:
        """
        Return the product of factors, conditioned on the given indicators.

        For random variables not mentioned in the indicators, then the result is the sum
        of the value product for each possible combination of states of the unmentioned
        random variables.

        If no indicators are provided, then the value of the partition function (z)
        is returned.

        If multiple indicators are provided for the same random variable, then all matching
        instances are summed.

        This method has the same semantics as `ProbabilitySpace.wmc` without conditioning.

        Warning:
            this is potentially computationally expensive as it marginalised random
            variables not mentioned in the given indicators.

        Args:
            *indicators: are indicators from random variables of this PGM.

        Returns:
            the product of factors, conditioned on the given instance. This is the
            computed value of the PGM, conditioned on the given instance.
        """
        # # Create a filter from the indicators
        # inst_filter: List[Set[int]] = [set() for _ in range(self.number_of_rvs)]
        # for indicator in indicators:
        #     rv_idx: int = indicator.rv_idx
        #     inst_filter[rv_idx].add(indicator.state_idx)
        # # Collect rvs not mentioned - to marginalise
        # for rv, rv_filter in zip(self.rvs, inst_filter):
        #     if len(rv_filter) == 0:
        #         rv_filter.update(rv.state_range())
        #
        # def _sum_inst(_instance: Instance) -> bool:
        #     return all(
        #         (_state in _rv_filter)
        #         for _state, _rv_filter in zip(_instance, inst_filter)
        #     )
        #
        # # Accumulate the result
        # sum_value = 0
        # for instance in self.instances():
        #     if _sum_inst(instance):
        #         sum_value += self.value_product(instance)
        #
        # return sum_value

        # Work out the space to sum over
        sum_space_set: List[Optional[Set[int]]] = [None] * self.number_of_rvs
        for indicator in indicators:
            rv_idx: int = indicator.rv_idx
            cur_set = sum_space_set[rv_idx]
            if cur_set is None:
                sum_space_set[rv_idx] = cur_set = set()
            cur_set.add(indicator.state_idx)

        # Convert to a list of states that we need to sum over.
        sum_space_list: List[List[int]] = [
            list(cur_set if cur_set is not None else rv.state_range())
            for cur_set, rv in zip(sum_space_set, self.rvs)
        ]

        # Accumulate the result
        return sum(
            self.value_product(instance)
            for instance in _combos(sum_space_list)
        )

    def dump_synopsis(
            self,
            *,
            prefix: str = '',
            precision: int = 3,
            max_state_digits: int = 21,
    ):
        """
        Print a synopsis of the PGM.
        This is intended for demonstration and debugging purposes.

        Args:
            prefix: optional prefix for indenting all lines.
            precision: a limit on the render precision of floating point numbers.
            max_state_digits: a limit on the number of digits when showing number of states as an integer.
        """
        # limit the precision when displaying number of states
        num_states: int = self.number_of_states
        number_of_parameters = sum(function.number_of_parameters for function in self.functions)
        number_of_nz_parameters = sum(function.number_of_parameters for function in self.non_zero_functions)

        if math.log10(num_states) > max_state_digits:
            log_states = math.log10(num_states)
            exp = int(log_states)
            man = math.pow(10, log_states - exp)
            num_states_str = f'{man:,.{precision}f}e+{exp}'
        else:
            num_states_str = f'{num_states:,}'

        log_2_num_states = math.log2(num_states)
        if (
                log_2_num_states == 0
                or (
                    log_2_num_states == int(log_2_num_states)
                    and math.log10(log_2_num_states) <= max_state_digits
                )
        ):
            log_2_num_states_str = f'{int(log_2_num_states):,}'
        else:
            log_2_num_states_str = f'{math.log2(num_states):,.{precision}f}'

        print(f'{prefix}name: {self.name}')
        print(f'{prefix}number of random variables: {self.number_of_rvs:,}')
        print(f'{prefix}number of indicators: {self.number_of_indicators:,}')
        print(f'{prefix}number of states: {num_states_str}')
        print(f'{prefix}log 2 of states: {log_2_num_states_str}')
        print(f'{prefix}number of factors: {self.number_of_factors:,}')
        print(f'{prefix}number of functions: {self.number_of_functions:,}')
        print(f'{prefix}number of non-zero functions: {self.number_of_non_zero_functions:,}')
        print(f'{prefix}number of parameters: {number_of_parameters:,}')
        print(f'{prefix}number of functions (excluding ZeroPotentialFunction): {self.number_of_non_zero_functions:,}')
        print(f'{prefix}number of parameters (excluding ZeroPotentialFunction): {number_of_nz_parameters:,}')
        print(f'{prefix}Bayesian structure: {self.is_structure_bayesian}')
        print(f'{prefix}CPT factors: {self.factors_are_cpts()}')

    def dump(
            self,
            *,
            prefix: str = '',
            indent: str = '  ',
            show_function_values: bool = False,
            precision: int = 3,
            max_state_digits: int = 21,
    ) -> None:
        """
        Print a dump of the PGM.
        This is intended for demonstration and debugging purposes.

        Args:
            prefix: optional prefix for indenting all lines.
            show_function_values: if true, then the function values will be dumped.
            indent: additional prefix to use for extra indentation.
            precision: a limit on the render precision of floating point numbers.
            max_state_digits: a limit on the number of digits when showing number of states as an integer.
        """

        next_prefix: str = prefix + indent
        next_next_prefix: str = next_prefix + indent

        print(f'{prefix}PGM id={id(self)} name={self.name!r}')
        self.dump_synopsis(prefix=next_prefix, precision=precision, max_state_digits=max_state_digits)

        print(f'{prefix}random variables ({self.number_of_rvs})')
        for rv in self.rvs:
            print(f'{next_prefix}{rv.idx:>3} {rv.name!r} ({len(rv)})', end='')
            if not rv.is_default_states():
                print(' [', end='')
                print(', '.join(repr(s) for s in rv.states), end='')
                print(']', end='')
            print()

        print(f'{prefix}factors ({self.number_of_factors})')
        for factor in self.factors:
            rv_idxs = [rv.idx for rv in factor.rvs]
            if factor.is_zero:
                function_ref = '<zero>'
            else:
                function = factor.function
                function_ref = f'{id(function)}: {function.__class__.__name__}'

            print(f'{next_prefix}{factor.idx:>3} rvs={rv_idxs} function={function_ref}')

        print(f'{prefix}functions ({self.number_of_functions})')
        for function in sorted(self.non_zero_functions, key=lambda f: id(f)):
            print(f'{next_prefix}{id(function):>13}: {function.__class__.__name__}')
            function.dump(prefix=next_next_prefix, show_function_values=show_function_values, show_id_class=False)

        print(f'{prefix}end PGM id={id(self)}')

    def _has_cycle(self, factor: Factor, child_to_factor: Dict[int, Factor], states: NDArrayUInt8) -> bool:
        """
        Support function for `is_structure_bayesian`.

        A recursive depth-first-search to see if the factors form a DAG.

        For a factor `f` the value of states[f.idx] is the search state.
        Specifically:
        state 0 => the factor has not been seen yet,
        state 1 => the factor is seen but not fully processed,
        state 2 => the factor is fully processed.

        Args:
            factor: the current Factor being checked.
            child_to_factor: a dictionary from `RandomVariable.idx` to Factor
                with that random variable as the child.
            states: depth-first-search states, i.e., `states[i]` is the state of a factor with `Factor.idx == i`.
        Returns:
            True if a directed cycle is detected.
        """
        f_idx: int = factor.idx
        match states.item(f_idx):
            case 1:
                return True
            case 0:
                states[f_idx] = 1
                for parent in factor.rvs[1:]:
                    parent_factor = child_to_factor[parent.idx]
                    if self._has_cycle(parent_factor, child_to_factor, states):
                        return True
                states[f_idx] = 2
                return False
        return False

    def _register_rv(self, rv: RandomVariable) -> None:
        """
        Called by the constructor of RandomVariable to record a newly created Random variable
        of this PGM.

        Args:
            rv: the newly constructed random variable.
        """
        assert rv.pgm is self
        self._rvs += (rv,)
        self._shape += (len(rv),)
        self._indicators += rv.indicators

    def _condition_str_rv(
            self,
            cur_str: str,
            cur_rv: Set[Indicator],
            sep: str = ', ',
            equal: str = '=',
            elem: str = ' in ',
    ) -> str:
        """
        Support method for `self.condition_str`.

        This is a method renders a condition defined by a set of indicators, of the same random variable.

        Args:
              cur_str: the string to append to.
              cur_rv: a set of indicators, all from the same random variable.
              sep: the separator string to use between condition components.
              equal: the string to use for _rv_ = _state_.
              elem: the string to use for _rv_ in _set_.

        Returns:
            `cur_str` appended with the new condition, `cur_rv`.
        """
        if cur_str != '':
            cur_str += sep
        if len(cur_rv) == 1:
            cur_str += self.indicator_str(*cur_rv, sep=equal)
        else:
            _cur_rv = sorted(cur_rv)
            rv = self._rvs[_cur_rv[0].rv_idx]
            states_str: str = sep.join(_clean_str(rv.states[ind.state_idx]) for ind in _cur_rv)
            cur_str += f'{_clean_str(rv)}{elem}{{{states_str}}}'
        return cur_str


@dataclass(frozen=True, eq=True, slots=True)
class Indicator:
    """
    An indicator identifies a random variable being in a particular state.

    Indicators are immutable and hashable.

    Note that an Indicator does not know which PGM it came from, therefore indicators from one PGM
    are interchangeable with indicators of another PGM so long as corresponding random variables of the
    PGMs are co-indexed (created in the same order) and corresponding random variables have the same
    states.

    Fields:
        rv_idx: `rv.idx` where `rv` is the random variable referenced by this indicator.
        state_idx: the state index of the state referenced by this indicator.
    """
    rv_idx: int
    state_idx: int

    def __lt__(self, other) -> bool:
        """
        Define a sort order over indicators.
        When sorted, indicators are ordered by random variable index, then by state index.
        """
        if isinstance(other, Indicator):
            if self.rv_idx < other.rv_idx:
                return True
            if self.rv_idx > other.rv_idx:
                return False
            return self.state_idx < other.state_idx
        return False


class RandomVariable(Sequence[Indicator]):
    """
    A random variable in a probabilistic graphical model.

    Random variables are immutable and hashable.

    Each RandomVariable has a fixed finite number of states.
    Its states are indexed by integers, counting from zero.

    Every RandomVariable object belongs to exactly one PGM object.

    Every random variable has an index (counting from zero) which is its position
    in the random variable's PGM list of random variables.

    A random variable behaves like a sequence of Indicators, where each indicator represents a random
    variable being in a particular state. Specifically for a random variable rv, len(rv) is the
    number of states of the random variable and rv[i] is the Indicators representing that
    rv is in the ith state. When sliced, the result is a tuple, i.e. rv[1:3] = (rv[1], rv[2]).

    A RandomVariable has a name. This is for human convenience and has no functional purpose
    within a PGM.
    """

    def __init__(self, pgm: PGM, name: str, states: Union[int, Sequence[State]]):
        """
        Create a new random variable, in the given PGM.

        Assumes:
            Provided states contain no duplicates.

        Args:
            pgm: the PGM that the random variable will belong to.
            name: a name for the random variable.
            states: either an integer number of states or a sequence of state values. If a
                single integer, `n`, is provided then the states will be 0, 1, ..., n-1.
                If a sequence of states are provided then the states must be unique.
        """
        self._pgm: PGM = pgm
        self._name: str = name

        if isinstance(states, int):
            states = tuple(range(states))

        self._states: Sequence[State] = tuple(states)
        self._inv_states: Dict[State, int] = {state: idx for idx, state in enumerate(self._states)}

        if len(self._inv_states) != len(self._states):
            raise ValueError('random variable states are not unique')

        self._offset: int = pgm.number_of_indicators
        self._idx: int = pgm.number_of_rvs
        self._indicators: Sequence[Indicator] = tuple(Indicator(self._idx, i) for i in range(len(self._states)))

        # Register self with our PGM
        # noinspection PyProtectedMember
        pgm._register_rv(self)

    @property
    def pgm(self) -> PGM:
        """
        Returns:
            The PGM that this random variable belongs to.
        """
        return self._pgm

    @property
    def name(self) -> str:
        """
        Returns:
            The name of this random variable.
        """
        return self._name

    @property
    def idx(self) -> int:
        """
        Returns:
            The index of this random variable into the PGM.

        Ensures:
            `self.pgm.rvs[self.idx] is self`.
        """
        return self._idx

    @property
    def offset(self) -> int:
        """
        Returns:
            The index into the PGM's indicators for the start of this random variable's indicators.

        Ensures:
            `self.pgm.indicators[self.offset + i] is self[i] for i in range(len(self))`.
        """
        return self._offset

    @property
    def states(self) -> Sequence[State]:
        """
        Returns:
            the states of this random variable, in state index order.
        """
        return self._states

    @property
    def indicators(self) -> Sequence[Indicator]:
        """
        Returns:
            the indicators of this random variable, in state index order.
        """
        return self._indicators

    def state_range(self) -> Iterable[int]:
        """
        Iterate over the state indexes of this random variable, in order.

        Returns:
            range(len(self))
        """
        return range(len(self._states))

    def factors(self) -> Iterable[Factor]:
        """
        Iterate over factors that this random variable participates in.
        This method performs a search through all `self.pgm.factors`.

        Returns:
            an iterator over factors.
        """
        for factor in self._pgm.factors:
            if self in factor.rvs:
                yield factor

    def markov_blanket(self) -> Set[RandomVariable]:
        """
        Return the set of random variable that are connected
        to this random variable by a factor.
        This method performs a search through all `self.pgm.factors`.

        Returns:
            a set of random variables connected to this random variable by any factor, excluding self.
        """
        result = set()
        for factor in self.factors():
            result.update(factor.rvs)
        result.discard(self)
        return result

    def state_idx(self, state: State) -> int:
        """
        Returns:
            the state index of the given state of this random variable.

        Assumes:
            the given state is a state of this random variable.
        """
        return self._inv_states[state]

    def is_default_states(self) -> bool:
        """
        Are the states of this random variable the default states.
        I.e., `self.states[i] == i, for all 0 <= i < len(self)`.

        Returns:
            True only if the states are the same as the state indexes.
        """
        return all(i == s for i, s in enumerate(self._states))

    def __str__(self) -> str:
        """
        Returns:
             the name of this random variable.
        """
        return self._name

    def __call__(self, state: State) -> Indicator:
        """
        Get the indicator for the given state.
        This is equivalent to self[self.state_idx(state)].

        Returns:
            an indicator of this random variable.

        Assumes:
            the given state is a state of this random variable.
        """
        return self._indicators[self._inv_states[state]]

    def __hash__(self) -> int:
        """
        A random variable is hashable.
        """
        return self._idx

    def __eq__(self, other) -> bool:
        """
        Two random variable are equal if they are the same object.
        """
        return self is other

    def equivalent(self, other: RandomVariable | Sequence[Indicator]) -> bool:
        """
        Two random variable are equivalent if their indicators are equal. Only
        random variable indexes and state indexes are checked.

        This ignores the names of the random variable and the names of their states.
        This means their indicators will work correctly in slot maps, even
        if from different PGMs.

        Args:
            other: either a random variable or a sequence of Indicators.

        Returns:
            True only if they represent the same sequence of indicators.
        """
        indicators = self._indicators
        if isinstance(other, RandomVariable):
            return self.idx == other.idx and len(self) == len(other)
        else:
            return (
                    len(indicators) == len(other) and
                    all(indicators[i] == other[i] for i in range(len(indicators)))
            )

    def __len__(self) -> int:
        """
        Returns:
            Number of states (or equivalently, the number of indicators) of this random variable.
        """
        return len(self._states)

    def __iter__(self) -> Iterator[Indicator]:
        """
        Iterate over the indicators of this random variable.
        """
        return iter(self._indicators)

    @overload
    def __getitem__(self, index: int) -> Indicator:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Indicator]:
        ...

    def __getitem__(self, index):
        """
        Get the indexed (or sliced) indicators.
        """
        return self._indicators[index]

    def index(self, value: Any, start: int = 0, stop: int = -1) -> int:
        """
        Returns the first index of `value`.
        Raises ValueError if the value is not present.
        Contracted by Sequence[Indicator].

        Warning:
            This method is different to `self.idx`.
        """
        if isinstance(value, Indicator):
            if value.rv_idx == self._idx:
                idx: int = value.state_idx
                if stop < 0:
                    stop = len(self) + stop + 1
                if 0 <= idx < len(self) and start <= idx < stop:
                    return value.state_idx
        raise ValueError(f'{value!r} is not an indicator of the random variable')

    def count(self, value: Any) -> int:
        """
        Returns the number of occurrences of `value`.
        Contracted by Sequence[Indicator].
        """
        if isinstance(value, Indicator):
            if value.rv_idx == self._idx and 0 <= value.state_idx < len(self):
                return 1
        return 0


class RVMap(Sequence[RandomVariable]):
    """
    Wrap a PGM to provide convenient access to PGM random variables.

    An RVMap of a PGM behaves exactly like the PGM `rvs` property. That it, it
    behaves like a sequence of RandomVariable objects.

    If the underlying PGM is updated, then the RVMap will automatically update.

    Additionally, an RVMap enables access to the PGM random variable via the name
    of each random variable.

    for example, if `pgm.rvs[1]` is a random variable named `xray`, then
    ```
    rvs = RVMap(pgm)

    # These all retrieve the same random variable object.
    xray = rvs[1]
    xray = rvs('xray')
    xray = rvs.xray
    ```

    To use an RVMap on a PGM, the variable names must be unique across the PGM.
    """

    def __init__(self, pgm: PGM, ignore_case: bool = False):
        """
        Construct an RVMap for the given PGM.

        Args:
            pgm: the PGM to wrap.
            ignore_case: if true, the variable name are not case-sensitive.
        """
        self._pgm: PGM = pgm
        self._ignore_case: bool = ignore_case
        self.__rv_map: Dict[str, RandomVariable] = {}
        self._reserved_names: Set[str] = {self._clean_name(name) for name in dir(self)}

        # Force the rv map cache to be updated.
        # This may raise an exception.
        _ = self._rv_map

    def _clean_name(self, name: str) -> str:
        """
        Adjust the case of the given name as needed.
        """
        return name.lower() if self._ignore_case else name

    @property
    def _rv_map(self) -> Dict[str, RandomVariable]:
        """
        Get the cached rv map, updating as needed if the PGM changed.
        Returns:
            a mapping from random variable name to random variable
        """
        if len(self.__rv_map) != len(self._pgm.rvs):
            # There is a difference between the map and the PGM - create a new map.
            self.__rv_map = {self._clean_name(rv.name): rv for rv in self._pgm.rvs}
            if len(self.__rv_map) != len(self._pgm.rvs):
                raise RuntimeError(f'random variable names are not unique')
            if not self._reserved_names.isdisjoint(self.__rv_map.keys()):
                raise RuntimeError(f'random variable names clash with reserved names.')
        return self.__rv_map

    def new_rv(self, name: str, states: Union[int, Sequence[State]]) -> RandomVariable:
        """
        As per `PGM.new_rv`.
        Delegate creating a new random variable to the PGM.

        Returns:
            a RandomVariable object belonging to the PGM.
        """
        return self._pgm.new_rv(name, states)

    def __len__(self) -> int:
        return len(self._pgm.rvs)

    def __getitem__(self, index: int) -> RandomVariable:
        return self._pgm.rvs[index]

    def items(self) -> Iterable[Tuple[str, RandomVariable]]:
        return self._rv_map.items()

    def keys(self) -> Iterable[str]:
        return self._rv_map.keys()

    def values(self) -> Iterable[RandomVariable]:
        return self._rv_map.values()

    def get(self, rv_name: str, default=None):
        return self._rv_map.get(self._clean_name(rv_name), default)

    def __call__(self, rv_name: str) -> RandomVariable:
        return self._rv_map[self._clean_name(rv_name)]

    def __getattr__(self, rv_name: str) -> RandomVariable:
        return self(rv_name)


class Factor:
    """
    A PGM factor over one or more random variables declares a relationship between
    those variables. A Factor also has a potential function associated with
    it which defines a real-number value with each combination of states of
    the random variables.

    The default potential function for a factor is a unique ZeroPotentialFunction.

    The order of a Factors random variables is important as many things will be
    co-indexed with the random variables. For example, the shape of a Factor is
    the tuple of random variable lengths.

    Note that multiple factors may share a potential function, so long as they all
    belong to the same PGM object and have the same shape.
    """

    def __init__(self, pgm: PGM, *rvs: RandomVariable):
        """
        Add a new factor to the given PGM.

        Assumes:
            The given random variables all belong to this PGM.
            The random variables contain no duplicates.

        Args:
            pgm: the PGM that the factor will belong to.
            *rvs: the random variables.

        Returns:
            a Factor object belonging to this PGM.
        """
        if len(set(rvs)) != len(rvs):
            raise ValueError('duplicated random variable in factor')
        if len(rvs) == 0:
            raise ValueError('must be at least one random variable')
        if any(rv.pgm is not pgm for rv in rvs):
            raise ValueError('random variable not from the same PGM')

        self._pgm: PGM = pgm
        self._idx: int = pgm.number_of_factors
        self._rvs: Sequence[RandomVariable] = tuple(rvs)
        self._shape: Shape = tuple(len(rv) for rv in rvs)

        self._zero_potential_function: ZeroPotentialFunction = ZeroPotentialFunction(self)
        self._potential_function: PotentialFunction = self._zero_potential_function

        # Register self with our PGM
        # noinspection PyProtectedMember
        pgm._factors += (self,)

    @property
    def rvs(self) -> Sequence[RandomVariable]:
        """
        Returns:
            The random variables of this factor.
        """
        return self._rvs

    @property
    def pgm(self) -> PGM:
        """
        Returns:
            The PGM that this factor belongs to.
        """
        return self._pgm

    @property
    def idx(self) -> int:
        """
        Returns:
            The index of this factor into the PGM.

        Ensures:
            `self.pgm.factors[self.idx] is self`.
        """
        return self._idx

    @property
    def shape(self) -> Shape:
        return self._shape

    @property
    def number_of_states(self) -> int:
        """
        How many distinct states are covered by this Factor.
        """
        return self._potential_function.number_of_states

    def __str__(self) -> str:
        """
        Return a human-readable string to represent this factor.
        This is intended mainly for debugging purposes.
        """
        return '(' + ', '.join([repr(str(rv)) for rv in self._rvs]) + ')'

    def __len__(self) -> int:
        """
        Returns:
            the number of random variables.
        """
        return len(self._rvs)

    @overload
    def __getitem__(self, index: int) -> RandomVariable:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[RandomVariable]:
        ...

    def __getitem__(self, index):
        return self._rvs[index]

    def instances(self, flip: bool = False) -> Iterable[Instance]:
        """
        Iterate over all possible instances, in natural index order (i.e.,
        last random variable changing most quickly).

        Args:
            flip: if true, then first random variable changes most quickly

        Returns:
            an iterator over tuples, each tuple holds random variable
            state indexes, co-indexed with this object's shape, i.e., self.shape.
        """
        return self.function.instances(flip)

    def parent_instances(self, flip: bool = False) -> Iterable[Instance]:
        """
        Iterate over all possible instances of parent random variable, in
        natural index order (i.e., last random variable changing most quickly).

        Args:
            flip: if true, then first random variable changes most quickly

        Returns:
            an iteration over tuples, each tuple holds random variable states
            co-indexed with this object's 'parent' shape, i.e., `self.shape[1:]`.
        """
        return self.function.parent_instances(flip)

    @property
    def is_zero(self) -> bool:
        """
        Is the potential function of this factor set to the special 'zero' potential function.
        """
        return self._potential_function is self._zero_potential_function

    @property
    def function(self) -> PotentialFunction:
        return self._potential_function

    @function.setter
    def function(self, function: PotentialFunction | Factor) -> None:
        """
        Set the potential function for this PGM factor to the given potential function
        or factor.

        Assumes:
            The given potential function belongs to the same PGM as this Factor.
            The potential function has the correct shape.
        """
        if isinstance(function, Factor):
            function = function.function
        assert isinstance(function, PotentialFunction)

        if self._potential_function is function:
            # nothing to do
            return

        if function.pgm is not self._pgm:
            raise ValueError(f'the given function is not of the same PGM as the factor')

        if function.shape != self._shape:
            raise ValueError(f'incorrect function shape: expected {self._shape}, got {function.shape}')

        if isinstance(function, ZeroPotentialFunction):
            self.set_zero()
        else:
            self._potential_function = function

    def set_zero(self) -> ZeroPotentialFunction:
        """
        Set the factor's potential function to its original ZeroPotentialFunction.

        Returns:
            the potential function.
        """
        self._potential_function = self._zero_potential_function
        return self._potential_function

    def set_dense(self) -> DensePotentialFunction:
        """
        Set to the potential function to a new `DensePotentialFunction` object.

        Returns:
            the potential function.
        """
        self._potential_function = DensePotentialFunction(self)
        return self._potential_function

    def set_sparse(self) -> SparsePotentialFunction:
        """
        Set to the potential function to a new `SparsePotentialFunction` object.

        Returns:
            the potential function.
        """
        self._potential_function = SparsePotentialFunction(self)
        return self._potential_function

    def set_compact(self) -> CompactPotentialFunction:
        """
        Set to the potential function to a new `CompactPotentialFunction` object.

        Returns:
            the potential function.
        """
        self._potential_function = CompactPotentialFunction(self)
        return self._potential_function

    def set_clause(self, *key: int) -> ClausePotentialFunction:
        """
        Set to the potential function to a new `ClausePotentialFunction` object.

        Args:
            *key: defines the random variable states of the clause. The key is a sequence of
                random variable state indexes, co-indexed with `Factor.rvs`.

        Returns:
            the potential function.

        Raises:
             KeyError: if the key is not valid for the shape of the factor.
        """
        self._potential_function = ClausePotentialFunction(self, key)
        return self._potential_function

    def set_cpt(self, tolerance: float = DEFAULT_TOLERANCE) -> CPTPotentialFunction:
        """
        Set to the potential function to a new `CPTPotentialFunction` object.

        Args:
            tolerance: a tolerance when testing if values are equal to zero or one.

        Returns:
            the potential function.

        Raises:
            ValueError: if tolerance is negative.
        """
        self._potential_function = CPTPotentialFunction(self, tolerance)
        return self._potential_function


@dataclass(frozen=True, eq=True)
class ParamId:
    """
    A ParamId identifies a parameter of a potential function.

    Parameter identifiers uniquely identify every parameter within a PGM.

    A ParamId is immutable and hashable.
    """
    function_id: int
    param_idx: int


class PotentialFunction(ABC):
    """
    A potential function defines the potential values for a Factor, where
    a factor joins one or more variables of a PGM.

    A potential function may be shared by several Factors of a PGM,
    i.e., can be applied to multiple variables.

    The `shape` of a potential function is a tuple of integers which defines
    the number of variables, len(shape), and the number of states of each
    variable, shape[i].

    The potential function value for variable states (x = i, y = j, ...) is given by
    self[i, j, ...], i.e., self.__getitem__((i, j, ...)). The tuple, (i, j, ...), is
    known as a Key.

    The values of a potential function are defined by potential function parameters.
    The number of potential function parameters is given by number_of_parameters.
    The value of each parameter is given by get_param(i), where i is the parameter index.

    Every valid key of the potential function is mapped either mapped to a parameter or is
    "guaranteed zero" which means that the value is zero and cannot be changed by changing
    the values of the potential function's parameters.
    """

    def __init__(self, factor: Factor):
        """
        Create a potential function compatible with the given factor.

        Ensures:
            Does not hold a reference to the given factor.
            Does not register the potential function with the PGM.

        Args:
            factor: which factor is this potential function is compatible with.
        """
        self._pgm: PGM = factor.pgm
        self._shape: Shape = factor.shape
        self._number_of_states = _multiply(self._shape)

    @property
    def pgm(self) -> PGM:
        """
        Returns:
            The PGM that this potential function belong to.
        """
        return self._pgm

    @property
    def shape(self) -> Shape:
        """
        Returns:
            The shape of this potential function.
        """
        return self._shape

    @property
    def number_of_rvs(self) -> int:
        """
        Returns:
            The number of random variables in this potential function.
        """
        return len(self._shape)

    @property
    def number_of_states(self) -> int:
        """
        How many distinct states are covered by this potential function.

        Returns:
            The size of the state space of this potential function.
        """
        return self._number_of_states

    @property
    def number_of_parent_states(self) -> int:
        """
        How many distinct states are covered by this potential function parents,
        i.e., excluding the first random variable.

        Returns:
            The size of the state space of this potential function parent random variables.
        """
        return _multiply(self._shape[1:])

    def count_usage(self) -> int:
        """
        Check all PGM factors to count the number of times that this potential function
        is used.

        Returns:
            the number of factors that use this potential function.
        """
        return sum(1 for factor in self._pgm.factors if factor.function is self)

    def check_key(self, key: Key) -> Instance:
        """
        Convert the key into an instance.

        Arg:
            key: defines an instance in the state space of the potential function.

        Returns:
            an instance, which is a tuple of state indexes, co-indexed with `self.rvs`.

        Raises:
             KeyError: if the key is not valid for the shape of the factor.
        """
        return check_key(self._shape, key)

    def valid_key(self, key: Key) -> bool:
        """
        Is the given key valid.

        Arg:
            key: defines an instance in the state space of the potential function.

        Returns:
            True only if the given key is valid.
        """
        return valid_key(self._shape, key)

    def valid_parameter(self, param_idx: int) -> bool:
        """
        Is the given parameter index valid.

        Arg:
            param_idx: a parameter index.

        Returns:
            True only if `0 <= param_idx < self.number_of_parameters`.
        """
        return 0 <= param_idx < self.number_of_parameters

    @property
    def is_sparse(self) -> bool:
        """
        Are there any 'guaranteed zero' parameters values.

        Returns:
            True only if `self.number_of_not_guaranteed_zero < self._number_of_states`.
        """
        return self.number_of_not_guaranteed_zero < self._number_of_states

    @property
    @abstractmethod
    def number_of_not_guaranteed_zero(self) -> int:
        """
        How many of the states of this potential function are not 'guaranteed zero'.
        That is, how many keys are associated with a parameter.

        Returns:
            The number of valid keys that are associated with a parameter.

        Ensures:
            0 <= self.number_of_not_guaranteed_zero <= self.number_of_states.
        """
        ...

    @property
    @abstractmethod
    def number_of_parameters(self) -> int:
        """
        Get the number of parameters defining the potential function values.
        Each valid key of the function maps either to a parameter
        is 'guaranteed zero'.

        Returns:
            The number of parameters.

        Ensures:
            0 <= self.number_of_parameters <= self.number_of_not_guaranteed_zero.
        """
        ...

    @property
    @abstractmethod
    def params(self) -> Iterable[Tuple[int, float]]:
        """
        Iterate the parameters and their associated values.

        Returns:
            An iterable over (param_idx, value) tuples, for every possible parameter.

        Assumes:
            The potential function is not mutated while iterating.
        """
        ...

    @property
    @abstractmethod
    def keys_with_param(self) -> Iterable[Tuple[Instance, int, float]]:
        """
        Iterate the keys that have a parameter associated with them.

        Returns:
            An iterable over (key, param_idx, value) tuples, for every key with an associated parameter.

        Assumes:
            The potential function is not mutated while iterating.
        """
        ...

    @abstractmethod
    def __getitem__(self, key: Key) -> float:
        """
        Get the potential function value for the given instance key.

        Arg:
            key: defines an instance in the state space of the potential function.

        Returns:
            The value of the potential function for the given key.

        Assumes:
            self.valid_key(key).
        """
        ...

    @abstractmethod
    def param_value(self, param_idx: int) -> float:
        """
        Get the potential function value by parameter index.

        Arg:
            param_idx: a parameter index.

        Assumes:
            `self.valid_parameter(param_idx)`.
        """
        ...

    @abstractmethod
    def param_idx(self, key: Key) -> Optional[int]:
        """
        Get the parameter index for the given potential function random variables states (key).

        Arg:
            key: defines an instance in the state space of the potential function.

        Returns:
            either `None` indicating a "guaranteed zero" value, or the parameter index holding
            the potential function value for the key.
        """
        ...

    def is_cpt(self, tolerance=DEFAULT_TOLERANCE) -> bool:
        """
        Is the potential function set with parameters values conforming to a
        Conditional Probability Table.

        Every parameter value must be non-negative.
        For every state of the parent (non-first slots)
        the sum of the parameters over the child states (first slots)
        must be either 1 or 0.

        Assumes:
            tolerance is non-negative.

        Args:
            tolerance: a tolerance when testing if values are equal to zero or one.

        Returns:
            True only if the potential function is compatible with being a CPT.
        """
        # This default implementation calculates the result the long way, by checking
        # every valid key of the potential function.
        # Subclasses may override this implementation.
        low: float = 1.0 - tolerance
        high: float = 1.0 + tolerance
        for parent_state in self.parent_instances():
            total: float = sum(
                self[(state,) + tuple(parent_state)]
                for state in range(self.shape[0])
            )
            if not ((low <= total <= high) or (0 <= total <= tolerance)):
                return False
        return True

    def natural_param_idx(self, key: Key) -> int:
        """
        Get the natural parameter index for the given key. This is the same index as used
        by a DensePotentialFunction with the same shape.

        Args:
            key: is a valid key of the potential function, referring to an instance in the factor's state space.

        Assumes:
            `self.valid_key(key)` is true.

        Returns:
            a hypothetical parameter index assuming that every valid key has a unique parameter
            as per DensePotentialFunction.
        """
        return _natural_key_idx(self._shape, key)

    def param_id(self, param_idx: int) -> ParamId:
        """
        Get a hashable object to represent the parameter with the given parameter index.

        Arg:
            param_idx: a parameter index.

        Returns:
             a hashable ParamId object for the parameter of this potential function.

        Raises:
            ValueError: if the parameter index is not valid.
        """
        if not (0 <= param_idx < self.number_of_parameters):
            raise ValueError(f'invalid parameter index: {param_idx}')
        return ParamId(id(self), param_idx)

    def items(self) -> Iterable[Tuple[Instance, float]]:
        """
        Iterate over all keys and values of this potential function.

        Returns:
            An iterator over all (key, value) pairs, where key is an Instance and value
            is the value of the potential function for the key.
        """
        for key in _combos_ranges(self._shape, flip=True):
            yield key, self[key]

    def instances(self, flip: bool = False) -> Iterable[Instance]:
        """
        Iterate over all possible instances, in natural index order (i.e.,
        last random variable changing most quickly).

        Args:
            flip: if true, then first random variable changes most quickly

        Returns:
            an iterator over tuples, each tuple holds random variable
            state indexes, co-indexed with this object's shape, i.e., self.shape.
        """
        return _combos_ranges(self._shape, flip=not flip)

    def parent_instances(self, flip: bool = False) -> Iterable[Instance]:
        """
        Iterate over all possible instances of parent random variable, in
        natural index order (i.e., last random variable changing most quickly).

        Args:
            flip: if true, then first random variable changes most quickly

        Returns:
            an iteration over tuples, each tuple holds random variable states
            co-indexed with this object's 'parent' shape, i.e., `self.shape[1:]`.
        """
        return _combos_ranges(self._shape[1:], flip=not flip)

    def __str__(self) -> str:
        """
        Provide a human-readable representation of this potential function.
        This is intended mainly for debugging purposes.
        """
        shape_str: str = ', '.join(str(x) for x in self._shape)
        return f'{self.__class__.__name__}({shape_str})'

    def dump(
            self,
            *,
            prefix: str = '',
            indent: str = '    ',
            show_function_values: bool = False,
            show_id_class: bool = True,
    ) -> None:
        """
        Print a dump of the function.
        This is intended for debugging purposes.

        Args:
            prefix: optional prefix for indenting all lines.
            indent: additional prefix to use for extra indentation.
            show_function_values: if true, then the function values will be dumped.
            show_id_class: if true, then the function id and class will be dumped.
        """

        shape_str: str = ', '.join(str(x) for x in self._shape)

        if show_id_class:
            print(f'{prefix}id: {id(self)}')
            print(f'{prefix}class: {self.__class__.__name__}')
        print(f'{prefix}usage: {self.count_usage()}')
        print(f'{prefix}rvs: {self.number_of_rvs}')
        print(f'{prefix}shape: ({shape_str})')
        print(f'{prefix}states: {self._number_of_states}')
        print(f'{prefix}guaranteed zero: {self._number_of_states - self.number_of_not_guaranteed_zero}')
        print(f'{prefix}not guaranteed zero: {self.number_of_not_guaranteed_zero}')
        print(f'{prefix}parameters: {self.number_of_parameters}')
        if show_function_values:
            next_prefix = prefix + indent
            for key, param_idx, value in self.keys_with_param:
                print(f'{next_prefix}{param_idx} {key} = {value}')


class ZeroPotentialFunction(PotentialFunction):
    """
    A ZeroPotentialFunction behaves like a DensePotentialFunction
    in that there is a parameter for each possible key.
    However, a PGM user has no way to change parameter values.
    Parameter values are always zero.
    Despite the inability to change the value of the parameters,
    no key is considered 'guaranteed zero'.

    The primary use of a ZeroPotentialFunction is as a placeholder
    within a factor, prior to parameter learning.
    """
    __slots__ = ()

    def __init__(self, factor: Factor):
        """
        Create a potential function for the given factor.

        Ensures:
            Does not hold a reference to the given factor.
            Does not register the potential function with the PGM.

        Args:
            factor: which factor is this potential function is compatible with.
        """
        super().__init__(factor)

    @property
    def number_of_not_guaranteed_zero(self) -> int:
        return self.number_of_states

    @property
    def number_of_parameters(self) -> int:
        return self.number_of_states

    @property
    def params(self) -> Iterable[Tuple[int, float]]:
        for param_idx in range(self.number_of_parameters):
            yield param_idx, 0

    @property
    def keys_with_param(self) -> Iterable[Tuple[Instance, int, float]]:
        for param_idx, instance in enumerate(self.instances()):
            yield instance, param_idx, 0

    def __getitem__(self, key: Key) -> float:
        self.check_key(key)
        return 0

    def param_value(self, param_idx: int) -> float:
        if not self.valid_parameter(param_idx):
            raise ValueError(f'invalid parameter index: {param_idx}')
        return 0

    def param_idx(self, key: Key) -> int:
        return _natural_key_idx(self._shape, key)

    def is_cpt(self, tolerance=DEFAULT_TOLERANCE) -> bool:
        return True


class DensePotentialFunction(PotentialFunction):
    """
    A dense (tabular) potential function.
    There is one parameter for each valid key of the potential function.
    The initial value for each parameter is zero.
    It is possible independently change any value corresponding to any key.
    """

    def __init__(self, factor: Factor):
        """
        Create a potential function for the given factor.

        Ensures:
            Does not hold a reference to the given factor.
            Does not register the potential function with the PGM.

        Args:
            factor: which factor is this potential function is compatible with.
        """
        super().__init__(factor)
        self._values: NDArrayFloat64 = np.zeros(self.number_of_states, dtype=np.float64)

    @property
    def number_of_not_guaranteed_zero(self) -> int:
        return self.number_of_states

    @property
    def number_of_parameters(self) -> int:
        return self.number_of_states

    def __getitem__(self, key: Key) -> float:
        return self._values.item(self.param_idx(key))

    def param_value(self, param_idx: int) -> float:
        return self._values.item(param_idx)

    def param_idx(self, key: Key) -> Optional[int]:
        return self.natural_param_idx(key)

    @property
    def params(self) -> Iterable[Tuple[int, float]]:
        # Type warning due to numpy type erasure
        # noinspection PyTypeChecker
        return enumerate(self._values)

    @property
    def keys_with_param(self) -> Iterable[Tuple[Instance, int, float]]:
        for param_idx, key in enumerate(self.instances()):
            value: float = self.param_value(param_idx)
            yield key, param_idx, value

    # Mutators

    def __setitem__(self, key: Key, value: float) -> None:
        """
        Set the potential function value, for a given key.

        Arg:
            key: defines an instance in the state space of the potential function.
            value: the new value of the potential function for the given key.

        Assumes:
            self.valid_key(key).
        """
        self._values[self.param_idx(key)] = value

    def set_param_value(self, param_idx: int, value: float) -> None:
        """
        Set the parameter value.

        Arg:
            param_idx: is the index of the parameter.
            value: the new value of the potential function for the given key.

        Assumes:
            self.valid_param(param_idx).
        """
        self._values[param_idx] = value

    def clear(self) -> DensePotentialFunction:
        """
        Set all values of the potential function to zero.
        
        Returns:
            self
        """
        return self.set_all(0)

    def normalise_cpt(self) -> DensePotentialFunction:
        """
        Normalise the parameter values as if this was a CPT.
        That is, treat the first random variable as the child and the others as parents;
        for each combination of parent states, ensure the parameters over the child
        states sum to 1 (or 0).
        
        Assumes:
            There are no negative parameter values.
            
        Returns:
            self
        """
        child = self._shape[0]
        parents = self._shape[1:]
        for parent_states in _combos_ranges(parents):
            keys = [(c,) + parent_states for c in range(child)]
            total = sum(self[key] for key in keys)
            if total != 0 and total != 1:
                for key in keys:
                    self[key] /= total
        return self

    def normalise(self, grouping_positions: Sequence[int] = ()) -> DensePotentialFunction:
        """
        Convert the potential function to a CPT with 'grouping_positions' nominating
        the parent random variables.
        
        I.e., for each possible key of the function with the same value at each
        grouping position, the sum of values for matching keys in the factor is scaled
        to be 1 (or 0).

        Parameter 'grouping_positions' are indices into `self.shape`. For example, the
        grouping positions of a factor with parent rvs 'conditioning_rvs', then
        grouping_positions = [i for i, rv in enumerate(factor.rvs) if rv in conditioning_rvs].

        Args:
            grouping_positions: indices into `self.shape`.
            
        Returns:
            self
        """
        _normalise_potential_function(self, grouping_positions)
        return self

    def set_iter(self, values: Iterable[float]) -> DensePotentialFunction:
        """
        Set the values of the potential function using the given iterator.

        Mapping instances to *values is as follows:
            Given Factor(rv1, rv2) where rv1 has 2 states, and rv2 has 3 states:
            values[0] represents instance (0,0)
            values[1] represents instance (0,1)
            values[2] represents instance (0,2)
            values[3] represents instance (1,0)
            values[4] represents instance (1,1)
            values[5] represents instance (1,2).

        For example: to set to counts, starting from 1, use `self.set_iter(itertools.count(1))`.

        Args:
            values: an iterable providing values to use.
            
        Returns:
            self
        """
        self._values = np.fromiter(
            values,
            dtype=np.float64,
            count=self.number_of_parameters
        )
        return self

    def set_stream(self, stream: Callable[[], float]) -> DensePotentialFunction:
        """
        Set the values of the potential function by repeatedly calling the stream function.
        The order of values is the same as set_iter.

        For example, to set to random numbers, use `self.set_stream(random.random)`.

        Args:
            stream: a callable taking no arguments, returning the values to use.
            
        Returns:
            self
        """
        return self.set_iter(iter(stream, None))

    def set_flat(self, *value: float) -> DensePotentialFunction:
        """
        Set the values of the potential function to the given values.
        The order of values is the same as set_iter.
        
        Args:
            *value: the values to use.
            
        Returns:
            self
            
        Raises:
            ValueError: if `len(value) != self.number_of_states`.
        """
        if len(value) != self.number_of_states:
            raise ValueError(f'wrong number of values: expected {self.number_of_states}, got {len(value)}')
        return self.set_iter(value)

    def set_all(self, value: float) -> DensePotentialFunction:
        """
        Set all values of the potential function to the given value.

        Args:
            value: the value to use.
            
        Returns:
            self
        """
        return self.set_iter(_repeat(value))

    def set_uniform(self) -> DensePotentialFunction:
        """
        Set all values of the potential function 1/number_of_states.
        
        Returns:
            self
        """
        return self.set_all(1.0 / self.number_of_states)


class SparsePotentialFunction(PotentialFunction):
    """
    A sparse potential function.

    There is one parameter for each non-zero key value.
    The user may set the value for any key and parameters will
    be automatically reconfigured as needed. Setting the value for
    a key to zero disassociates the key from its parameter and
    thus makes that key "guaranteed zero".
    """

    def __init__(self, factor: Factor):
        """
        Create a potential function for the given factor.

        Ensures:
            Does not hold a reference to the given factor.
            Does not register the potential function with the PGM.

        Args:
            factor: which factor is this potential function is compatible with.
        """
        super().__init__(factor)
        self._values: List[float] = []
        self._params: Dict[Instance, int] = {}

    @property
    def number_of_not_guaranteed_zero(self) -> int:
        return len(self._params)

    @property
    def number_of_parameters(self) -> int:
        return len(self._params)

    def __getitem__(self, key: Key) -> float:
        param_idx: Optional[int] = self.param_idx(key)
        if param_idx is None:
            return 0
        else:
            return self._values[param_idx]

    def param_value(self, param_idx: int) -> float:
        return self._values[param_idx]

    def param_idx(self, key: Key) -> Optional[int]:
        return self._params.get(_key_to_instance(key))

    @property
    def params(self) -> Iterable[Tuple[int, float]]:
        return enumerate(self._values)

    @property
    def keys_with_param(self) -> Iterable[Tuple[Instance, int, float]]:
        for key, param_idx in self._params.items():
            value: float = self._values[param_idx]
            yield key, param_idx, value

    # Mutators

    def __setitem__(self, key: Key, value: float) -> None:
        """
        Set the potential function value, for a given key.
        
        If value is zero, then the key will become "guaranteed zero".

        Arg:
            key: defines an instance in the state space of the potential function.
            value: the new value of the potential function for the given key.

        Assumes:
            self.valid_key(key).
        """
        instance: Instance = _key_to_instance(key)
        param_idx: Optional[int] = self._params.get(instance)

        if param_idx is None:
            if value == 0:
                # Nothing to do
                return
            param_idx = len(self._values)
            self._values.append(value)
            self._params[instance] = param_idx
            return

        if value != 0:
            # Simple case
            self._values[param_idx] = value
            return

        # This is the case where the key was associated with a parameter
        # but the value is being set to zero, so we
        # need to clear an existing non-zero parameter.
        # This code operates by first ensuring the parameter is the last one,
        # then popping the last parameter.

        end: int = len(self._values) - 1
        if param_idx != end:
            # need to swap the parameter with the end.
            self._values[param_idx] = self._values[end]

            for test_instance, test_param_idx in self._params.items():
                if test_param_idx == end:
                    self._params[test_instance] = param_idx
                    # There will only be one, so we can break now
                    break

        # Remove the parameter
        self._values.pop()
        self._params.pop(instance)

    def set_param_value(self, param_idx: int, value: float) -> None:
        """
        Set the parameter value.

        Arg:
            param_idx: is the index of the parameter.
            value: the new value of the potential function for the given key.

        Assumes:
            self.valid_param(param_idx).
        """
        self._values[param_idx] = value

    def clear(self) -> SparsePotentialFunction:
        """
        Set all values of the potential function to zero.
        
        Returns:
            self
        """
        self._values = []
        self._params = {}
        return self

    def normalise_cpt(self) -> SparsePotentialFunction:
        """
        Normalise the parameter values as if this was a CPT.
        That is, treat the first random variable as the child and the others as parents;
        for each combination of parent states, ensure the parameters over
        the child states sum to 1 (or 0).

        Returns:
            self
        """
        grouping_positions = list(range(1, self.number_of_rvs))
        _normalise_potential_function(self, grouping_positions)
        return self

    def normalise(self, grouping_positions=()) -> SparsePotentialFunction:
        """
        Convert the potential function to a CPT with 'grouping_positions' nominating
        the parent random variables.
        
        I.e., for each possible key of the function with the same value at each
        grouping position, the sum of values for matching keys in the factor is scaled
        to be 1 (or 0).

        Parameter 'grouping_positions' are indices into function.shape. For example, the
        grouping positions of a factor with parent rvs 'conditioning_rvs', then
        grouping_positions = [i for i, rv in enumerate(factor.rvs) if rv in conditioning_rvs].

        Returns:
            self
        """
        _normalise_potential_function(self, grouping_positions)
        return self

    def set_iter(self, values: Iterable[float]) -> SparsePotentialFunction:
        """
        Set the values of the potential function using the given iterator.

        Mapping instances to *values is as follows:
            Given Factor(rv1, rv2) where rv1 has 2 states, and rv2 has 3 states:
            values[0] represents instance (0,0)
            values[1] represents instance (0,1)
            values[2] represents instance (0,2)
            values[3] represents instance (1,0)
            values[4] represents instance (1,1)
            values[5] represents instance (1,2).

        For example: to set to counts, starting from 1, use `self.set_iter(itertools.count(1))`.

        Args:
            values: an iterable providing values to use.

        Returns:
            self
        """
        self.clear()
        for instance, value in zip(self.instances(), values):
            if value != 0:
                self._params[instance] = len(self._values)
                self._values.append(value)
        return self

    def set_stream(self, stream: Callable[[], float]) -> SparsePotentialFunction:
        """
        Set the values of the potential function by repeatedly calling the stream function.
        The order of values is the same as set_iter.

        For example, to set to random numbers, use `self.set_stream(random.random)`.

        Args:
            stream: a callable taking no arguments, returning the values to use.

        Returns:
            self
        """
        return self.set_iter(iter(stream, None))

    def set_flat(self, *value: float) -> SparsePotentialFunction:
        """
        Set the values of the potential function to the given values.
        The order of values is the same as set_iter.

        Args:
            *value: the values to use.

        Returns:
            self

        Raises:
            ValueError: if `len(value) != self.number_of_states`.
        """
        if len(value) != self.number_of_states:
            raise ValueError(f'wrong number of values: expected {self.number_of_states}, got {len(value)}')
        return self.set_iter(value)

    def set_all(self, value: float) -> SparsePotentialFunction:
        """
        Set all values of the potential function to the given value.

        Args:
            value: the value to use.

        Returns:
            self
        """
        if value == 0:
            return self.clear()
        else:
            return self.set_iter(_repeat(value))

    def set_uniform(self) -> SparsePotentialFunction:
        """
        Set all values of the potential function 1/number_of_states.
        
        Returns:
            self
        """
        return self.set_all(1.0 / self.number_of_states)


class CompactPotentialFunction(PotentialFunction):
    """
    A compact potential function is sparse, where values for keys of
    the same value are represented by a single parameter.

    There is one parameter for each unique, non-zero key value.
    The user may set the value for any key and parameters will
    be automatically reconfigured as needed. Setting the value for
    a key to zero disassociates the key from its parameter and
    thus makes that key "guaranteed zero".
    """

    def __init__(self, factor: Factor):
        """
        Create a potential function for the given factor.

        Ensures:
            Does not hold a reference to the given factor.
            Does not register the potential function with the PGM.

        Args:
            factor: which factor is this potential function is compatible with.
        """
        super().__init__(factor)
        self._values: List[float] = []
        self._counts: List[int] = []
        self._map: Dict[Instance, int] = {}
        self._inv_map: Dict[float, int] = {}

    @property
    def number_of_not_guaranteed_zero(self) -> int:
        return len(self._map)

    @property
    def number_of_parameters(self) -> int:
        return len(self._values)

    @property
    def params(self) -> Iterable[Tuple[int, float]]:
        return enumerate(self._values)

    @property
    def keys_with_param(self) -> Iterable[Tuple[Instance, int, float]]:
        for key, param_idx in self._map.items():
            value: float = self._values[param_idx]
            yield key, param_idx, value

    def __getitem__(self, key: Key) -> float:
        param_idx: Optional[int] = self.param_idx(key)
        if param_idx is None:
            return 0
        else:
            return self._values[param_idx]

    def param_value(self, param_idx: int) -> float:
        return self._values[param_idx]

    def param_idx(self, key: Key) -> Optional[int]:
        return self._map.get(_key_to_instance(key))

    # Mutators

    def __setitem__(self, key: Key, value: float) -> None:
        """
        Set the potential function value, for a given key.

        If value is zero, then the key will become "guaranteed zero".
        If the value is the same as an existing parameter value, then
        that parameter will be reused.

        Arg:
            key: defines an instance in the state space of the potential function.
            value: the new value of the potential function for the given key.

        Assumes:
            self.valid_key(key).
        """
        instance: Instance = _key_to_instance(key)

        param_idx: Optional[int] = self._map.get(instance)

        if param_idx is None:
            # previous value for the key was zero
            if value == 0:
                # nothing to do
                return
            param_idx: Optional[int] = self._inv_map.get(value)
            if param_idx is not None:
                # the value already exists in the function, so reuse it
                self._map[instance] = param_idx
                self._counts[param_idx] += 1
            else:
                # need to allocate a new value
                new_param_idx: int = len(self._values)
                self._values.append(value)
                self._counts.append(1)
                self._inv_map[value] = new_param_idx
                self._map[instance] = new_param_idx
            return

        # the key previously had a non-zero value
        prev_value: float = self._values[param_idx]

        if value == prev_value:
            # nothing to do
            return

        reference_count: int = self._counts[param_idx]
        if reference_count == 1:
            if value != 0:
                # simple case
                self._values[param_idx] = value
            else:
                # need to remove the parameter
                self._remove_param(param_idx)
                self._map.pop(instance)
                self._inv_map.pop(prev_value)
            return

        # decrement the reference count of the previous parameter
        self._counts[param_idx] = reference_count - 1

        # allocate the key to a different parameter
        param_idx: Optional[int] = self._inv_map.get(value)
        if param_idx is not None:
            # the value already exists in the function, so reuse it
            self._map[instance] = param_idx
            self._counts[param_idx] += 1
        else:
            # need to allocate a new value
            new_param_idx: int = len(self._values)
            self._values.append(value)
            self._counts.append(1)
            self._inv_map[value] = new_param_idx
            self._map[instance] = new_param_idx

    def set_iter(self, values: Iterable[float]) -> CompactPotentialFunction:
        """
        Set the values of the potential function using the given iterator.

        Mapping instances to *values is as follows:
            Given Factor(rv1, rv2) where rv1 has 2 states, and rv2 has 3 states:
            values[0] represents instance (0,0)
            values[1] represents instance (0,1)
            values[2] represents instance (0,2)
            values[3] represents instance (1,0)
            values[4] represents instance (1,1)
            values[5] represents instance (1,2).

        For example: to set to counts, starting from 1, use `self.set_iter(itertools.count(1))`.

        Args:
            values: an iterable providing values to use.

        Returns:
            self
        """
        self.clear()
        for instance, value in zip(self.instances(), values):
            self[instance] = value
        return self

    def set_stream(self, stream: Callable[[], float]) -> CompactPotentialFunction:
        """
        Set the values of the potential function by repeatedly calling the stream function.
        The order of values is the same as set_iter.

        For example, to set to random numbers, use `self.set_stream(random.random)`.

        Args:
            stream: a callable taking no arguments, returning the values to use.

        Returns:
            self
        """
        return self.set_iter(iter(stream, None))

    def set_flat(self, *value: float) -> CompactPotentialFunction:
        """
        Set the values of the potential function to the given values.
        The order of values is the same as set_iter.

        Args:
            *value: the values to use.

        Returns:
            self

        Raises:
            ValueError: if `len(value) != self.number_of_states`.
        """
        if len(value) != self.number_of_states:
            raise ValueError(f'wrong number of values: expected {self.number_of_states}, got {len(value)}')
        return self.set_iter(value)

    def set_all(self, value: float) -> CompactPotentialFunction:
        """
        Set all values of the potential function to the given value.

        Args:
            value: the value to use.

        Returns:
            self
        """
        self.clear()
        if value != 0:
            self._values = [value]
            self._counts = [self.number_of_states]
            self._inv_map = {value: 0}
            self._map = {instance: 0 for instance in self.instances()}
        return self

    def set_uniform(self) -> CompactPotentialFunction:
        """
        Set all values of the potential function 1/number_of_states.

        Returns:
            self
        """
        return self.set_all(1.0 / self.number_of_states)

    def clear(self) -> CompactPotentialFunction:
        """
        Set all values of the potential function to zero.
        
        Returns:
            self
        """
        self._values = []
        self._counts = []
        self._map = {}
        self._inv_map = {}
        return self

    def _remove_param(self, param_idx: int) -> None:
        """
        Remove the indexed parameter from self._params and self._counts.
        If the parameter is not at the end of the list of parameters
        then it will be swapped with the last parameter in the list.
        """

        # ensure the parameter is at the end of the list
        end: int = len(self._values) - 1
        if param_idx != end:
            # swap `param_idx` with `end`
            end_value: float = self._values[end]
            self._values[param_idx] = end_value
            self._counts[param_idx] = self._counts[end]
            self._inv_map[end_value] = param_idx
            for instance, instance_param_idx in self._map.items():
                if instance_param_idx == end:
                    self._map[instance] = param_idx

        # remove the end parameter
        self._values.pop()
        self._counts.pop()


class ClausePotentialFunction(PotentialFunction):
    """
    A clause potential function represents a clause From a CNF formula.
    I.e. a clause over variables X, Y, Z, is a disjunction of the form: 'X=x or Y=y or Z=z'.

    A clause potential function is guaranteed zero for a key where the clause is false,
    i.e., when 'X != x and Y != y and Z != z'.

    For keys where the clause is true, the value of the potential function
    is given by the only parameter of the potential function. That parameter
    is called the clause 'weight' and is notionally 1.

    The weight of a clause is permitted to be zero, but that is _not_ equivalent to
    guaranteed-zero.
    """

    def __init__(self, factor: Factor, key: Key, weight: float = 1):
        """
        Create a clause potential function for the given factor.

        Ensures:
            Does not hold a reference to the given factor.
            Does not register the potential function with the PGM.

        Raises:
             KeyError: if the key is not valid for the shape of the factor.

        Args:
            factor: which factor is this potential function is compatible with.
            key: defines the random variable states of the clause.
        """
        super().__init__(factor)
        self._weight: float = weight
        self._clause: Instance = self.check_key(key)
        self._num_not_guaranteed_zero: int = _zero_space(self.shape)

    @property
    def number_of_not_guaranteed_zero(self) -> int:
        return self._num_not_guaranteed_zero

    @property
    def number_of_parameters(self) -> int:
        return 1

    @property
    def params(self) -> Iterable[Tuple[int, float]]:
        return ((0, self._weight),)

    @property
    def keys_with_param(self) -> Iterable[Tuple[Instance, int, float]]:
        value = self._weight
        for i in range(self.number_of_rvs):
            key = list(self._clause)
            for j in range(self.shape[i]):
                key[i] = j
            yield tuple(key), 0, value

    def __getitem__(self, key: Key) -> float:
        instance: Instance = self.check_key(key)
        for key_state_idx, clause_state_idx in zip(instance, self._clause):
            if key_state_idx == clause_state_idx:
                return self._weight
        return 0

    def param_value(self, param_idx: int) -> float:
        if param_idx != 0:
            raise IndexError(param_idx)
        return self._weight

    def param_idx(self, key: Key) -> Optional[int]:
        instance: Instance = _key_to_instance(key)
        if instance == self._clause:
            return 0
        else:
            return None

    def is_cpt(self, tolerance=DEFAULT_TOLERANCE) -> bool:
        """
        A ClausePotentialFunction can only be a CTP when all entries are zero.
        """
        return -tolerance <= self._weight <= tolerance

    def is_sparse(self) -> bool:
        return True

    @property
    def weight(self) -> float:
        """
        Returns:
            the "weight" parameter defining the potential function.
        """
        return self._weight

    @property
    def clause(self) -> Instance:
        """
        Returns:
            the clause defining the potential function.
        """
        return self._clause

    # Mutators

    @weight.setter
    def weight(self, value: float) -> None:
        """
        Set the weight parameter to the given value.
        """
        self._weight = value

    @clause.setter
    def clause(self, key: Key) -> None:
        """
        Set the clause to the given key.

        Raises:
             KeyError: if the key is not valid for the shape of the factor.
        """
        self._clause = self.check_key(key)


class CPTPotentialFunction(PotentialFunction):
    """
    A potential function implementing a sparse Conditional Probability Table (CPT).

    The first random variable in the signature is the child, and the remaining random
    variables are parents.

    For each instantiation of the parent random variables there is a Conditioned Probability
    Distribution (CPD) over the states of the child random variable.

    If a CPD is not provided for a parent instantiation, then that parent instantiation
    is taken to have probability zero (i.e., all values of the CPD are guaranteed zero).
    """

    def __init__(self, factor: Factor, tolerance: float):
        """
        Create a CPT potential function for the given factor.

        Ensures:
            Does not hold a reference to the given factor.
            Does not register the potential function with the PGM.

        Args:
            factor: which factor is this potential function is compatible with.
            tolerance: a tolerance when testing if values are equal to zero or one.

        Raises:
            ValueError: if tolerance is negative.
        """
        super().__init__(factor)

        if tolerance < 0:
            raise ValueError('tolerance cannot be negative')

        self._child_size: int = self.shape[0]
        self._parent_shape: Shape = self.shape[1:]
        self._map: Dict[Instance, int] = {}
        self._values: List[float] = []
        self._inv_map: List[Instance] = []
        self._tolerance = tolerance

    @property
    def number_of_not_guaranteed_zero(self) -> int:
        return len(self._values)

    @property
    def number_of_parameters(self) -> int:
        return len(self._values)

    def is_cpt(self, tolerance=DEFAULT_TOLERANCE) -> bool:
        if tolerance >= self._tolerance:
            return True
        else:
            # The requested tolerance is tighter than ensured.
            # Need to use the default method.
            return super().is_cpt(tolerance)

    @property
    def params(self) -> Iterable[Tuple[int, float]]:
        return enumerate(self._values)

    @property
    def keys_with_param(self) -> Iterable[Tuple[Instance, int, float]]:
        child_size: int = self._child_size
        for param_idx, value in enumerate(self._values):
            parent: Instance = self._inv_map[param_idx // child_size]
            key: Instance = (param_idx % child_size,) + tuple(parent)
            yield key, param_idx, value

    def __getitem__(self, key: Key) -> float:
        param_idx: Optional[int] = self.param_idx(key)
        if param_idx is None:
            return 0
        else:
            return self._values[param_idx]

    def param_value(self, param_idx: int) -> float:
        return self._values[param_idx]

    def param_idx(self, key: Key) -> Optional[int]:
        instance: Instance = self.check_key(key)
        offset: Optional[int] = self._map.get(instance[1:])
        if offset is None:
            return None
        else:
            return offset + instance[0]

    @property
    def parent_shape(self) -> Shape:
        """
        What is the shape of the parents.
        """
        return self._parent_shape

    @property
    def number_of_parent_states(self) -> int:
        """
        How many combinations of parent states.
        """
        return _multiply(self._parent_shape)

    @property
    def number_of_child_states(self) -> int:
        """
        Number of child random variable states.

        This is the same as the number of values in each conditional
        probability distribution. This is equivalent to `self.shape[0]`.

        Returns:
            the number of child states.
        """
        return self._child_size

    def get_cpd(self, parent_states: Key) -> List[float]:
        """
        Get the CPD conditioned on parent states indicated by `parent_states`.
        
        Args:
            parent_states: indicates the parent states.

        Returns:
            The conditioned probability distribution.
        """
        parent_instance: Instance = check_key(self._parent_shape, parent_states)
        offset: Optional[int] = self._map.get(parent_instance)
        child_size: int = self._child_size
        if offset is None:
            return [0] * child_size
        else:
            return self._values[offset:offset + child_size]

    def cpds(self) -> Iterable[Tuple[Instance, Sequence[float]]]:
        """
        Iterate over (parent_states, cpd) tuples.
        This will exclude zero CPDs.
        Do not change CPDs to (or from) zero while iterating over them.
        
        Get the CPD conditioned on parent states indicated by `parent_states`.
        
        Returns:
            an iterator over pairs (instance, cpd) where,
            instance: is indicates the state of the parent random variables.
            cpd: is the conditioned probability distribution, for the parent instance.
        """
        for parent_instance, offset in self._map.items():
            cpd = self._values[offset:offset + self._child_size]
            yield parent_instance, cpd

    # Mutators

    def clear(self) -> CPTPotentialFunction:
        """
        Set all values of the potential function to zero.
        
        Returns:
            self
        """
        self._map = {}
        self._values = []
        self._inv_map = []
        return self

    def set_uniform(self) -> CPTPotentialFunction:
        """
        Set each CPD to a uniform distribution.
        
        Returns:
            self
        """
        self.clear()
        for parent_states in self.parent_instances():
            self.set_cpd_uniform(parent_states)
        return self

    def set_random(self, random: Callable[[], float], sparsity: float = 0) -> CPTPotentialFunction:
        """
        Set the values of the potential function to random CPDs.
        
        Args:
            random: is a stream of random numbers, assumed uniformly distributed in the interval [0, 1].
            sparsity: sets the expected proportion of probability values that are zero.
            
        Returns:
            self
        """
        self.clear()
        for parent_states in self.parent_instances():
            self.set_cpd_random(parent_states, random, sparsity)
        return self

    def set(self, *rows: Tuple[Key, Sequence[float]]) -> CPTPotentialFunction:
        """
        Calls self.set_cpd(parent_states, cpd) for each row (parent_states, cpd)
        in rows. Any unmentioned parent states will have zero probabilities.

        Example usage, assuming three Boolean random variables:
            pgm.Factor(x, y, z).set_cpt().set(
                # y  z    x[0] x[1]
                ((0, 0), (0.1, 0.9)),
                ((0, 1), (0.1, 0.9)),
                ((1, 0), (0.1, 0.9)),
                ((1, 1), (0.1, 0.9))
            )
        
        Args:
            *rows: are tuples (key, cpd) used to set the potential function values.
            
        Raises:
            ValueError: if a CPD is not valid.
            
        Returns:
            self
        """
        self.clear()
        for parent_states, cpd in rows:
            self.set_cpd(parent_states, cpd)
        return self

    def set_all(self, *cpds: Optional[Sequence[float]]) -> CPTPotentialFunction:
        """
        Set all CPDs using the given `cpds` which are taken to be in order of the parent states
        with the last variable of the parent changing state most rapidly, as per parent_states().

        If insufficient CPDs are provided then the remaining parent instantiations are taken to be
        impossible (i.e. not set and guaranteed zero).
        If too many CPDs are provided then the extras are ignored.
        Any list entry may be None, indicating 'guaranteed zero' for the associated parent states.

        Args:
            *cpds: are the CPDs used to set the potential function values.
            
        Raises:
            ValueError: if a CPD is not valid.
            
        Returns:
            self
        """
        self.clear()
        for parent_states, cpd in zip(self.parent_instances(), cpds):
            self.set_cpd(parent_states, cpd)
        return self

    def set_cpd(self, parent_states: Key, cpd: Optional[Sequence[float]]) -> CPTPotentialFunction:
        """
        Set the CPD of the given parent states to the given cpd.
        If cpd is None or all zeros, then this is equivalent to clear_cpd(parent_states).
        
        Args:
            parent_states: indicates the CPD to set, based on the parent states.
            cpd: is a conditioned probability distribution, or None indicating `guaranteed zero`.
            
        Raises:
            ValueError: if the CPD is not valid.
            KeyError if the key is not valid.

        Returns:
            self
        """
        parent_instance: Instance = check_key(self._parent_shape, parent_states)

        if cpd is None:
            self._clear_cpd(parent_instance)
            return self

        if len(cpd) != self._child_size:
            raise ValueError(f'CPD incorrect size: expected {self._child_size}, got {len(cpd)}')
        if not all(0 <= value <= 1 for value in cpd):
            raise ValueError(f'not a valid CPD: {cpd!r}')

        total_value = sum(cpd)
        if total_value < self._tolerance:
            self._clear_cpd(parent_instance)
            return self

        if total_value < 1 - self._tolerance or total_value > 1 + self._tolerance:
            raise ValueError(f'not a valid CPD: sum of values = {total_value}')

        offset: Optional[int] = self._map.get(parent_instance)
        child_size: int = self._child_size
        if offset is None:
            offset = len(self._values)
            self._values.extend(cpd)
            self._map[parent_instance] = offset
            self._inv_map.append(parent_instance)
        else:
            self._values[offset:offset + child_size] = cpd

        return self

    def clear_cpd(self, parent_states: Key) -> CPTPotentialFunction:
        """
        Set the CPD of the given parent_states to all 'guaranteed zero'.
        
        Args:
            parent_states: indicates the CPD to clear, based on the parent states.

        Raises:
            KeyError if the key is not valid.

        Returns:
            self
        """
        parent_instance: Instance = check_key(self._parent_shape, parent_states)
        self._clear_cpd(parent_instance)
        return self

    def set_cpd_uniform(self, parent_states: Key) -> CPTPotentialFunction:
        """
        Set the CPD of the given parent_states to a uniform CPD.
        
        Args:
            parent_states: indicates the CPD to clear, based on the parent states.

        Raises:
            KeyError if the key is not valid.

        Returns:
            self
        """
        num_states = self.number_of_child_states
        cpd = [1.0 / num_states] * num_states
        return self.set_cpd(parent_states, cpd)

    def set_cpd_random(
            self,
            parent_states: Key,
            random: Callable[[], float],
            sparsity: float = 0,
    ) -> CPTPotentialFunction:
        """
        Set the CPD of the given parent_states to a random CPD.

        Args:
            parent_states: identifies the CPD being set.
            random: is a stream of random numbers, assumed uniformly distributed in the interval [0, 1].
            sparsity: sets the expected proportion of probability values that are zero.

        Returns:
            self
        """
        cpd = np.zeros(self.number_of_child_states, dtype=np.float64)
        if sparsity <= 0:
            for i in range(len(cpd)):
                cpd[i] = 0.0000001 + random()
        else:
            for i in range(len(cpd)):
                if random() > sparsity:
                    cpd[i] = 0.0000001 + random()
        sum_value = np.sum(cpd)
        if sum_value > 0:
            cpd /= sum_value
            return self.set_cpd(parent_states, cpd)
        else:
            return self.clear_cpd(parent_states)

    def _clear_cpd(self, parent_instance: Instance) -> None:
        """
        Remove the parent instance from the parameters
        """
        offset: Optional[int] = self._map.get(parent_instance)
        if offset is None:
            # nothing to do
            return

        child_size: int = self._child_size
        end_offset: int = len(self._values) - child_size
        if offset != end_offset:
            # need to swap parameters
            end_cpd = self._values[end_offset:]
            end_parent_instance = self._inv_map[-1]

            self._values[offset:offset + child_size] = end_cpd
            self._map[end_parent_instance] = offset
            self._inv_map[offset // child_size] = end_parent_instance

        self._map.pop(parent_instance)
        self._inv_map.pop()
        for _ in range(child_size):
            self._values.pop()


def default_pgm_name(pgm: PGM) -> str:
    """
    If no name is provided to a PGM constructor, then this will be the default name for the PGM.

    Args:
        pgm: a PGM object.

    Returns:
        a name for the PGM if none is given at construction time.
    """
    return 'PGM_' + str(id(pgm))


def check_key(shape: Shape, key: Key) -> Instance:
    """
    Convert the key into an instance.

    Args:
        shape: the shape defining the state space.
        key: a key into the state space.

    Returns:
        A instance from the state space, as a tuple of state indexes, co-indexed with the given shape.

    Raises:
        KeyError if the key is not valid.
    """
    _key: Instance = _key_to_instance(key)
    if len(_key) != len(shape):
        raise KeyError(f'not a valid key for shape {shape}: {key!r}')
    if all((0 <= i <= m) for i, m in zip(_key, shape)):
        return tuple(_key)
    raise KeyError(f'not a valid key for shape {shape}: {key!r}')


def valid_key(shape: Shape, key: Key) -> bool:
    """
    Is the given key valid.

    Args:
        shape: the shape defining the state space.
        key: a key into the state space.

    Returns:
        True only if tke key is valid for the given shape.
    """
    try:
        check_key(shape, key)
        return True
    except KeyError:
        return False


def number_of_states(*rvs: RandomVariable) -> int:
    """
    Returns:
        What is the size of the state space, i.e., `multiply(len(rv) for rv in self.rvs)`.
    """
    return _multiply(len(rv) for rv in rvs)


def rv_instances(*rvs: RandomVariable, flip: bool = False) -> Iterable[Instance]:
    """
    Enumerate instances of the given random variables.

    Each instance is a tuples of state indexes, co-indexed with the given random variables.

    The order is the natural index order (i.e., last random variable changing most quickly).

    Args:
        flip: if true, then first random variable changes most quickly.

    Returns:
        an iteration over tuples, each tuple holds state indexes
        co-indexed with the given random variables.
    """
    shape = [len(rv) for rv in rvs]
    return _combos_ranges(shape, flip=not flip)


def rv_instances_as_indicators(*rvs: RandomVariable, flip: bool = False) -> Iterable[Sequence[Indicator]]:
    """
    Enumerate instances of the given random variables.

    Each instance is a tuples of indicators, co-indexed with the given random variables.

    The order is the natural index order (i.e., last random variable changing most quickly).

    Args:
        flip: if true, then first random variable changes most quickly.

    Returns:
        an iteration over tuples, each tuples holds random variable indicators
        co-indexed with the given random variables.
    """
    return _combos(rvs, flip=not flip)


def _key_to_instance(key: Key) -> Instance:
    """
    Convert a key to an instance.

    Args:
        key: a key into a state space.

    Returns:
        A instance from the state space, as a tuple of state indexes, co-indexed with the given shape.

    Assumes:
        The key is valid for the implied state space.
    """
    if isinstance(key, int):
        return (key,)
    else:
        return tuple(key)


def _natural_key_idx(shape: Shape, key: Key) -> int:
    """
    What is the natural index of the given key, assuming the given shape.

    Args:
        shape: the shape defining the state space.
        key: a key into the state space.

    Returns:
        an index as per enumerated instances in their natural order, i.e.
        last random variable changing most quickly.

    Assumes:
        The key is valid for the shape.
    """
    instance: Instance = _key_to_instance(key)
    result: int = instance[0]
    for s, i in zip(shape[1:], instance[1:]):
        result = result * s + i
    return result


def _zero_space(shape: Shape) -> int:
    """
    Return the size of the zero space of the given shape. This is the number
    of possible instances in the state space that do not have a zero in the instance.

    The zero space is the same as the shape but with one less state
    for each random variable.

    Args:
        shape: the shape defining the state space.

    Returns:
        the size of the zero space.
    """
    return _multiply(x - 1 for x in shape)


def _normalise_potential_function(
        function: Union[DensePotentialFunction, SparsePotentialFunction],
        grouping_positions: Sequence[int],
) -> None:
    """
    Convert the potential function to a CPT with 'grouping_positions' nominating
    the parent random variables.
    
    I.e., for each possible key of the function with the same value at each
    grouping position, the sum of values for matching keys in the factor is scaled
    to be 1 (or 0).

    Parameter 'grouping_positions' are indices into `function.shape`. For example, the
    grouping positions of a factor with parent rvs 'conditioning_rvs', then
    grouping_positions = [i for i, rv in enumerate(factor.rvs) if rv in conditioning_rvs].

    Args:
        function: the potential function to normalise.
        grouping_positions: indices into `function.shape`.
    """
    if len(grouping_positions) == 0:
        total = sum(
            function.param_value(param_idx)
            for param_idx in range(function.number_of_parameters)
        )
        if total != 0 and total != 1:
            for param_key, param_idx, param_value in function.keys_with_param:
                function.set_param_value(param_idx, param_value / total)
    else:
        group_sum = {}
        for param_key, param_idx, param_value in function.keys_with_param:
            group = tuple(param_key[i] for i in grouping_positions)
            group_sum[group] = group_sum.get(group, 0) + param_value

        for param_key, param_idx, param_value in function.keys_with_param:
            group = tuple(param_key[i] for i in grouping_positions)
            total = group_sum[group]
            if total > 0:
                function.set_param_value(param_idx, param_value / total)


_CLEAN_CHARS: Set[str] = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-+~?.')


def _clean_str(s) -> str:
    """
    Quote a string if empty or not all characters are in _CLEAN_CHARS.
    This is used when rendering indicators.
    """
    s = str(s)
    if len(s) == 0 or not all(c in _CLEAN_CHARS for c in s):
        return repr(s)
    else:
        return s
