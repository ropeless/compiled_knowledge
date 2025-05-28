from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple, List, Iterator, Set, Iterable, Optional, Callable

import numpy as np

from ck.circuit import Circuit, VarNode, CircuitNode
from ck.pgm import PGM, ParamId, Factor, PotentialFunction, RandomVariable, ZeroPotentialFunction
from ck.pgm_circuit.slot_map import SlotMap, SlotKey
from ck.pgm_compiler.support.circuit_table import CircuitTable, TableInstance
from ck.utils.iter_extras import pairs
from ck.utils.map_list import MapList
from ck.utils.np_extras import NDArray, NDArrayFloat64


@dataclass
class FactorTables:
    circuit: Circuit  # The host circuit
    number_of_indicators: int  # number of indicator variables
    number_of_parameters: int  # number of parameter variables (i.e., non-const, in-use parameters)
    slot_map: SlotMap  # map from Indicator or ParamId object to a circuit var index.
    tables: Sequence[CircuitTable]  # one CircuitTable for each PGM factor.

    # For a non-const, in-use parameter with id `param_id`, the PGM value of that
    # parameter was `self.parameter_values[self.slot_map[param_id] - self.number_of_indicators]`.
    parameter_values: NDArray

    def get_table(self, factor: Factor) -> CircuitTable:
        return self.tables[factor.idx]


def make_factor_tables(
        pgm: PGM,
        const_parameters: bool,
        multiply_indicators: bool,
        pre_prune_factor_tables: bool,
) -> FactorTables:
    """
    Consistently and efficiently create circuit tables for factors of a PGM.

    Creates:
    * a circuit,
    * a circuit variable for each indicator of the PGM,
    * a circuit variable for each non-constant, in-use potential function parameter.
    * a circuit table for each Factor of the PGM,

    The parameter of each potential function will be converted either
    eiter to a circuit constant (if const_parameters is true) or a circuit
    variable (if const_parameters is false).

    Random variables will be multiplied into factor circuit tables if
    `multiply_indicators` is true.

    A slot map will be created that maps PGM indicators and parameter ids to circuit var indices.
    Specifically, a circuit var will be added for each indicator,
    in the order they appear in `pgm.indicators`. Circuit vars for parameter ids will be added
    after those for indicators, and only if const_parameters is false.

    Args:
        pgm: The PGM with the random variables, factors, and potential functions.
        const_parameters: if true, then potential function parameters will be circuit constants,
            otherwise they will be circuit variables, with entries in the returned slot map.
        multiply_indicators: if true then indicator variables will be multiplied into an acceptable
            factor.
        pre_prune_factor_tables: if true, then heuristics will be used to remove any provably zero row.

    Returns:
        FactorTables, holding a slot_map and a circuit table for each PGM factor.
    """

    # Create circuit and initialise the slot map with indicator variables
    circuit = Circuit()
    slot_map: Dict[SlotKey, int] = {
        indicator: circuit.new_var().idx
        for indicator in pgm.indicators
    }

    # Get the circuit table rows for each potential function
    # functions_rows[id(function)] = rows for the function
    functions_rows: Dict[int, _FunctionRows]
    if const_parameters:
        functions_rows = {
            id(function): _rows_for_function_const(function, circuit)
            for function in pgm.functions
        }
    else:
        functions_rows = {
            id(function): _rows_for_function_var(function, circuit, slot_map)
            for function in pgm.functions
        }

    # Link factors to function rows.
    # factor_rows[id(factor)] = rows for the factor
    factor_rows: Dict[int, _FactorRows] = {}
    for factor in pgm.factors:
        rows: _FunctionRows = functions_rows[id(factor.function)]
        rows.use_count += 1
        factor_rows[id(factor)] = _FactorRows(factor, rows)

    # Check to see if any factor rows can be pre-pruned.
    if pre_prune_factor_tables:
        _pre_prune_factor_tables(list(factor_rows.values()))

    # Allocated random variables to factors
    factors_mul_rvs: MapList[int, RandomVariable]
    if multiply_indicators:
        def _factor_size(_factor: Factor) -> int:
            return len(factor_rows[id(_factor)])

        factors_mul_rvs = _assign_rvs_to_factors(pgm, _factor_size)
    else:
        factors_mul_rvs = MapList()  # no assignment of rvs to factors.

    # Make a circuit table for each factor. `tables[factor.index]` is the circuit table for `factor`.
    tables: List[CircuitTable] = [
        _make_factor_table(factor, circuit, slot_map, factor_rows[id(factor)], factors_mul_rvs)
        for factor in pgm.factors
    ]

    # Extract the parameter values (if they are circuit vars).
    number_of_indicators: int = pgm.number_of_indicators
    number_of_parameters: int = len(slot_map) - number_of_indicators
    parameter_values: NDArrayFloat64 = np.zeros(number_of_parameters, dtype=np.float64)
    if not const_parameters:
        for function in pgm.functions:
            for param_index, value in function.params:
                param_id: ParamId = function.param_id(param_index)
                slot: Optional[int] = slot_map.get(param_id)
                if slot is not None:
                    parameter_values[slot - number_of_indicators] = value

    return FactorTables(
        circuit=circuit,
        number_of_indicators=number_of_indicators,
        number_of_parameters=number_of_parameters,
        slot_map=slot_map,
        tables=tables,
        parameter_values=parameter_values,
    )


def _assign_rvs_to_factors(
        pgm: PGM,
        factor_size: Callable[[Factor], int],
) -> MapList[int, RandomVariable]:
    """
    Assign each random variable to the smallest factor containing it.

    Returns:
        a map from factor id to list of random variables assigned to that factor
    """
    factors = pgm.factors
    rvs = pgm.rvs

    # For each rv, get the factors it is in
    rv_factors: MapList[int, Factor] = MapList()  # rv index to list of Factors with that rv.
    for factor in factors:
        for rv in factor.rvs:
            rv_factors.append(rv.idx, factor)

    # For each rv, assign it to a factor for multiplication
    factors_mul_rvs: MapList[int, RandomVariable] = MapList()  # factor id to list of rvs
    for rv_index in range(len(rvs)):
        candidates: Sequence[Factor] = rv_factors.get(rv_index, ())
        if len(candidates) > 0:
            best_factor = min(candidates, key=factor_size)
            factors_mul_rvs.append(id(best_factor), rvs[rv_index])

    return factors_mul_rvs


class _FunctionRows:
    def __init__(self, rows: Dict[TableInstance, CircuitNode], use_count: int = 0):
        self.rows: Dict[TableInstance, CircuitNode] = rows
        self.use_count: int = use_count


class _FactorRows:
    def __init__(self, factor: Factor, rows: _FunctionRows):
        self.rows: _FunctionRows = rows
        self.rv_indexes: Tuple[int, ...] = tuple(rv.idx for rv in factor.rvs)

    def __len__(self) -> int:
        return len(self.rows.rows)

    def items(self) -> Iterable[Tuple[TableInstance, CircuitNode]]:
        return self.rows.rows.items()

    def prune(self, extra_keys: Set[TableInstance]) -> None:
        """
        Remove the given keys from the factor's function rows.
        """
        if len(extra_keys) > 0:
            new_rows: Dict[TableInstance, CircuitNode] = {
                instance: node
                for instance, node in self.rows.rows.items()
                if instance not in extra_keys
            }
            if self.rows.use_count > 1:
                self.rows.use_count -= 1
                self.rows = _FunctionRows(new_rows, 1)
            else:
                self.rows.rows = new_rows


class _FactorPair:
    def __init__(self, x: _FactorRows, y: _FactorRows):
        self.x: _FactorRows = x
        self.y: _FactorRows = y

        x_set = set(self.x.rv_indexes)

        # Identify all random variables used by x and y
        self.all_rv_indexes: Set[int] = x_set.union(self.y.rv_indexes)

        # Identify common random variables between x and y
        # Keep them in a stable order
        self.co_rv_indexes: Tuple[int, ...] = tuple(x_set.intersection(self.y.rv_indexes))

        # Cache mappings from result Instance to index into source Instance (x or y).
        # This will be used in indexing and product loops to pull our needed values
        # from the source instances.
        self.co_from_x_map = tuple(x.rv_indexes.index(rv_index) for rv_index in self.co_rv_indexes)
        self.co_from_y_map = tuple(y.rv_indexes.index(rv_index) for rv_index in self.co_rv_indexes)

    def prune(self) -> None:
        """
        Prune any rows from x and y that cannot join to each other.
        """
        co_from_x_map = self.co_from_x_map
        co_from_y_map = self.co_from_y_map
        x_rows = self.x.rows.rows
        y_rows = self.y.rows.rows

        x_co_set: Set[TableInstance] = {
            tuple(instance[i] for i in co_from_x_map)
            for instance in x_rows.keys()
        }

        y_co_set: Set[TableInstance] = {
            tuple(instance[i] for i in co_from_y_map)
            for instance in y_rows.keys()
        }

        # Keys in x that will not join to y
        x_extra_keys: Set[TableInstance] = {
            instance
            for instance in x_rows.keys()
            if tuple(instance[i] for i in co_from_x_map) not in y_co_set
        }

        # Keys in y that will not join to x
        y_extra_keys: Set[TableInstance] = {
            instance
            for instance in y_rows.keys()
            if tuple(instance[i] for i in co_from_y_map) not in x_co_set
        }

        self.x.prune(x_extra_keys)
        self.y.prune(y_extra_keys)


def _pre_prune_factor_tables(factor_rows: Sequence[_FactorRows]) -> None:
    """
    It may be possible to reduce the size of a table for a factor.

    If two factors contain a common random variable then at some point their product
    will be formed, which may eliminate rows. This method identifies and removes
    such rows.
    """
    # Find all pairs of factors that have at least one common random variable.
    pairs_to_check: List[_FactorPair] = [
        _FactorPair(f1, f2)
        for f1, f2 in pairs(factor_rows)
        if not set(f1.rv_indexes).isdisjoint(f1.rv_indexes)
    ]

    # Simple version.
    for pair in pairs_to_check:
        pair.prune()

    # Earlier version.
    # This version re-checks processed pairs that may get benefit from a subsequent pruning.
    # Unfortunately, this is computationally expensive, and provides no practical benefit.
    #
    # pairs_done: List[_FactorPair] = []
    # while len(pairs_to_check) > 0:
    #     pair: _FactorPair = pairs_to_check.pop()
    #     x: _FactorRows = pair.x
    #     y: _FactorRows = pair.y
    #
    #     x_size = len(x)
    #     y_size = len(y)
    #     pair.prune()
    #
    #     # See if any pairs need re-checking
    #     rvs_affected: Set[int] = set()
    #     if x_size != len(x):
    #         rvs_affected.update(x.rv_indexes)
    #     if y_size != len(y):
    #         rvs_affected.update(y.rv_indexes)
    #     if len(rvs_affected) > 0:
    #         next_pairs_done: List[_FactorPair] = []
    #         for pair in pairs_done:
    #             if rvs_affected.isdisjoint(pair.all_rv_indexes):
    #                 next_pairs_done.append(pair)
    #             else:
    #                 pairs_to_check.append(pair)
    #         pairs_done = next_pairs_done
    #
    #     # Mark the current pair as done.
    #     pairs_done.append(pair)


def _make_factor_table(
        factor: Factor,
        circuit: Circuit,
        slot_map: Dict[SlotKey, int],
        rows: _FactorRows,
        factors_mul_rvs: MapList[int, RandomVariable],
) -> CircuitTable:
    # Get random variables to multiply into the table
    factor_mul_rvs: Sequence[RandomVariable] = factors_mul_rvs.get(id(factor), ())

    # Create the empty circuit table
    factor_rv_indexes: Sequence[int] = tuple(rv.idx for rv in factor.rvs)

    if len(factor_mul_rvs) == 0:
        # Trivial case - no random variables to multiply into the table.
        return CircuitTable(circuit, factor_rv_indexes, rows.items())

    # Work out what element in an instance of the factor will select the indicator
    # variable for each mul rv.
    # inst_to_mul[i] is the index into factor.rvs for factor_mul_rvs[i]
    inst_to_mul: Sequence[int] = tuple(factor_rv_indexes.index(rv.idx) for rv in factor_mul_rvs)

    # Map a state index of a mul rv to its indicator circuit variable.
    # mul_rvs_vars[i][j] is the indicator circuit variable for factor_mul_rvs[i][j]
    mul_rvs_vars: Sequence[Sequence[CircuitNode]] = tuple(
        tuple(circuit.vars[slot_map[ind]] for ind in rv.indicators)
        for rv in factor_mul_rvs
    )

    def _result_rows() -> Iterator[Tuple[TableInstance, CircuitNode]]:
        for instance, node in rows.items():
            to_mul = tuple(
                mul_vars[instance[inst_index]]
                for inst_index, mul_vars in zip(inst_to_mul, mul_rvs_vars)
            )
            if not node.is_one:
                to_mul += (node,)
            if len(to_mul) == 0:
                yield instance, circuit.one
            elif len(to_mul) == 1:
                yield instance, to_mul[0]
            else:
                yield instance, circuit.optimised_mul(to_mul)

    return CircuitTable(circuit, factor_rv_indexes, _result_rows())


def _rows_for_function_const(
        function: PotentialFunction,
        circuit: Circuit,
) -> _FunctionRows:
    """
    Get the rows (instance, node) for the given potential function
    where each node is a circuit constant.
    This will exclude zero values.
    """
    if isinstance(function, ZeroPotentialFunction):
        # shortcut
        return _FunctionRows({})

    return _FunctionRows({
        tuple(instance): circuit.const(value)
        for instance, _, value in function.keys_with_param
        if value != 0
    })


def _rows_for_function_var(
        function: PotentialFunction,
        circuit: Circuit,
        slot_map: Dict[SlotKey, int],
) -> _FunctionRows:
    """
    Get the rows (instance, node) for the given potential function
    where each node is a circuit variable.
    """

    def _create_param_var(param_id: ParamId) -> VarNode:
        """
        Create a circuit variable for the given parameter id.
        This assumes one does not already exist for the parameter id.
        """
        assert param_id not in slot_map.keys(), 'parameter should not already have a circuit var'
        node: VarNode = circuit.new_var()
        slot_map[param_id] = node.idx
        return node

    return _FunctionRows({
        tuple(instance): _create_param_var(function.param_id(param_index))
        for instance, param_index, _ in function.keys_with_param
    })
