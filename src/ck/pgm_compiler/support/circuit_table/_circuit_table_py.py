from __future__ import annotations

from typing import Sequence, Tuple, Dict, Iterable, Set, Iterator

from ck.circuit import CircuitNode, Circuit
from ck.utils.map_list import MapList

TableInstance = Tuple[int, ...]


class CircuitTable:
    """
    A circuit table manages a set of CircuitNodes, where each node corresponds
    to an instance for a set of (zero or more) random variables.

    Operations on circuit tables typically add circuit nodes to the circuit. It will
    heuristically avoid adding unnecessary nodes (e.g. addition of zero, multiplication
    by zero or one.) However, it may be that interim circuit nodes are created that
    end up not being used. Consider calling `Circuit.remove_unreachable_op_nodes` after
    completing all circuit table operations.

    It is generally expected that no CircuitTable row will be created with a constant
    zero node. These are assumed to be optimised out already.
    """

    def __init__(
            self,
            circuit: Circuit,
            rv_idxs: Sequence[int],
            rows: Iterable[Tuple[TableInstance, CircuitNode]] = (),
    ):
        """
        Args:
            circuit: the circuit whose nodes are being managed by this table.
            rv_idxs: indexes of random variables.
            rows: optional rows to add to the table.

        Assumes:
            * rv_idxs contains no duplicates.
            * all row instances conform to the indexed random variables.
            * all row circuit nodes belong to the given circuit.
        """
        self._circuit: Circuit = circuit
        self._rv_idxs: Tuple[int, ...] = tuple(rv_idxs)
        self._rows: Dict[TableInstance, CircuitNode] = dict(rows)

    @property
    def circuit(self) -> Circuit:
        return self._circuit

    @property
    def rv_idxs(self) -> Tuple[int, ...]:
        return self._rv_idxs

    def __len__(self) -> int:
        return len(self._rows)

    def get(self, key, default=None):
        return self._rows.get(key, default)

    def keys(self) -> Iterable[TableInstance]:
        return self._rows.keys()

    def values(self) -> Iterable[CircuitNode]:
        return self._rows.values()

    def items(self) -> Iterable[Tuple[TableInstance, CircuitNode]]:
        return self._rows.items()

    def __getitem__(self, key):
        return self._rows[key]

    def __setitem__(self, key, value):
        self._rows[key] = value

    def top(self) -> CircuitNode:
        """
        Get the circuit top value.

        Raises:
            RuntimeError if there is more than one row in the table.

        Returns:
            A single circuit node.
        """
        if len(self._rows) == 0:
            return self._circuit.zero
        elif len(self._rows) == 1:
            return next(iter(self._rows.values()))
        else:
            raise RuntimeError('cannot get top node from a table with more that 1 row')


# ==================================================================================
#  Circuit Table Operations
# ==================================================================================


def sum_out(table: CircuitTable, rv_idxs: Iterable[int]) -> CircuitTable:
    """
    Return a circuit table that results from summing out
    the given random variables of this circuit table.

    Normally this will return a new table. However, if rv_idxs is empty,
    then the given table is returned unmodified.

    Raises:
        ValueError if rv_idxs is not a subset of table.rv_idxs.
        ValueError if rv_idxs contains duplicates.
    """
    rv_idxs: Sequence[int] = tuple(rv_idxs)

    if len(rv_idxs) == 0:
        # nothing to do
        return table

    rv_idxs_set: Set[int] = set(rv_idxs)
    if len(rv_idxs_set) != len(rv_idxs):
        raise ValueError('rv_idxs contains duplicates')
    if not rv_idxs_set.issubset(table.rv_idxs):
        raise ValueError('rv_idxs is not a subset of table.rv_idxs')

    remaining_rv_idxs = tuple(
        rv_index
        for rv_index in table.rv_idxs
        if rv_index not in rv_idxs_set
    )
    num_remaining = len(remaining_rv_idxs)
    if num_remaining == 0:
        # Special case: summing out all random variables
        return sum_out_all(table)

    # index_map[i] is the location in table.rv_idxs for remaining_rv_idxs[i]
    index_map = tuple(
        table.rv_idxs.index(remaining_rv_index)
        for remaining_rv_index in remaining_rv_idxs
    )

    # This is a one-pass version to sum the groups. The
    # two-pass version (below) seems to have better performance.
    #
    # circuit: Circuit = table.circuit
    # result = CircuitTable(circuit, remaining_rv_idxs)
    # result_rows: Dict[TableInstance, CircuitNode] = result._rows
    # for instance, node in table.items():
    #     group_instance = tuple(instance[i] for i in index_map)
    #     prev_sum = result_rows.get(group_instance)
    #     if prev_sum is None:
    #         result_rows[group_instance] = node
    #     else:
    #         result_rows[group_instance] = circuit.add(prev_sum, node)
    # return result

    groups: MapList[TableInstance, CircuitNode] = MapList()
    for instance, node in table.items():
        group_instance = tuple(instance[i] for i in index_map)
        groups.append(group_instance, node)
    circuit: Circuit = table.circuit
    return CircuitTable(
        circuit,
        remaining_rv_idxs,
        (
            (group, circuit.add(to_add))
            for group, to_add in groups.items()
        )
    )


def sum_out_all(table: CircuitTable) -> CircuitTable:
    """
    Return a circuit table that results from summing out
    all random variables of this circuit table.
    """
    circuit: Circuit = table.circuit
    num_rows: int = len(table)
    if num_rows == 0:
        return CircuitTable(circuit, ())
    elif num_rows == 1:
        node = next(iter(table.values()))
    else:
        node: CircuitNode = circuit.optimised_add(table.values())
        if node.is_zero:
            return CircuitTable(circuit, ())

    return CircuitTable(circuit, (), [((), node)])


def project(table: CircuitTable, rv_idxs: Iterable[int]) -> CircuitTable:
    """
    Call `sum_out(table, to_sum_out)`, where
    `to_sum_out = table.rv_idxs - rv_idxs`.
    """
    to_sum_out: Set[int] = set(table.rv_idxs)
    to_sum_out.difference_update(rv_idxs)
    return sum_out(table, to_sum_out)


def product(x: CircuitTable, y: CircuitTable) -> CircuitTable:
    """
    Return a circuit table that results from the product of the two given tables.

    If x or y have a single row with value 1, then the other table is returned. Otherwise,
    a new circuit table will be constructed and returned.
    """
    circuit: Circuit = x.circuit
    if y.circuit is not circuit:
        raise ValueError('circuit tables must refer to the same circuit')

    # Make the smaller table 'y', and the other 'x'.
    # This is to minimise the index size on 'y'.
    if len(x) < len(y):
        x, y = y, x

    x_rv_idxs: Tuple[int, ...] = x.rv_idxs
    y_rv_idxs: Tuple[int, ...] = y.rv_idxs

    # Special case: y == 0 or 1, and has no random variables.
    if y_rv_idxs == ():
        if len(y) == 1 and y.top().is_one:
            return x
        elif len(y) == 0:
            return CircuitTable(circuit, x_rv_idxs)

    # Set operations on rv indexes. After these operations:
    # * co_rv_idxs is the set of rv indexes common (co) to x and y,
    # * yo_rv_idxs is the set of rv indexes in y only (yo), and not in x.
    yo_rv_idxs_set: Set[int] = set(y_rv_idxs)
    co_rv_idxs_set: Set[int] = set(x_rv_idxs)
    co_rv_idxs_set.intersection_update(yo_rv_idxs_set)
    yo_rv_idxs_set.difference_update(co_rv_idxs_set)

    if len(co_rv_idxs_set) == 0:
        # Special case: no common random variables.
        return _product_no_common_rvs(x, y)

    # Convert random variable index sets to sequences
    yo_rv_idxs: Tuple[int, ...] = tuple(yo_rv_idxs_set)  # y only random variables
    co_rv_idxs: Tuple[int, ...] = tuple(co_rv_idxs_set)  # common random variables

    # Cache mappings from result Instance to index into source Instance (x or y).
    # This will be used in indexing and product loops to pull our needed values
    # from the source instances.
    co_from_x_map = tuple(x.rv_idxs.index(rv_index) for rv_index in co_rv_idxs)
    co_from_y_map = tuple(y.rv_idxs.index(rv_index) for rv_index in co_rv_idxs)
    yo_from_y_map = tuple(y.rv_idxs.index(rv_index) for rv_index in yo_rv_idxs)

    # Index the y rows by common-only key (y is the smaller of the two tables).
    y_index: MapList[TableInstance, Tuple[TableInstance, CircuitNode]] = MapList()
    for y_instance, y_node in y.items():
        co = tuple(y_instance[i] for i in co_from_y_map)
        yo = tuple(y_instance[i] for i in yo_from_y_map)
        y_index.append(co, (yo, y_node))

    def _result_rows() -> Iterator[Tuple[TableInstance, CircuitNode]]:
        # Iterate over x rows, yielding (instance, value).
        # Rows with constant node values of one are optimised out.
        for _x_instance, _x_node in x.items():
            _co = tuple(_x_instance[i] for i in co_from_x_map)
            if _x_node.is_one:
                # Multiplying by one.
                # Iterate over matching y rows.
                for _yo, _y_node in y_index.get(_co, ()):
                    yield _x_instance + _yo, _y_node
            else:
                # Iterate over matching y rows.
                for _yo, _y_node in y_index.get(_co, ()):
                    if _y_node.is_one:
                        yield _x_instance + _yo, _x_node
                    else:
                        yield _x_instance + _yo, circuit.mul(_x_node, _y_node)

    return CircuitTable(circuit, x_rv_idxs + yo_rv_idxs, _result_rows())


def _product_no_common_rvs(x: CircuitTable, y: CircuitTable) -> CircuitTable:
    """
    Return the product of x and y, where x and y have no common random variables.

    This is an optimisation of more general product algorithm as no index needs
    to be construction based on the common random variables.

    Rows with constant node values of one are optimised out.

    Assumes:
        * There are no common random variables between x and y.
        * x and y are for the same circuit.
    """
    circuit: Circuit = x.circuit

    result_rv_idxs: Tuple[int, ...] = x.rv_idxs + y.rv_idxs

    def _result_rows() -> Iterator[Tuple[TableInstance, CircuitNode]]:
        for x_instance, x_node in x.items():
            if x_node.is_one:
                for y_instance, y_node in y.items():
                    yield x_instance + y_instance, y_node
            else:
                for y_instance, y_node in y.items():
                    if y_node.is_one:
                        yield x_instance + y_instance, y_node
                    else:
                        yield x_instance + y_instance, circuit.mul(x_node, y_node)

    return CircuitTable(circuit, result_rv_idxs, _result_rows())
