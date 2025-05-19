from __future__ import annotations

from typing import Sequence, Tuple, Iterable, Iterator

from ck.circuit import CircuitNode, Circuit, OpNode, MUL

TableInstance = Tuple[int, ...]


cdef class CircuitTable:
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

    cdef public object circuit
    cdef public tuple[int, ...] rv_idxs
    cdef public dict[tuple[int, ...], CircuitNode] rows

    def __init__(
            self,
            circuit: Circuit,
            rv_idxs: Sequence[int, ...],
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
        self.circuit = circuit
        self.rv_idxs = tuple(rv_idxs)
        self.rows = dict(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def get(self, key, default=None):
        return self.rows.get(key, default)

    def __getitem__(self, key):
        return self.rows[key]

    def __setitem__(self, key, value):
        self.rows[key] = value

    cpdef object top(self):  # -> CircuitNode:
        # Get the circuit top value.
        #
        # Raises:
        #     RuntimeError if there is more than one row in the table.
        #
        # Returns:
        #     A single circuit node.
        cdef int number_of_rows = len(self.rows)
        if number_of_rows == 0:
            return self.circuit.zero
        elif number_of_rows == 1:
            return next(iter(self.rows.values()))
        else:
            raise RuntimeError('cannot get top node from a table with more that 1 row')


# ==================================================================================
#  Circuit Table Operations
# ==================================================================================

cpdef object sum_out(object table: CircuitTable, object rv_idxs: Iterable[int]):  # -> CircuitTable:
    # Return a circuit table that results from summing out
    # the given random variables of this circuit table.
    #
    # Normally this will return a new table. However, if rv_idxs is empty,
    # then the given table is returned unmodified.
    #
    # Raises:
    #     ValueError if rv_idxs is not a subset of table.rv_idxs.
    #     ValueError if rv_idxs contains duplicates.
    cdef tuple[int, ...] rv_idxs_seq = tuple(rv_idxs)

    if len(rv_idxs_seq) == 0:
        # nothing to do
        return table

    cdef set[int] rv_idxs_set = set(rv_idxs_seq)
    if len(rv_idxs_set) != len(rv_idxs_seq):
        raise ValueError('rv_idxs contains duplicates')
    if not rv_idxs_set.issubset(table.rv_idxs):
        raise ValueError('rv_idxs is not a subset of table.rv_idxs')

    cdef int rv_index
    cdef list[int] remaining_rv_idxs = []
    for rv_index in table.rv_idxs:
        if rv_index not in rv_idxs_set:
            remaining_rv_idxs.append(rv_index)

    cdef int num_remaining = len(remaining_rv_idxs)
    if num_remaining == 0:
        # Special case: summing out all random variables
        return sum_out_all(table)

    # index_map[i] is the location in table.rv_idxs for remaining_rv_idxs[i]
    cdef list[int] index_map = []
    for rv_index in remaining_rv_idxs:
        index_map.append(_find(table.rv_idxs, rv_index))

    cdef dict[tuple[int, ...], list[object]] groups = {}
    cdef object got
    cdef list[int] group_instance
    cdef tuple[int, ...] group_instance_tuple
    cdef int i
    cdef object node
    cdef tuple[int, ...] instance
    for instance, node in table.rows.items():
        group_instance = []
        for i in index_map:
            group_instance.append(instance[i])
        group_instance_tuple = tuple(group_instance)
        got = groups.get(group_instance_tuple)
        if got is None:
            groups[group_instance_tuple] = [node]
        else:
            got.append(node)

    cdef object circuit = table.circuit
    cdef object new_table = CircuitTable(circuit, remaining_rv_idxs)
    cdef dict[tuple[int, ...], object] rows = new_table.rows

    for group_instance_tuple, to_add in groups.items():
        node = circuit.optimised_add(to_add)
        if not node.is_zero():
            rows[group_instance_tuple] = node

    return new_table


cpdef object sum_out_all(object table: CircuitTable):  # -> CircuitTable:
    # Return a circuit table that results from summing out
    # all random variables of this circuit table.
    circuit: Circuit = table.circuit
    num_rows: int = len(table)
    if num_rows == 0:
        return CircuitTable(circuit, ())
    elif num_rows == 1:
        node = next(iter(table.rows.values()))
    else:
        node: CircuitNode = circuit.optimised_add(table.rows.values())
        if node.is_zero():
            return CircuitTable(circuit, ())

    return CircuitTable(circuit, (), [((), node)])


cpdef object project(object table: CircuitTable, object rv_idxs: Iterable[int]):  # -> CircuitTable:
    # Call `sum_out(table, to_sum_out)`, where
    # `to_sum_out = table.rv_idxs - rv_idxs`.
    cdef set[int] to_sum_out = set(table.rv_idxs)
    to_sum_out.difference_update(rv_idxs)
    return sum_out(table, to_sum_out)


cpdef object product(x: CircuitTable, y: CircuitTable):  # -> CircuitTable:
    # Return a circuit table that results from the product of the two given tables.
    #
    # If x or y equals `one_table`, then the other table is returned. Otherwise,
    # a new circuit table will be constructed and returned.
    cdef int i
    cdef object circuit = x.circuit
    if y.circuit is not circuit:
        raise ValueError('circuit tables must refer to the same circuit')

    # Make the smaller table 'y', and the other 'x'.
    # This is to minimise the index size on 'y'.
    if len(x) < len(y):
        x, y = y, x

    # Special case: y == 0 or 1, and has no random variables.
    if len(y.rv_idxs) == 0:
        if len(y) == 1 and y.top().is_one():
            return x
        elif len(y) == 0:
            return CircuitTable(circuit, x.rv_idxs)

    # Set operations on rv indexes. After these operations:
    # * co_rv_idxs is the set of rv indexes common (co) to x and y,
    # * yo_rv_idxs is the set of rv indexes in y only (yo), and not in x.
    cdef set[int] yo_rv_idxs_set = set(y.rv_idxs)
    cdef set[int] co_rv_idxs_set = set(x.rv_idxs)
    co_rv_idxs_set.intersection_update(yo_rv_idxs_set)
    yo_rv_idxs_set.difference_update(co_rv_idxs_set)

    if len(co_rv_idxs_set) == 0:
        # Special case: no common random variables.
        return _product_no_common_rvs(x, y)

    # Convert random variable index sets to sequences
    cdef tuple[int, ...] yo_rv_idxs = tuple(yo_rv_idxs_set)  # y only random variables
    cdef tuple[int, ...] co_rv_idxs = tuple(co_rv_idxs_set)  # common random variables

    # Cache mappings from result Instance to index into source Instance (x or y).
    # This will be used in indexing and product loops to pull our needed values
    # from the source instances.
    cdef list[int] co_from_x_map = []
    cdef list[int] co_from_y_map = []
    cdef list[int] yo_from_y_map = []
    for rv_index in co_rv_idxs:
        co_from_x_map.append(_find(x.rv_idxs, rv_index))
        co_from_y_map.append(_find(y.rv_idxs, rv_index))
    for rv_index in yo_rv_idxs:
        yo_from_y_map.append(_find(y.rv_idxs, rv_index))

    cdef list[int] co
    cdef list[int] yo
    cdef object got
    cdef tuple[int, ...] co_tuple
    cdef tuple[int, ...] yo_tuple

    cdef object table = CircuitTable(circuit, x.rv_idxs + yo_rv_idxs)
    cdef dict[tuple[int, ...], object] rows = table.rows


    # Index the y rows by common-only key (y is the smaller of the two tables).
    cdef dict[tuple[int, ...], list[tuple[tuple[int, ...], object]]] y_index = {}
    for y_instance, y_node in y.rows.items():
        co = []
        yo = []
        for i in co_from_y_map:
            co.append(y_instance[i])
        for i in yo_from_y_map:
            yo.append(y_instance[i])
        co_tuple = tuple(co)
        yo_tuple = tuple(yo)
        got = y_index.get(co_tuple)
        if got is None:
            y_index[co_tuple] = [(yo_tuple, y_node)]
        else:
            got.append((yo_tuple, y_node))


    # Iterate over x rows, inserting (instance, value).
    # Rows with constant node values of one are optimised out.
    for x_instance, x_node in x.rows.items():
        co = []
        for i in co_from_x_map:
            co.append(x_instance[i])
        co_tuple = tuple(co)

        if x_node.is_one():
            # Multiplying by one.
            # Iterate over matching y rows.
            got = y_index.get(co_tuple)
            if got is not None:
                for yo_tuple, y_node in got:
                    rows[x_instance + yo_tuple] = y_node
        else:
            # Iterate over matching y rows.
            got = y_index.get(co_tuple)
            if got is not None:
                for yo_tuple, y_node in got:
                    rows[x_instance + yo_tuple] = _optimised_mul(circuit, x_node, y_node)

    return table


cdef int _find(tuple[int, ...] xs, int x):
    cdef int i
    for i in range(len(xs)):
        if xs[i] == x:
            return i
    # Very unexpected
    raise RuntimeError('not found')


cdef object _product_no_common_rvs(x: CircuitTable, y: CircuitTable):  # -> CircuitTable:
    # Return the product of x and y, where x and y have no common random variables.
    #
    # This is an optimisation of more general product algorithm as no index needs
    # to be construction based on the common random variables.
    #
    # Rows with constant node values of one are optimised out.
    #
    # Assumes:
    #     * There are no common random variables between x and y.
    #     * x and y are for the same circuit.
    cdef object circuit = x.circuit
    cdef object table = CircuitTable(circuit, x.rv_idxs + y.rv_idxs)
    cdef tuple[int, ...] instance

    for x_instance, x_node in x.rows.items():
        if x_node.is_one():
            for y_instance, y_node in y.rows.items():
                instance = x_instance + y_instance
                table.rows[instance] = y_node
        else:
            for y_instance, y_node in y.rows.items():
                instance = x_instance + y_instance
                table.rows[instance] = _optimised_mul(circuit, x_node, y_node)

    return table


cdef object _optimised_mul(object circuit: Circuit, object x: CircuitNode, object y: CircuitNode):  # -> CircuitNode
    if x.is_zero():
        return x
    if y.is_zero():
        return y
    if x.is_one():
        return y
    if y.is_one():
        return x
    return circuit.mul(x, y)