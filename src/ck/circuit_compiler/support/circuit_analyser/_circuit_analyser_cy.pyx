from dataclasses import dataclass
from typing import List, Dict, Sequence, Set

from ck.circuit._circuit_cy cimport Circuit, OpNode, VarNode, CircuitNode, ConstNode
from cython.operator cimport postincrement


@dataclass
class CircuitAnalysis:
    """
    A data structure representing the analysis of a function defined by
    a circuit which chosen input variables and output result nodes.
    """

    var_nodes: Sequence[VarNode]  # specified input var nodes
    result_nodes: Sequence[CircuitNode]  # specified result nodes
    op_nodes: Sequence[OpNode]  # in-use op nodes, in computation order
    const_nodes: Sequence[ConstNode]  # in_use const nodes, in arbitrary order
    op_to_result: Dict[int, int]  # op nodes in the result, op_node = result[idx]: id(op_node) -> idx
    op_to_tmp: Dict[int, int]  # op nodes needing tmp memory, using tmp[idx]: id(op_node) -> idx


def analyze_circuit(
        var_nodes: Sequence[VarNode],
        result_nodes: Sequence[CircuitNode],
) -> CircuitAnalysis:
    """
    Analyzes a circuit as a function from var_nodes to result_nodes,
    returning a CircuitAnalysis object.

    Args:
        var_nodes: The chosen input variable nodes of the circuit.
        result_nodes: The chosen output result nodes of the circuit.

    Returns:
        A CircuitAnalysis object.
    """
    cdef list[CircuitNode] results_list = list(result_nodes)

    # What op nodes are in use
    cdef list[OpNode] op_nodes = _reachable_op_nodes(results_list)

    # What constant values are in use
    cdef set[int] seen_const_nodes = set()
    cdef list[ConstNode] const_nodes  = []

    def _register_const(_node: ConstNode) -> None:
        nonlocal seen_const_nodes
        nonlocal const_nodes
        _node_id: int = id(_node)
        if _node_id not in seen_const_nodes:
            const_nodes.append(_node)
            seen_const_nodes.add(_node_id)

    # Register all the used constants
    for op_node in op_nodes:
        for node in op_node.args:
            if isinstance(node, ConstNode):
                _register_const(node)
    for node in results_list:
        if isinstance(node, ConstNode):
            _register_const(node)
    for node in var_nodes:
        if node.is_const():
            _register_const(node.const)

    # What op nodes are in the result.
    # Dict op_to_result maps id(OpNode) to result index.
    cdef dict[int, int] op_to_result = {
        id(node): i
        for i, node in enumerate(result_nodes)
        if isinstance(node, OpNode)
    }

    # Assign all other op nodes to a tmp slot.
    # Dict op_to_tmp maps id(OpNode) to tmp index.
    cdef int tmp_idx = 0
    op_to_tmp: Dict[int, int] = {
        id(op_node): postincrement(tmp_idx)
        for op_node in op_nodes
        if id(op_node) not in op_to_result
    }

    return CircuitAnalysis(
        var_nodes=var_nodes,
        result_nodes=result_nodes,
        op_nodes=op_nodes,
        const_nodes=const_nodes,
        op_to_result=op_to_result,
        op_to_tmp=op_to_tmp,
    )


cdef list[OpNode] _reachable_op_nodes(list[CircuitNode] results):
    if len(results) == 0:
        return []
    cdef Circuit circuit = results[0].circuit
    return circuit.find_reachable_op_nodes(results)
