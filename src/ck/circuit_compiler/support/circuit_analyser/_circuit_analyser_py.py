from dataclasses import dataclass
from itertools import count
from typing import List, Dict, Sequence, Set

from ck.circuit import OpNode, VarNode, CircuitNode, ConstNode


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
    # What op nodes are in use
    op_nodes: List[OpNode] = (
        [] if len(result_nodes) == 0
        else result_nodes[0].circuit.reachable_op_nodes(*result_nodes)
    )

    # What constant values are in use
    seen_const_nodes: Set[int] = set()
    const_nodes: List[ConstNode] = []

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
    for node in result_nodes:
        if isinstance(node, ConstNode):
            _register_const(node)
    for node in var_nodes:
        if node.is_const():
            _register_const(node.const)

    # What op nodes are in the result.
    # Dict op_to_result maps id(OpNode) to result index.
    op_to_result: Dict[int, int] = {
        id(node): i
        for i, node in enumerate(result_nodes)
        if isinstance(node, OpNode)
    }

    # Assign all other op nodes to a tmp slot.
    # Dict op_to_tmp maps id(OpNode) to tmp index.
    tmp_idx = count()
    op_to_tmp: Dict[int, int] = {
        id(op_node): next(tmp_idx)
        for op_node in op_nodes
        if id(op_node) not in op_to_result
    }
    del tmp_idx

    return CircuitAnalysis(
        var_nodes=var_nodes,
        result_nodes=result_nodes,
        op_nodes=op_nodes,
        const_nodes=const_nodes,
        op_to_result=op_to_result,
        op_to_tmp=op_to_tmp,
    )
