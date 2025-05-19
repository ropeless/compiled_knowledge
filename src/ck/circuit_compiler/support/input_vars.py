"""
This module supports circuit compilers and interpreters by inferring and checking input variables
that are explicitly or implicitly referred to by a client.
"""

from enum import Enum
from itertools import chain
from typing import Sequence, Optional, Set, Iterable, List

from ck.circuit import VarNode, Circuit, CircuitNode, OpNode


class InferVars(Enum):
    """
    An enum specifying how to automatically infer a program's input variables.
    """

    ALL = 'all'  # all circuit vars are input vars
    REF = 'ref'  # only referenced vars are input vars
    LOW = 'low'  # input vars are circuit vars[0 : max_referenced + 1]


# Type for specifying input circuit vars
InputVars = InferVars | Sequence[VarNode] | VarNode


def infer_input_vars(
        circuit: Optional[Circuit],
        results: Sequence[CircuitNode],
        input_vars: InputVars,
) -> Sequence[VarNode]:
    """
    Infer what circuit is being referred to, based on Program constructor arguments.
    Infer what input variable are being referred to, based on Program constructor arguments.
    Check that all input vars and results nodes are in the circuit.

    Returns:
        The inferred input circuit vars.

    Raises:
        ValueError: if the circuit is unknown, but it is needed.
        ValueError: if not all nodes are from the same circuit.

    Ensures:
        circuit is None implies len(input_vars) == 0
    """
    cct: Optional[Circuit] = _infer_circuit(circuit, results, input_vars)
    input_vars: Sequence[VarNode] = _infer_input(cct, results, input_vars)

    # Check that all results nodes and input vars are in the circuit.
    if cct is not None:
        for n in chain(results, input_vars):
            if n.circuit is not cct:
                raise ValueError('a var node or result node is not in the inferred circuit')

    return input_vars


def _infer_circuit(
        cct: Optional[Circuit],
        results: Sequence[CircuitNode],
        input_vars: InputVars,
) -> Optional[Circuit]:
    """
    Infer what circuit is being referred to, based on Program constructor arguments.
    """
    if cct is not None:
        return cct
    if len(results) > 0:
        return results[0].circuit
    if isinstance(input_vars, CircuitNode):
        return input_vars.circuit
    if not isinstance(input_vars, InferVars):
        # input vars is a sequence of CircuitNode
        for input_var in input_vars:
            return input_var.circuit

    return None


def _infer_input(
        cct: Optional[Circuit],
        results: Sequence[CircuitNode],
        input_vars: InputVars,
) -> Sequence[VarNode]:
    """
    Infer what input variable are being referred to, based on Program constructor arguments.
    """

    have_results: bool = len(results) > 0

    if input_vars == InferVars.ALL:
        if have_results:
            return cct.vars
        else:
            return ()

    elif input_vars == InferVars.LOW:
        if have_results:
            to_index: int = max((var.idx for var in _find_vars(results)), default=-1) + 1
            return cct.vars[:to_index]
        else:
            return ()

    elif input_vars == InferVars.REF:
        return tuple(sorted(_find_vars(results)))

    elif isinstance(input_vars, VarNode):
        input_vars = (input_vars,)

    # Assume input_vars is a Sequence[VarNode]

    in_vars: Sequence[VarNode] = tuple(input_vars)

    # check no duplicate in_vars
    input_var_indices: Set[int] = {var.idx for var in in_vars}
    if len(input_var_indices) != len(in_vars):
        raise ValueError('cannot have duplicate circuit variables as inputs')

    # ensure that the input vars cover what is needed.
    needed_var_indices: Set[int] = {var.idx for var in _find_vars(results)}
    if not input_var_indices.issuperset(needed_var_indices):
        raise ValueError('input var nodes does not cover all need var nodes for result')

    return in_vars


def _find_vars(nodes: Iterable[CircuitNode]) -> List[VarNode]:
    """
    Get the set of all VarNode nodes that are not set constant, reachable from the given nodes.
    """
    seen: Set[int] = set()
    var_nodes: List[VarNode] = []
    __find_vars_r(nodes, seen, var_nodes)
    return var_nodes


def __find_vars_r(nodes: Iterable[CircuitNode], seen: Set[int], var_nodes: List[VarNode]) -> None:
    """
    Recursive support for _find_vars.
    """
    for node in nodes:
        if id(node) not in seen:
            seen.add(id(node))
            if isinstance(node, VarNode) and not node.is_const():
                var_nodes.append(node)
            elif isinstance(node, OpNode):
                __find_vars_r(node.args, seen, var_nodes)
