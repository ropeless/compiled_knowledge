from typing import Protocol, Optional

from ck.circuit import Circuit, CircuitNode
from ck.circuit_compiler.support.input_vars import InputVars, InferVars
from ck.program import RawProgram


class CircuitCompiler(Protocol):
    def __call__(
            self,
            *result: CircuitNode,
            input_vars: InputVars = InferVars.ALL,
            circuit: Optional[Circuit] = None,
    ) -> RawProgram:
        """
        A PGM compiler compiles selected results nodes of an arithmetic circuit to a program.

        Args:
            *result: one or more circuit of nodes defining the result of the program function.
                All result node must be from the same circuit.
            input_vars: how to determine the input variables. Either a sequence of VarNodes, or a single
                VarNode, or a `InferVars` member. The default is to use all circuit variables, in index order.
            circuit: optionally explicitly specify the Circuit (mandatory if no result nodes are provided).

        Returns:
            a RawProgram which implements the arithmetic circuit function.
        """
