from __future__ import annotations

from typing import Iterable, Optional

from ck.circuit import Circuit, VarNode, ConstValue, ConstNode


class TmpConst:
    """
    A TmpConst enables the consistent, temporary setting and clearing
    of circuit variables to constant values.

    Example usage:
        with TmpConst(my_circuit) as tmp:
            tmp.set_const(0, 123.456)
            tmp.set_const(my_list_of_vars, 110.99)
            program = Program(my_top)
        # now use 'program'

    Within the 'with' section, circuit variables are set to const values, which
    the compiler can optimise. When the 'with' section exits, the variables
    are restored to their original state.
    """
    __slots__ = ('_circuit', '_undo')

    def __init__(self, circuit: Circuit):
        self._circuit = circuit
        self._undo = []

    def set_const(self, var: VarNode | int | Iterable[VarNode | int], value: ConstValue | ConstNode | None) -> None:
        """
        Set the given circuit variable to the given value.

        Args:
            var: can either be a VarNode, or an index to a var node, or multiple thereof.
            value: the temporary constant value, or None to clear a constant value.
        """
        if isinstance(var, int):
            var: VarNode = self._circuit.vars[var]
            self._append_undo(var)
            var.const = value
        elif isinstance(var, VarNode):
            self._append_undo(var)
            var.const = value
        else:
            # Assume it's an iterable
            for v in var:
                self.set_const(v, value)

    def undo(self) -> None:
        """
        Undo any changes to the circuit variables made with this TmpConst.
        """
        while len(self._undo) > 0:
            self._undo.pop()()

    def _append_undo(self, var: VarNode) -> None:
        """
        Push an undo function for var onto the self._undo stack.
        """
        prev_value: Optional[ConstNode] = var.const

        def undo():
            var.const = prev_value

        self._undo.append(undo)

    def __enter__(self) -> TmpConst:
        # nothing to do - all work done by __init__
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.undo()
        return exc_val is None
