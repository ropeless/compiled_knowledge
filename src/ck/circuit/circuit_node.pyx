from __future__ import annotations

from typing import Optional, Tuple

# Python Type for values of ConstNode objects
ConstValue = float | int | bool

cdef class CircuitNode:
    """
    A node in an arithmetic circuit.
    Each node is either an op, var, or const node.

    Each op node is either a mul, add or sub node. Each op
    node has zero or more arguments. Each argument is another node.

    Every var node has an index, `idx`, which is an integer counting from zero, and denotes
    its creation order.

    A var node may be temporarily set to be a constant node, which may
    be useful for optimising a compiled circuit.
    """
    cdef public object circuit

    def __init__(self, circuit):
        self.circuit = circuit

    cpdef int is_zero(self) except*:
        return False

    cpdef int is_one(self) except*:
        return False

    def __add__(self, other: CircuitNode | ConstValue):
        return self.circuit.add(self, other)

    def __mul__(self, other: CircuitNode | ConstValue):
        return self.circuit.mul(self, other)


cdef class ConstNode(CircuitNode):
    cdef public object value

    """
    A node in a circuit representing a constant value.
    """
    def __init__(self, circuit, value: ConstValue):
        super().__init__(circuit)
        self.value: ConstValue = value

    cpdef int is_zero(self) except*:
        # noinspection PyProtectedMember
        return self is self.circuit.zero

    cpdef int is_one(self) except*:
        # noinspection PyProtectedMember
        return self is self.circuit.one

    def __str__(self) -> str:
        return 'const(' + str(self.value) + ')'

    def __lt__(self, other) -> bool:
        if isinstance(other, ConstNode):
            return self.value < other.value
        else:
            return False

cdef class VarNode(CircuitNode):
    """
    A node in a circuit representing an input variable.
    """
    cdef public int idx
    cdef object _const

    def __init__(self, circuit, idx: int):
        super().__init__(circuit)
        self.idx = idx
        self._const = None

    cpdef int is_zero(self) except*:
        return self._const is not None and self._const.is_zero()

    cpdef int is_one(self) except*:
        return self._const is not None and self._const.is_one()

    cpdef int is_const(self) except*:
        return self._const is not None

    @property
    def const(self) -> Optional[ConstNode]:
        return self._const

    @const.setter
    def const(self, value: ConstValue | ConstNode | None) -> None:
        if value is None:
            self._const = None
        else:
            self._const = self.circuit.const(value)

    def __lt__(self, other) -> bool:
        if isinstance(other, VarNode):
            return self.idx < other.idx
        else:
            return False

    def __str__(self) -> str:
        if self._const is None:
            return 'var[' + str(self.idx) + ']'
        else:
            return 'var[' + str(self.idx) + '] = ' + str(self._const.value)

cdef class OpNode(CircuitNode):
    """
    A node in a circuit representing an arithmetic operation.
    """
    cdef public tuple[object, ...] args
    cdef public str symbol

    def __init__(self, object circuit, symbol: str, tuple[object, ...] args: Tuple[CircuitNode]):
        super().__init__(circuit)
        self.args = tuple(args)
        self.symbol = str(symbol)

    def __str__(self) -> str:
        return self.symbol + '\\' + str(len(self.args))

cdef class MulNode(OpNode):
    """
    A node in a circuit representing a multiplication operation.
    """
    def __init__(self, object circuit, tuple[object, ...] args: Tuple[CircuitNode, ...]):
        super().__init__(circuit, 'mul', args)

cdef class AddNode(OpNode):
    """
    A node in a circuit representing an addition operation.
    """
    def __init__(self, circuit, tuple[object, ...] args: Tuple[CircuitNode, ...]):
        super().__init__(circuit, 'add', args)
