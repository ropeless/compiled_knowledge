cdef class Circuit:
    cdef public list[VarNode] vars
    cdef public list[OpNode] ops
    cdef public ConstNode zero
    cdef public ConstNode one
    cdef dict[object, ConstNode] _const_map
    cdef object __derivatives

    cdef OpNode op(self, int symbol, tuple[CircuitNode, ...] nodes)
    cdef list[OpNode] find_reachable_op_nodes(self, list[CircuitNode] nodes)
    cdef void _remove_unreachable_op_nodes(self, list[CircuitNode] nodes)
    cdef list[CircuitNode] _check_nodes(self, object nodes)
    cdef void __check_nodes(self, object nodes, list[CircuitNode] result)
    cdef object _derivatives(self, CircuitNode f)

cdef class CircuitNode:
    cdef public Circuit circuit
    cdef public bint is_zero
    cdef public bint is_one

cdef class ConstNode(CircuitNode):
    cdef public object value

cdef class VarNode(CircuitNode):
    cdef public int idx
    cdef object _const

    cpdef int is_const(self) except*

cdef class OpNode(CircuitNode):
    cdef public tuple[object, ...] args
    cdef public int symbol
