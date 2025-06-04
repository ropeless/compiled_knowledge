# There are two implementations of the `circuit` module are provided
# for developer R&D purposes. One is pure Python and the other is Cython.
# Which implementation is used can be selected here.

# from ._circuit_py import (
from ._circuit_cy import (
    Circuit,
    CircuitNode,
    VarNode,
    ConstNode,
    OpNode,
    Args,
    ConstValue,
    MUL,
    ADD,
)
from .tmp_const import TmpConst
