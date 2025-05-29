# There are two implementations of the `circuit` module are provided
# for developer R&D purposes. One is pure Python and the other is Cython.
# Which implementation is used can be selected here.
# A similar selection can be made for the `circuit_table` module.
# Note that if the Cython implementation is chosen for `circuit_table` then
# the Cython implementation must be chosen for `circuit`.

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
