# There are two implementations of the `circuit_table` module are provided
# for developer R&D purposes. One is pure Python and the other is Cython.
# Which implementation is used can be selected here.
#
# A similar selection can be made for the `circuit` module.
# Note that if the Cython implementation is chosen for `circuit_table` then
# the Cython implementation must be chosen for `circuit`.

# from ._circuit_table_py import (
from ._circuit_table_cy import (
    CircuitTable,
    TableInstance,
    sum_out,
    sum_out_all,
    project,
    product,
)
