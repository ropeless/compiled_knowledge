from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import ctypes as ct

from ck import circuit
from ck.circuit import OpNode
from ck.circuit_compiler.support.circuit_analyser import CircuitAnalysis
from ck.program.raw_program import RawProgramFunction
from ck.utils.np_extras import NDArrayNumeric, DTypeNumeric

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cython.operator cimport dereference as deref, preincrement as incr, postdecrement

cimport numpy as cnp

cnp.import_array()

DTYPE_FLOAT64 = np.float64
ctypedef cnp.float64_t DTYPE_FLOAT64_t


ctypedef fused cvm_type:
    double
    float
    int
    long
    short

DTYPE_TO_CVM_TYPE: Dict[DTypeNumeric, str] = {
    np.float64: 'double',
    np.float32: 'float',
    np.intc: 'int',
    np.long: 'long',
    np.short: 'short',
}


def make_function(analysis: CircuitAnalysis, dtype: DTypeNumeric) -> Tuple[RawProgramFunction, int]:
    """
    Make a RawProgram function that interprets the circuit.

    Args:
        analysis: A circuit analysis object defining the function.
        dtype: a numpy data type that must be a key in the dictionary, DTYPE_TO_CVM_TYPE.

    Returns:
        (function, number_of_tmps)
    """

    cdef Instructions instructions
    np_consts: NDArrayNumeric
    instructions, np_consts = _make_instructions_from_analysis(analysis, dtype)

    ptr_type = ct.POINTER(np.ctypeslib.as_ctypes_type(dtype))
    c_np_consts = np_consts.ctypes.data_as(ptr_type)

    cvm_type_name: str = DTYPE_TO_CVM_TYPE[dtype]

    # RawProgramFunction = Callable[[ct.POINTER, ct.POINTER, ct.POINTER], None]
    if cvm_type_name == 'double':
        def function(vars_in: ct.POINTER, tmps: ct.POINTER, result: ct.POINTER) -> None:
            cdef size_t vars_in_addr = ct.cast(vars_in, ct.c_void_p).value
            cdef size_t tmps_addr = ct.cast(tmps, ct.c_void_p).value
            cdef size_t consts_addr = ct.cast(c_np_consts, ct.c_void_p).value
            cdef size_t result_addr = ct.cast(result, ct.c_void_p).value
            cvm[double](
                <double *> vars_in_addr,
                <double *> tmps_addr,
                <double *> consts_addr,
                <double *> result_addr,
                instructions,
            )
    elif cvm_type_name == 'float':
        def function(vars_in: ct.POINTER, tmps: ct.POINTER, result: ct.POINTER) -> None:
            cdef size_t vars_in_addr = ct.cast(vars_in, ct.c_void_p).value
            cdef size_t tmps_addr = ct.cast(tmps, ct.c_void_p).value
            cdef size_t consts_addr = ct.cast(c_np_consts, ct.c_void_p).value
            cdef size_t result_addr = ct.cast(result, ct.c_void_p).value
            cvm[float](
                <float *> vars_in_addr,
                <float *> tmps_addr,
                <float *> consts_addr,
                <float *> result_addr,
                instructions,
            )
    elif cvm_type_name == 'int':
        def function(vars_in: ct.POINTER, tmps: ct.POINTER, result: ct.POINTER) -> None:
            cdef size_t vars_in_addr = ct.cast(vars_in, ct.c_void_p).value
            cdef size_t tmps_addr = ct.cast(tmps, ct.c_void_p).value
            cdef size_t consts_addr = ct.cast(c_np_consts, ct.c_void_p).value
            cdef size_t result_addr = ct.cast(result, ct.c_void_p).value
            cvm[int](
                <int *> vars_in_addr,
                <int *> tmps_addr,
                <int *> consts_addr,
                <int *> result_addr,
                instructions,
            )
    elif cvm_type_name == 'long':
        def function(vars_in: ct.POINTER, tmps: ct.POINTER, result: ct.POINTER) -> None:
            cdef size_t vars_in_addr = ct.cast(vars_in, ct.c_void_p).value
            cdef size_t tmps_addr = ct.cast(tmps, ct.c_void_p).value
            cdef size_t consts_addr = ct.cast(c_np_consts, ct.c_void_p).value
            cdef size_t result_addr = ct.cast(result, ct.c_void_p).value
            cvm[long](
                <long *> vars_in_addr,
                <long *> tmps_addr,
                <long *> consts_addr,
                <long *> result_addr,
                instructions,
            )
    elif cvm_type_name == 'short':
        def function(vars_in: ct.POINTER, tmps: ct.POINTER, result: ct.POINTER) -> None:
            cdef size_t vars_in_addr = ct.cast(vars_in, ct.c_void_p).value
            cdef size_t tmps_addr = ct.cast(tmps, ct.c_void_p).value
            cdef size_t consts_addr = ct.cast(c_np_consts, ct.c_void_p).value
            cdef size_t result_addr = ct.cast(result, ct.c_void_p).value
            cvm[short](
                <short *> vars_in_addr,
                <short *> tmps_addr,
                <short *> consts_addr,
                <short *> result_addr,
                instructions,
            )
    else:
        raise ValueError(f'cvm_type_name unexpected: {cvm_type_name!r}')

    return function, len(analysis.op_to_tmp)

# VM instructions
cdef int ADD = circuit.ADD
cdef int MUL = circuit.MUL
cdef int COPY = max(ADD, MUL) + 1

# VM arrays
cdef int VARS = 0
cdef int TMPS = 1
cdef int CONSTS = 2
cdef int RESULT = 3

cdef tuple[Instructions, cnp.ndarray] _make_instructions_from_analysis(
        object analysis: CircuitAnalysis,
        object dtype: DTypeNumeric,
):
    # Store const values in a numpy array
    node_to_const_idx: Dict[int, int] = {
        id(node): i
        for i, node in enumerate(analysis.const_nodes)
    }
    np_consts: NDArrayNumeric = np.zeros(len(node_to_const_idx), dtype=dtype)
    for i, node in enumerate(analysis.const_nodes):
        np_consts[i] = node.value

    # Where to get input values for each possible node.
    cdef dict[int, ElementID] node_to_element = {}
    # const nodes
    for node_id, const_idx in node_to_const_idx.items():
        node_to_element[node_id] = ElementID(CONSTS, const_idx)
    # var nodes
    for i, var_node in enumerate(analysis.var_nodes):
        if var_node.is_const():
            node_to_element[id(var_node)] = node_to_element[id(var_node.const)]
        else:
            node_to_element[id(var_node)] = ElementID(VARS, i)
    # op nodes
    for node_id, tmp_idx in analysis.op_to_tmp.items():
        node_to_element[node_id] = ElementID(TMPS, tmp_idx)
    for node_id, result_idx in analysis.op_to_result.items():
        node_to_element[node_id] = ElementID(RESULT, result_idx)

    # Build instructions
    instructions: Instructions = Instructions()

    cdef object op_node
    for op_node in analysis.op_nodes:
        instructions.append_op(op_node.symbol, op_node, node_to_element)

    # Add any copy operations, i.e., result nodes that are not op nodes
    for i, node in enumerate(analysis.result_nodes):
        if not isinstance(node, OpNode):
            dest: ElementID = ElementID(RESULT, i)
            src: ElementID = node_to_element[id(node)]
            instructions.append_copy(src, dest)

    return instructions, np_consts

cdef struct ElementID:
    int array  # VARS, TMPS, CONSTS, RESULT
    int index  # index into the array

cdef struct Instruction:
    int             symbol  # ADD, MUL, COPY
    Py_ssize_t      num_args
    ElementID*      args
    ElementID       dest

cdef class Instructions:
    cdef Instruction* instructions
    cdef int allocated
    cdef int num_instructions

    def __init__(self) -> None:
        self.num_instructions = 0
        self.allocated = 64
        self.instructions = <Instruction *> PyMem_Malloc(self.allocated * sizeof(Instruction))

    cdef void append_copy(
            self,
            ElementID src,
            ElementID dest,
    ) except*:
        c_args = <ElementID *> PyMem_Malloc(sizeof(ElementID))
        if not c_args:
            raise MemoryError()

        c_args[0] = src
        self._append(COPY, 1, c_args, dest)

    cdef void append_op(self, int symbol, object op_node: OpNode, dict[int, ElementID] node_to_element) except*:
        args = op_node.args
        cdef Py_ssize_t num_args = len(args)

        # Create the instruction arguments array
        c_args = <ElementID *> PyMem_Malloc(num_args * sizeof(ElementID))
        if not c_args:
            raise MemoryError()

        cdef Py_ssize_t i = num_args
        while i > 0:
            i -= 1
            c_args[i] = node_to_element[id(args[i])]

        dest: ElementID = node_to_element[id(op_node)]

        self._append(symbol, num_args, c_args, dest)

    cdef void _append(self, int symbol, Py_ssize_t num_args, ElementID * c_args, ElementID dest) except*:
        cdef int i

        cdef int num_instructions = self.num_instructions

        # Ensure sufficient instruction memory
        cdef int allocated = self.allocated
        if num_instructions == allocated:
            allocated *= 2
            self.instructions = <Instruction *> PyMem_Realloc(
                self.instructions,
                allocated * sizeof(Instruction),
            )
            if not self.instructions:
                raise MemoryError()
            self.allocated = allocated

        # Add the instruction
        self.instructions[num_instructions] = Instruction(
            symbol,
            num_args,
            c_args,
            dest
        )
        self.num_instructions = num_instructions + 1

    def __dealloc__(self) -> None:
        cdef Instruction * instructions = self.instructions
        if instructions:
            for i in range(self.num_instructions):
                PyMem_Free(instructions[i].args)
            PyMem_Free(instructions)

cdef void cvm(
        cvm_type * vars_in,
        cvm_type * tmps,
        cvm_type * consts,
        cvm_type * result,
        Instructions instructions,
):
    # Core virtual machine (for fused type 'cvm_type').

    cdef int symbol
    cdef cvm_type accumulator
    cdef ElementID * args
    cdef ElementID * args_end
    cdef ElementID * dest

    # Index the four arrays by constants VARS, TMPS, CONSTS, and RESULT
    cdef (cvm_type *) arrays[4]
    arrays[VARS] = vars_in
    arrays[TMPS] = tmps
    arrays[CONSTS] = consts
    arrays[RESULT] = result

    cdef Instruction * instruction_ptr = instructions.instructions
    cdef Instruction * instruction_end = instruction_ptr + instructions.num_instructions

    while instruction_ptr < instruction_end:

        symbol = instruction_ptr.symbol
        args = instruction_ptr.args

        accumulator = arrays[args.array][args.index]

        if symbol == ADD:
            args_end = args + instruction_ptr.num_args
            incr(args)
            while args < args_end:
                accumulator += arrays[args.array][args.index]
                incr(args)
        elif symbol == MUL:
            args_end = args + instruction_ptr.num_args
            incr(args)
            while args < args_end:
                accumulator *= arrays[args.array][args.index]
                incr(args)
        # else symbol == COPY, nothing to do

        dest = &instruction_ptr.dest
        arrays[dest.array][dest.index] = accumulator

        # Advance the instruction pointer
        incr(instruction_ptr)


# This is the older 4.0.0a17 version which seems to be 10% faster sometimes!
#
# cdef void cvm(
#         cvm_type * vars_in,
#         cvm_type * tmps,
#         cvm_type * consts,
#         cvm_type * result,
#         Instructions instructions,
# ):
#     # Core virtual machine (for fused type 'cvm_type').
#
#     cdef int symbol
#     cdef Py_ssize_t i
#     cdef cvm_type accumulator
#     cdef ElementID* args
#     cdef ElementID  elem
#
#     # Index the four arrays by constants VARS, TMPS, CONSTS, and RESULT
#     cdef (cvm_type*) arrays[4]
#     arrays[VARS] = vars_in
#     arrays[TMPS] = tmps
#     arrays[CONSTS] = consts
#     arrays[RESULT] = result
#
#     cdef Instruction* instruction_ptr = instructions.instructions
#     cdef Py_ssize_t num_instructions = instructions.num_instructions
#
#     while num_instructions > 0:
#         num_instructions -= 1
#
#         symbol = instruction_ptr.symbol
#         args = instruction_ptr.args
#
#         elem = args[0]
#         accumulator = arrays[elem.array][elem.index]
#
#         if symbol == ADD:
#             i = instruction_ptr.num_args
#             while i > 1:
#                 i -= 1
#                 elem = args[i]
#                 accumulator += arrays[elem.array][elem.index]
#         elif symbol == MUL:
#             i = instruction_ptr.num_args
#             while i > 1:
#                 i -= 1
#                 elem = args[i]
#                 accumulator *= arrays[elem.array][elem.index]
#         # else symbol == COPY, nothing to do
#
#         elem = instruction_ptr.dest
#         arrays[elem.array][elem.index] = accumulator
#
#         # Advance the instruction pointer
#         instruction_ptr = &(instruction_ptr[1])
