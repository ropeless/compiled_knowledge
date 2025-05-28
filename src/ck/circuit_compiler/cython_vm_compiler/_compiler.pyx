from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import ctypes as ct

from ck import circuit
from ck.circuit import CircuitNode, ConstNode, VarNode, OpNode, Circuit
from ck.circuit_compiler.support.circuit_analyser import CircuitAnalysis, analyze_circuit
from ck.circuit_compiler.support.input_vars import infer_input_vars, InputVars
from ck.program.raw_program import RawProgram, RawProgramFunction
from ck.utils.np_extras import DType, NDArrayNumeric, NDArray, DTypeNumeric

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

cimport numpy as cnp
cimport cython

cnp.import_array()

DTYPE_FLOAT64 = np.float64
ctypedef cnp.float64_t DTYPE_FLOAT64_t



def make_function(
        var_nodes: Sequence[VarNode],
        result_nodes: Sequence[CircuitNode],
        dtype: DTypeNumeric,
) -> Tuple[RawProgramFunction, int]:
    """
    Make a RawProgram function that interprets the circuit.

    Returns:
        (function, number_of_tmps)
    """

    analysis: CircuitAnalysis = analyze_circuit(var_nodes, result_nodes)
    cdef Instructions instructions
    np_consts: NDArrayNumeric
    instructions, np_consts = _make_instructions_from_analysis(analysis, dtype)

    ptr_type = ct.POINTER(np.ctypeslib.as_ctypes_type(dtype))
    c_np_consts = np_consts.ctypes.data_as(ptr_type)

    # RawProgramFunction = Callable[[ct.POINTER, ct.POINTER, ct.POINTER], None]
    def function(vars_in: ct.POINTER, tmps: ct.POINTER, result: ct.POINTER) -> None:
        cdef size_t vars_in_addr = ct.cast(vars_in, ct.c_void_p).value
        cdef size_t tmps_addr = ct.cast(tmps, ct.c_void_p).value
        cdef size_t consts_addr = ct.cast(c_np_consts, ct.c_void_p).value
        cdef size_t result_addr = ct.cast(result, ct.c_void_p).value
        cvm_float64(
            <double*> vars_in_addr,
            <double*> tmps_addr,
            <double*> consts_addr,
            <double*> result_addr,
            instructions,
        )

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


def _make_instructions_from_analysis(
        analysis: CircuitAnalysis,
        dtype: DTypeNumeric,
) -> Tuple[Instructions, NDArrayNumeric]:
    if dtype != np.float64:
        raise RuntimeError(f'only DType {np.float64} currently supported')

    # Store const values in a numpy array
    node_to_const_idx: Dict[int, int] = {
        id(node): i
        for i, node in enumerate(analysis.const_nodes)
    }
    np_consts: NDArrayNumeric = np.zeros(len(node_to_const_idx), dtype=dtype)
    for i, node in enumerate(analysis.const_nodes):
        np_consts[i] = node.value

    # Where to get input values for each possible node.
    node_to_element: Dict[int, ElementID] = {}
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

    op_node: OpNode
    for op_node in analysis.op_nodes:
        dest: ElementID = node_to_element[id(op_node)]
        args: list[ElementID] = [
            node_to_element[id(arg)]
            for arg in op_node.args
        ]
        instructions.append(op_node.symbol, args, dest)

    # Add any copy operations, i.e., result nodes that are not op nodes
    for i, node in enumerate(analysis.result_nodes):
        if not isinstance(node, OpNode):
            dest: ElementID = ElementID(RESULT, i)
            args: list[ElementID] = [node_to_element[id(node)]]
            instructions.append(COPY, args, dest)

    return instructions, np_consts


cdef struct ElementID:
    int array  # VARS, TMPS, CONSTS, RESULT
    int index  # index into the array


cdef struct Instruction:
    int             symbol  # ADD, MUL, COPY
    int             num_args
    ElementID*      args
    ElementID       dest


cdef class Instructions:
    cdef Instruction* instructions
    cdef int allocated
    cdef int num_instructions

    def __init__(self) -> None:
        self.num_instructions = 0
        self.allocated = 64
        self.instructions = <Instruction*> PyMem_Malloc(self.allocated * sizeof(Instruction))

    cdef void append(self, int symbol, list[ElementID] args, ElementID dest):
        cdef int num_args = len(args)
        cdef int i

        # Create the instruction arguments array
        c_args = <ElementID*> PyMem_Malloc(num_args * sizeof(ElementID))
        if not c_args:
            raise MemoryError()
        for i in range(num_args):
            c_args[i] = args[i]

        cdef int num_instructions = self.num_instructions

        # Ensure sufficient instruction memory
        cdef int allocated = self.allocated
        if num_instructions == allocated:
            allocated *= 2
            self.instructions = <Instruction*> PyMem_Realloc(
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
        cdef Instruction* instructions = self.instructions
        if instructions:
            for i in range(self.num_instructions):
                PyMem_Free(instructions[i].args)
            PyMem_Free(instructions)



@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void cvm_float64(
    double* vars_in,
    double* tmps,
    double* consts,
    double* result,
    Instructions instructions,
) except *:
    # Core virtual machine (for dtype float64).

    cdef int i, symbol
    cdef double accumulator
    cdef ElementID* args
    cdef ElementID  elem

    # Index the four arrays by constants VARS, TMPS, CONSTS, and RESULT
    cdef (double*) arrays[4]
    arrays[VARS] = vars_in
    arrays[TMPS] = tmps
    arrays[CONSTS] = consts
    arrays[RESULT] = result

    cdef Instruction* instruction_ptr = instructions.instructions
    cdef int num_instructions = instructions.num_instructions

    while num_instructions > 0:
        num_instructions -= 1

        symbol = instruction_ptr.symbol
        args = instruction_ptr.args

        elem = args[0]
        accumulator = arrays[elem.array][elem.index]

        if symbol == ADD:
            i = instruction_ptr.num_args
            while i > 1:
                i -= 1
                elem = args[i]
                accumulator += arrays[elem.array][elem.index]
        elif symbol == MUL:
            i = instruction_ptr.num_args
            while i > 1:
                i -= 1
                elem = args[i]
                accumulator *= arrays[elem.array][elem.index]
        # else symbol == COPY, nothing to do

        elem = instruction_ptr.dest
        arrays[elem.array][elem.index] = accumulator

        # Advance the instruction pointer
        instruction_ptr = &(instruction_ptr[1])
