from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional, Tuple, List, Dict

import llvmlite.binding as llvm
import llvmlite.ir as ir
import numpy as np
import ctypes as ct

from .support.circuit_analyser import CircuitAnalysis, analyze_circuit
from .support.input_vars import InputVars, InferVars, infer_input_vars
from .support.llvm_ir_function import IRFunction, DataType, TypeInfo, compile_llvm_program, LLVMRawProgram
from ..circuit import ADD as _ADD, MUL as _MUL, ConstValue
from ..circuit import Circuit, VarNode, CircuitNode, OpNode
from ..program.raw_program import RawProgramFunction

DEFAULT_TYPE_INFO: TypeInfo = DataType.FLOAT_64.value

# Byte code operations
# _ADD: int = circuit.ADD
# _MUL: int = circuit.MUL
_END: int = max(_ADD, _MUL) + 1

# arrays
_VARS: int = 0
_TMPS: int = 1
_RESULT: int = 2
_CONSTS: int = 3

_SET_CONSTS_FUNCTION_NAME: str = 'set_consts'
_SET_INSTRUCTIONS_FUNCTION_NAME: str = 'set_instructions'


def compile_circuit(
        *result: CircuitNode,
        input_vars: InputVars = InferVars.ALL,
        circuit: Optional[Circuit] = None,
        data_type: DataType | TypeInfo = DEFAULT_TYPE_INFO,
        keep_llvm_program: bool = True,
        compile_arrays: bool = False,
        opt: int = 2,
) -> LLVMRawProgram:
    """
    Compile the given circuit using LLVM.

    This creates an LLVM program where each circuit op node is converted to
    one or more LLVM binary op machine code instructions. For large circuits
    this results in a large LLVM program which can be slow to compile.

    This compiler produces a RawProgram that _does_ use client managed working memory.

    Conforms to the CircuitCompiler protocol.

    Args:
        *result: result nodes nominating the results of the returned program.
        input_vars: How to determine the input variables.
        circuit: optionally explicitly specify the Circuit.
        data_type: What data type to use for arithmetic calculations. Either a DataType member or TypeInfo.
        keep_llvm_program: if true, the LLVM program will be kept. This is required for picking.
        compile_arrays: if true, the global array values are included in the LLVM program.
        opt:The optimization level to use by LLVM MC JIT.

    Returns:
        a raw program.

    Raises:
        ValueError: if the circuit is unknown, but it is needed.
        ValueError: if not all nodes are from the same circuit.
        ValueError: if the program data type could not be interpreted.
    """
    in_vars: Sequence[VarNode] = infer_input_vars(circuit, result, input_vars)
    var_indices: Sequence[int] = tuple(var.idx for var in in_vars)

    # Get the type info
    type_info: TypeInfo
    if isinstance(data_type, DataType):
        type_info = data_type.value
    elif isinstance(data_type, TypeInfo):
        type_info = data_type
    else:
        raise ValueError(f'could not interpret program data type: {data_type!r}')

    # Compile the circuit to an LLVM module representing a RawProgramFunction
    llvm_program: str
    number_of_tmps: int
    llvm_program, number_of_tmps, consts, byte_code = _make_llvm_program(in_vars, result, type_info, compile_arrays)

    # Compile the LLVM program to a native executable
    engine: llvm.ExecutionEngine
    function: RawProgramFunction
    engine, function = compile_llvm_program(llvm_program, dtype=type_info.dtype, opt=opt)

    if compile_arrays:
        return LLVMRawProgram(
            function=function,
            dtype=type_info.dtype,
            number_of_vars=len(var_indices),
            number_of_tmps=number_of_tmps,
            number_of_results=len(result),
            var_indices=var_indices,
            llvm_program=llvm_program if keep_llvm_program else None,
            engine=engine,
            opt=opt,
        )
    else:
        # Arrays `consts` and `byte_code` are not compiled into the LLVM program
        # so they need to be stored explicitly.
        return LLVMRawProgramWithArrays(
            function=function,
            dtype=type_info.dtype,
            number_of_vars=len(var_indices),
            number_of_tmps=number_of_tmps,
            number_of_results=len(result),
            var_indices=var_indices,
            llvm_program=llvm_program if keep_llvm_program else None,
            engine=engine,
            opt=opt,
            instructions=np.array(byte_code, dtype=np.uint8),
            consts=np.array(consts, dtype=type_info.dtype),
        )


@dataclass
class LLVMRawProgramWithArrays(LLVMRawProgram):
    instructions: np.ndarray
    consts: np.ndarray

    def __post_init__(self):
        self._set_globals(self.instructions, _SET_INSTRUCTIONS_FUNCTION_NAME)
        self._set_globals(self.consts, _SET_CONSTS_FUNCTION_NAME)

    def __getstate__(self):
        state = super().__getstate__()
        state['instructions'] = self.instructions
        state['consts'] = self.consts
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.instructions = state['instructions']
        self.consts = state['consts']
        self._set_globals(self.instructions, _SET_INSTRUCTIONS_FUNCTION_NAME)
        self._set_globals(self.consts, _SET_CONSTS_FUNCTION_NAME)

    def _set_globals(self, data: np.ndarray, func_name: str) -> None:
        ptr_type = ct.POINTER(np.ctypeslib.as_ctypes_type(data.dtype))
        c_np_data = data.ctypes.data_as(ptr_type)

        function_ptr = self.engine.get_function_address(func_name)
        function = ct.CFUNCTYPE(None, ptr_type)(function_ptr)
        function(c_np_data)


def _make_llvm_program(
        in_vars: Sequence[VarNode],
        result: Sequence[CircuitNode],
        type_info: TypeInfo,
        compile_arrays: bool,
) -> Tuple[str, int, List[ConstValue], List[int]]:
    """
    Construct the LLVM program (i.e., LLVM module).

    Returns:
        (llvm_program, number_of_tmps, consts, byte_code)
    """
    llvm_function = IRFunction(type_info)

    builder = llvm_function.builder
    type_info = llvm_function.type_info
    module = llvm_function.module

    analysis: CircuitAnalysis = analyze_circuit(in_vars, result)
    const_values: List[ConstValue] = [const_node.value for const_node in analysis.const_nodes]

    max_index_size: int = max(
        len(analysis.var_nodes),  # number of inputs
        len(analysis.result_nodes),  # number of outputs
        len(analysis.op_to_tmp),  # number of tmps
        len(analysis.const_nodes),  # number of constants
    )
    data_idx_bytes: int = _get_bytes_needed(max_index_size)

    max_num_args: int = max((len(op_node.args) for op_node in analysis.op_nodes), default=0)
    num_args_bytes: int = _get_bytes_needed(max_num_args)

    data_type: ir.Type = type_info.llvm_type
    byte_type: ir.Type = ir.IntType(8)
    data_idx_type: ir.Type = ir.IntType(data_idx_bytes * 8)

    byte_code: List[int] = _make_byte_code(analysis, data_idx_bytes, num_args_bytes)

    inst_idx_bytes: int = _get_bytes_needed(len(byte_code))
    inst_idx_type: ir.Type = ir.IntType(inst_idx_bytes * 8)

    if compile_arrays:
        # Put constants into the LLVM module
        consts_array_type = ir.ArrayType(data_type, len(analysis.const_nodes))
        consts_global = ir.GlobalVariable(module, consts_array_type, name='consts')
        consts_global.global_constant = True
        consts_global.initializer = ir.Constant(consts_array_type, const_values)
        data_idx_0 = ir.Constant(data_idx_type, 0)
        consts: ir.Value = builder.gep(consts_global, [data_idx_0,  data_idx_0])

        # Put bytecode into the LLVM module
        instructions_array_type = ir.ArrayType(byte_type, len(byte_code))
        instructions_global = ir.GlobalVariable(module, instructions_array_type, name='instructions')
        instructions_global.global_constant = True
        instructions_global.initializer = ir.Constant(instructions_array_type, byte_code)
        inst_idx_0 = ir.Constant(inst_idx_type, 0)
        instructions: ir.Value = builder.gep(instructions_global, [inst_idx_0, inst_idx_0])
    else:
        # Just create two global variables that will be set externally.
        const_ptr_type = data_type.as_pointer()
        consts_global = ir.GlobalVariable(module, const_ptr_type, name='consts')
        consts_global.initializer = ir.Constant(const_ptr_type, None)
        consts: ir.Value = builder.load(consts_global)

        instructions_ptr_type = byte_type.as_pointer()
        instructions_global = ir.GlobalVariable(module, instructions_ptr_type, name='instructions')
        instructions_global.initializer =ir.Constant(instructions_ptr_type, None)
        instructions: ir.Value = builder.load(instructions_global)

    interp = _InterpBuilder(builder, type_info, inst_idx_type, data_idx_bytes, num_args_bytes, consts, instructions)
    interp.make_interpreter()

    if not compile_arrays:
        # add functions to set global arrays
        interp.make_set_consts_function(consts_global)
        interp.make_set_instructions_function(instructions_global)

    # print(llvm_function.llvm_program())
    # exit(99)

    return llvm_function.llvm_program(), len(analysis.op_to_tmp), const_values, byte_code


class _InterpBuilder:
    """
    Helper to write the LLVM function for the byte code interpreter.
    """

    def __init__(
            self,
            builder: ir.IRBuilder,
            type_info: TypeInfo,
            inst_idx_type: ir.Type,
            index_bytes: int,
            num_args_bytes: int,
            consts: ir.Value,
            instructions: ir.Value,
    ):
        self.builder: ir.IRBuilder = builder
        self.index_bytes: int = index_bytes
        self.num_args_bytes: int = num_args_bytes
        self.type_info: TypeInfo = type_info

        self.data_type: ir.Type = type_info.llvm_type
        self.byte_type: ir.Type = ir.IntType(8)
        self.inst_idx_type: ir.Type = inst_idx_type
        self.data_idx_type: ir.Type = ir.IntType(index_bytes * 8)
        self.num_args_type: ir.Type = ir.IntType(num_args_bytes * 8)

        self.data_idx_0 = ir.Constant(self.data_idx_type, 0)
        self.data_idx_1 = ir.Constant(self.data_idx_type, 1)
        self.inst_idx_0 = ir.Constant(self.inst_idx_type, 0)
        self.inst_idx_1 = ir.Constant(self.inst_idx_type, 1)
        self.num_args_0 = ir.Constant(self.num_args_type, 0)
        self.num_args_1 = ir.Constant(self.num_args_type, 1)

        self.consts: ir.Value = consts
        self.instructions: ir.Value = instructions

        # allocate locals
        self.local_idx = builder.alloca(self.inst_idx_type, name='idx')
        self.local_num_args = builder.alloca(self.num_args_type, name='num_args')
        self.local_accumulator = builder.alloca(self.data_type, name='accumulator')
        self.local_arrays = builder.alloca(self.data_type.as_pointer(), size=4, name='arrays')

        # local_arrays = [vars, tmps, result, consts]
        ir_vars_idx = ir.Constant(self.byte_type, _VARS)
        ir_tmps_idx = ir.Constant(self.byte_type, _TMPS)
        ir_result_idx = ir.Constant(self.byte_type, _RESULT)
        ir_consts_idx = ir.Constant(self.byte_type, _CONSTS)
        function: ir.Function = builder.function
        local_arrays = self.local_arrays
        builder.store(function.args[0], builder.gep(local_arrays, [ir_vars_idx]))
        builder.store(function.args[1], builder.gep(local_arrays, [ir_tmps_idx]))
        builder.store(function.args[2], builder.gep(local_arrays, [ir_result_idx]))
        builder.store(consts, builder.gep(local_arrays, [ir_consts_idx]))

        # local_idx = 0
        builder.store(self.inst_idx_0, self.local_idx)

    def make_set_consts_function(self, consts_ptr: ir.GlobalVariable):
        builder = self.builder
        module = builder.module
        function_type = ir.FunctionType(ir.VoidType(), (self.data_type.as_pointer(),))
        function = ir.Function(module, function_type, name=_SET_CONSTS_FUNCTION_NAME)
        bb_entry = function.append_basic_block('entry')
        builder.position_at_end(bb_entry)
        arg = function.args[0]
        builder.store(arg, consts_ptr)
        builder.ret_void()

    def make_set_instructions_function(self, instructions_ptr: ir.GlobalVariable):
        builder = self.builder
        module = builder.module
        function_type = ir.FunctionType(ir.VoidType(), (self.byte_type.as_pointer(),))
        function = ir.Function(module, function_type, name=_SET_INSTRUCTIONS_FUNCTION_NAME)
        bb_entry = function.append_basic_block('entry')
        builder.position_at_end(bb_entry)
        arg = function.args[0]
        builder.store(arg, instructions_ptr)
        builder.ret_void()

    def add(self, x: ir.Value, y: ir.Value) -> ir.Value:
        return self.type_info.add(self.builder, x, y)

    def mul(self, x: ir.Value, y: ir.Value) -> ir.Value:
        return self.type_info.mul(self.builder, x, y)

    def make_interpreter(self):
        """
        Write the bytecode interpreter
        """
        builder: ir.IRBuilder = self.builder
        function: ir.Function = builder.function

        bb_while = function.append_basic_block('while')
        bb_body = function.append_basic_block('body')
        bb_mul = function.append_basic_block('mul')
        bb_mul_op = function.append_basic_block('mul_op')
        bb_add = function.append_basic_block('add')
        bb_add_op = function.append_basic_block('add_op')
        bb_op_continue = function.append_basic_block('op_continue')
        bb_finish = function.append_basic_block('finish')

        # block: entry
        # (locals already set up in the constructor)
        builder.branch(bb_while)

        # block: while
        builder.position_at_end(bb_while)
        # load current instruction
        idx = builder.load(self.local_idx)
        inst = builder.load(builder.gep(self.instructions, [idx]))
        idx = builder.add(idx, self.inst_idx_1)
        #
        cmp_end = builder.icmp_unsigned('==', inst, ir.Constant(self.byte_type, _END))
        builder.cbranch(cmp_end, bb_finish, bb_body)

        # block: body
        builder.position_at_end(bb_body)
        # load number of args
        idx, num_args = self._read_number(idx, self.num_args_bytes)
        builder.store(num_args, self.local_num_args)
        # load first arg value into the accumulator
        idx, arg0 = self._load_value(idx)
        builder.store(arg0, self.local_accumulator)
        # save the current bytecode index
        builder.store(idx, self.local_idx)
        #
        cmp_end = builder.icmp_unsigned('==', inst, ir.Constant(self.byte_type, _MUL))
        builder.cbranch(cmp_end, bb_mul, bb_add)

        # block: mul
        builder.position_at_end(bb_mul)
        num_args = builder.load(self.local_num_args)
        num_args = builder.sub(num_args, self.num_args_1)
        builder.store(num_args, self.local_num_args)
        more_args = builder.icmp_unsigned('>', num_args, self.num_args_0)
        builder.cbranch(more_args, bb_mul_op, bb_op_continue)

        # block: mul_op
        builder.position_at_end(bb_mul_op)
        idx = builder.load(self.local_idx)
        idx, value = self._load_value(idx)
        acc = builder.load(self.local_accumulator)
        acc = self.mul(acc, value)
        builder.store(acc, self.local_accumulator)
        builder.store(idx, self.local_idx)
        builder.branch(bb_mul)

        # block: add
        builder.position_at_end(bb_add)
        num_args = builder.load(self.local_num_args)
        num_args = builder.sub(num_args, self.num_args_1)
        builder.store(num_args, self.local_num_args)
        more_args = builder.icmp_unsigned('>', num_args, self.num_args_0)
        builder.cbranch(more_args, bb_add_op, bb_op_continue)

        # block: add_op
        builder.position_at_end(bb_add_op)
        idx = builder.load(self.local_idx)
        idx, value = self._load_value(idx)
        acc = builder.load(self.local_accumulator)
        acc = self.add(acc, value)
        builder.store(acc, self.local_accumulator)
        builder.store(idx, self.local_idx)
        builder.branch(bb_add)

        # block: op_continue
        builder.position_at_end(bb_op_continue)
        # get where we store the result
        idx = builder.load(self.local_idx)
        idx, ptr = self._load_value_ptr(idx)
        builder.store(idx, self.local_idx)
        # get and store the result
        acc = builder.load(self.local_accumulator)
        builder.store(acc, ptr)
        builder.branch(bb_while)

        # block: finish
        builder.position_at_end(bb_finish)
        builder.ret_void()

    def _read_number(self, idx: ir.Value, num_bytes: int) -> Tuple[ir.Value, ir.Value]:
        """

        Args:
            idx: current instruction index
            num_bytes: how many bytes to read from the instruction stream to form the number

        Returns:
            (idx, number)
            idx: is the updated instruction index
            number: is the read number
        """
        builder = self.builder

        llvm_type: ir.Type = ir.IntType(num_bytes * 8)

        number: ir.Value = builder.load(builder.gep(self.instructions, [idx]))
        idx = builder.add(idx, self.inst_idx_1)

        if num_bytes > 1:
            eight = ir.Constant(llvm_type, 8)
            number = builder.zext(number, llvm_type)
            for _ in range(num_bytes - 1):
                next_byte = builder.load(builder.gep(self.instructions, [idx]))
                number = builder.add(builder.shl(number, eight), builder.zext(next_byte, llvm_type))
                idx = builder.add(idx, self.inst_idx_1)

        return idx, number

    def _load_value_ptr(self, idx: ir.Value) -> Tuple[ir.Value, ir.Value]:
        builder = self.builder

        # load array first index
        index_0 = builder.load(builder.gep(self.instructions, [idx]))
        idx = builder.add(idx, self.inst_idx_1)

        # load array second index
        idx, index_1 = self._read_number(idx, self.index_bytes)

        # get the pointer
        array = builder.load(builder.gep(self.local_arrays, [index_0]))
        ptr = builder.gep(array, [index_1])

        return idx, ptr

    def _load_value(self, idx: ir.Value) -> Tuple[ir.Value, ir.Value]:
        idx, ptr = self._load_value_ptr(idx)
        value = self.builder.load(ptr)
        return idx, value


@dataclass
class _ElementID:
    """
    A 2D index into the function's `arrays`.
    """
    array: int  # which array: VARS, TMPS, CONSTS, RESULT
    index: int  # index into the array


def _make_byte_code(analysis: CircuitAnalysis, data_idx_bytes: int, num_args_bytes: int) -> List[int]:
    # Index input value elements for each possible input node.
    node_to_element: Dict[int, _ElementID] = {}
    # const nodes
    for i, node in enumerate(analysis.const_nodes):
        node_to_element[id(node)] = _ElementID(_CONSTS, i)
    # var nodes
    for i, var_node in enumerate(analysis.var_nodes):
        if var_node.is_const():
            node_to_element[id(var_node)] = node_to_element[id(var_node.const)]
        else:
            node_to_element[id(var_node)] = _ElementID(_VARS, i)
    # op nodes
    for node_id, tmp_idx in analysis.op_to_tmp.items():
        node_to_element[node_id] = _ElementID(_TMPS, tmp_idx)
    for node_id, result_idx in analysis.op_to_result.items():
        node_to_element[node_id] = _ElementID(_RESULT, result_idx)

    # Make byte code
    byte_code: List[int] = []
    for op_node in analysis.op_nodes:
        # write the op code
        byte_code.append(op_node.symbol)  # _ADD or _MUL
        # write the number of args
        byte_code.extend(_to_bytes(len(op_node.args), num_args_bytes))
        # write the element id for each arg
        for arg_node in op_node.args:
            element_id: _ElementID = node_to_element[id(arg_node)]
            byte_code.append(element_id.array)
            byte_code.extend(_to_bytes(element_id.index, data_idx_bytes))
        # write the element id for the result
        element_id: _ElementID = node_to_element[id(op_node)]
        byte_code.append(element_id.array)
        byte_code.extend(_to_bytes(element_id.index, data_idx_bytes))
    # ...any final copy instructions
    for idx, node in enumerate(analysis.result_nodes):
        if not isinstance(node, OpNode):
            byte_code.append(_ADD)
            byte_code.extend(_to_bytes(1, num_args_bytes))

            element_id: _ElementID = node_to_element[id(node)]
            byte_code.append(element_id.array)
            byte_code.extend(_to_bytes(element_id.index, data_idx_bytes))

            byte_code.append(_RESULT)
            byte_code.extend(_to_bytes(idx, data_idx_bytes))

    # write the sentinel - 'end' op code
    byte_code.append(_END)

    return byte_code


def _to_bytes(value: int, num_bytes: int) -> List[int]:
    buffer: List[int] = []
    for _ in range(num_bytes):
        buffer.append(value % 256)
        value //= 256
    assert value == 0
    buffer.reverse()
    return buffer


def _get_bytes_needed(size: int) -> int:
    index_bytes: int
    for index_bytes in [1, 2, 4, 8]:
        if size < 2 ** (index_bytes * 8 - 1):
            return index_bytes
    raise ValueError(f'size are too large to represent: {size}')
