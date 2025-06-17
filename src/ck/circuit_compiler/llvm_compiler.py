from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence, Optional, Tuple, Dict, Protocol, assert_never

import llvmlite.binding as llvm
import llvmlite.ir as ir

from .support.circuit_analyser import CircuitAnalysis, analyze_circuit
from .support.input_vars import InputVars, InferVars, infer_input_vars
from .support.llvm_ir_function import IRFunction, DataType, TypeInfo, compile_llvm_program, LLVMRawProgram, IrBOp
from ..circuit import Circuit, VarNode, CircuitNode, OpNode, MUL, ADD, ConstNode
from ..program.raw_program import RawProgramFunction


class Flavour(Enum):
    STACK = 0  # No working temporary memory requested - all on stack.
    TMPS = 1  # Working temporary memory used for op node calculations.
    FUNCS = 2  # Working temporary memory used for op node calculations, one sub-function per op-node.


DEFAULT_TYPE_INFO: TypeInfo = DataType.FLOAT_64.value
DEFAULT_FLAVOUR: Flavour = Flavour.TMPS


def compile_circuit(
        *result: CircuitNode,
        input_vars: InputVars = InferVars.ALL,
        circuit: Optional[Circuit] = None,
        data_type: DataType | TypeInfo = DEFAULT_TYPE_INFO,
        flavour: Flavour = DEFAULT_FLAVOUR,
        keep_llvm_program: bool = True,
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
        flavour: what flavour of LLVM program to construct.
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
    llvm_program, number_of_tmps = _make_llvm_program(in_vars, result, type_info, flavour)

    # Compile the LLVM program to a native executable
    engine: llvm.ExecutionEngine
    function: RawProgramFunction
    engine, function = compile_llvm_program(llvm_program, dtype=type_info.dtype, opt=opt)

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


def _make_llvm_program(
        in_vars: Sequence[VarNode],
        result: Sequence[CircuitNode],
        type_info: TypeInfo,
        flavour: Flavour,
) -> Tuple[str, int]:
    """
    Returns:
        (llvm_program, number_of_tmps)
    """
    llvm_function = IRFunction(type_info)

    builder = llvm_function.builder
    type_info = llvm_function.type_info
    function = llvm_function.function

    analysis: CircuitAnalysis = analyze_circuit(in_vars, result)

    function_builder: _FunctionBuilder
    if flavour == Flavour.STACK:
        function_builder = _FunctionBuilderStack(
            builder=builder,
            analysis=analysis,
            llvm_type=type_info.llvm_type,
            llvm_idx_type=ir.IntType(32),
            in_args=function.args[0],
            out_args=function.args[2],
            ir_cache={},
        )
    elif flavour == Flavour.TMPS:
        function_builder = _FunctionBuilderTmps(
            builder=builder,
            analysis=analysis,
            llvm_type=type_info.llvm_type,
            llvm_idx_type=ir.IntType(32),
            in_args=function.args[0],
            tmp_args=function.args[1],
            out_args=function.args[2],
        )
    elif flavour == Flavour.FUNCS:
        function_builder = _FunctionBuilderFuncs(
            builder=builder,
            analysis=analysis,
            llvm_type=type_info.llvm_type,
            llvm_idx_type=ir.IntType(32),
            in_args=function.args[0],
            tmp_args=function.args[1],
            out_args=function.args[2],
        )
    else:
        raise ValueError(f'unknown LLVM program flavour: {flavour!r}')

    # Add a calculation for each op node
    for op_node in analysis.op_nodes:
        if op_node.symbol == ADD:
            op: IrBOp = type_info.add
        elif op_node.symbol == MUL:
            op: IrBOp = type_info.mul
        else:
            raise RuntimeError(f'unknown op node: {op_node.symbol!r}')
        function_builder.process_op_node(op_node, op)

    # Copy any non-op node values to the results
    for idx, node in enumerate(result):
        if not isinstance(node, OpNode):
            value: ir.Value = function_builder.value(node)
            function_builder.store_result(value, idx)

    # Return from the function
    builder.ret_void()

    return llvm_function.llvm_program(), function_builder.number_of_tmps()


class _FunctionBuilder(Protocol):
    def process_op_node(self, op_node: OpNode, op: IrBOp) -> None:
        ...

    def value(self, node: CircuitNode) -> ir.Value:
        ...

    def store_result(self, value: ir.Value, idx: int) -> None:
        ...

    def number_of_tmps(self) -> int:
        ...


@dataclass
class _FunctionBuilderTmps(_FunctionBuilder):
    """
    A function builder that puts op node calculations into the temporary working memory.
    """
    builder: ir.IRBuilder
    analysis: CircuitAnalysis
    llvm_type: ir.Type
    llvm_idx_type: ir.Type
    in_args: ir.Value
    tmp_args: ir.Value
    out_args: ir.Value

    def number_of_tmps(self) -> int:
        return len(self.analysis.op_to_tmp)

    def process_op_node(self, op_node: OpNode, op: IrBOp) -> None:
        value: ir.Value = self.value(op_node.args[0])
        for arg in op_node.args[1:]:
            next_value: ir.Value = self.value(arg)
            value = op(self.builder, value, next_value)
        self.store_calculation(value, op_node)

    def value(self, node: CircuitNode) -> ir.Value:
        """
        Return an IR value for the given circuit node.
        """
        node_id: int = id(node)

        # If it is a constant...
        if isinstance(node, ConstNode):
            return ir.Constant(self.llvm_type, node.value)

        builder = self.builder

        # If it is a var...
        if isinstance(node, VarNode):
            if node.is_const():
                return ir.Constant(self.llvm_type, node.const.value)
            else:
                return builder.load(builder.gep(self.in_args, [ir.Constant(self.llvm_idx_type, node.idx)]))

        analysis = self.analysis

        # If it is an op _not_ in the results...
        idx: Optional[int] = analysis.op_to_tmp.get(node_id)
        if idx is not None:
            return builder.load(builder.gep(self.tmp_args, [ir.Constant(self.llvm_idx_type, idx)]))

        # If it is an op in the results...
        idx: Optional[int] = analysis.op_to_result.get(node_id)
        if idx is not None:
            return builder.load(builder.gep(self.out_args, [ir.Constant(self.llvm_idx_type, idx)]))

        assert_never('not reached')

    def store_calculation(self, value: ir.Value, op_node: OpNode) -> None:
        """
        Store the given IR value as a result for the given op node.
        """
        builder = self.builder
        analysis = self.analysis
        node_id: int = id(op_node)

        # If it is an op _not_ in the results...
        idx: Optional[int] = analysis.op_to_tmp.get(node_id)
        if idx is not None:
            ptr: ir.GEPInstr = builder.gep(self.tmp_args, [ir.Constant(self.llvm_idx_type, idx)])
            builder.store(value, ptr)
            return

        # If it is an op in the results...
        idx: Optional[int] = analysis.op_to_result.get(node_id)
        if idx is not None:
            ptr: ir.GEPInstr = builder.gep(self.out_args, [ir.Constant(self.llvm_idx_type, idx)])
            builder.store(value, ptr)
            return

        assert_never('not reached')

    def store_result(self, value: ir.Value, idx: int) -> None:
        """
        Store the given IR value in the indexed result slot.
        """
        builder = self.builder
        ptr: ir.GEPInstr = builder.gep(self.out_args, [ir.Constant(self.llvm_idx_type, idx)])
        builder.store(value, ptr)


class _FunctionBuilderFuncs(_FunctionBuilderTmps):
    """
    A function builder that puts op node calculations into the temporary working memory,
    but each op node becomes its own sub-function.
    """

    def process_op_node(self, op_node: OpNode, op: IrBOp) -> None:
        builder: ir.IRBuilder = self.builder
        save_block = builder.block

        sub_function_name: str = f'sub_{id(op_node)}'
        function_type = builder.function.type.pointee
        sub_function = ir.Function(builder.module, function_type, name=sub_function_name)
        sub_function.attributes.add('noinline')  # alwaysinline, noinline
        bb_entry = sub_function.append_basic_block(sub_function_name + '_entry')
        self.builder.position_at_end(bb_entry)

        value: ir.Value = self.value(op_node.args[0])
        for arg in op_node.args[1:]:
            next_value: ir.Value = self.value(arg)
            value = op(self.builder, value, next_value)
        self.store_calculation(value, op_node)

        builder.ret_void()

        # Restore builder to main function
        builder.position_at_end(save_block)
        builder.call(sub_function, [self.in_args, self.tmp_args, self.out_args])


@dataclass
class _FunctionBuilderStack(_FunctionBuilder):
    """
    A function builder that puts op node calculations onto the stack.
    """
    builder: ir.IRBuilder
    analysis: CircuitAnalysis
    llvm_type: ir.Type
    llvm_idx_type: ir.Type
    in_args: ir.Value
    out_args: ir.Value
    ir_cache: Dict[int, ir.Value]

    def number_of_tmps(self) -> int:
        return 0

    def process_op_node(self, op_node: OpNode, op: IrBOp) -> None:
        value: ir.Value = self.value(op_node.args[0])
        for arg in op_node.args[1:]:
            next_value: ir.Value = self.value(arg)
            value = op(self.builder, value, next_value)
        self.store_calculation(value, op_node)

    def value(self, node: CircuitNode) -> ir.Value:
        """
        Return an IR value for the given circuit node.
        """
        node_id: int = id(node)

        # First check if it is in the IR cache
        cached: Optional[ir.Value] = self.ir_cache.get(node_id)
        if cached is not None:
            return cached

        # If it is a constant...
        if isinstance(node, ConstNode):
            value = ir.Constant(self.llvm_type, node.value)
            self.ir_cache[node_id] = value
            return value

        builder = self.builder

        # If it is a var...
        if isinstance(node, VarNode):
            if node.is_const():
                value = ir.Constant(self.llvm_type, node.const.value)
            else:
                value = builder.load(builder.gep(self.in_args, [ir.Constant(self.llvm_idx_type, node.idx)]))
            self.ir_cache[node_id] = value
            return value

        # If it is an op in the results...
        idx: Optional[int] = self.analysis.op_to_result.get(node_id)
        if idx is not None:
            return builder.load(builder.gep(self.out_args, [ir.Constant(self.llvm_idx_type, idx)]))

        assert_never('not reached')

    def store_calculation(self, value: ir.Value, op_node: OpNode) -> None:
        """
        Store the given IR value as a result for the given op node.
        """
        node_id: int = id(op_node)

        # If it is an op in the results...
        idx: Optional[int] = self.analysis.op_to_result.get(node_id)
        if idx is not None:
            builder = self.builder
            ptr: ir.GEPInstr = builder.gep(self.out_args, [ir.Constant(self.llvm_idx_type, idx)])
            builder.store(value, ptr)
            return

        # Just put it in the ir_cache.
        # This effectively forces the LLVM compiler to put it on the stack when registers run out.
        self.ir_cache[node_id] = value

    def store_result(self, value: ir.Value, idx: int) -> None:
        """
        Store the given IR value in the indexed result slot.
        """
        builder = self.builder
        ptr: ir.GEPInstr = builder.gep(self.out_args, [ir.Constant(self.llvm_idx_type, idx)])
        builder.store(value, ptr)
