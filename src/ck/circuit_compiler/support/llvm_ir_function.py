import ctypes as ct
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Tuple, Optional

import llvmlite.binding as llvm
import llvmlite.ir as ir

from ck.program import RawProgram
from ck.program.raw_program import RawProgramFunction
from ck.utils.np_extras import DType, DTypeNumeric

__LLVM_INITIALISED: bool = False

_LVM_FUNCTION_NAME: str = 'main'

# Type for an LLVM builder binary Operation
IrBOp = Callable[[ir.IRBuilder, ir.Value, ir.Value], ir.Value]

IrBoolType = ir.IntType(1)  # Type for an LLVM Boolean.


@dataclass(frozen=True)
class TypeInfo:
    """
    Record compiler related information contingent on a given numpy/ctypes `dtype`

    An instance of this data type defines a mathematical ring, i.e., an atomic machine
    data type and arithmetic operations over them.
    """

    dtype: DTypeNumeric  # This is the same as numpy `dtype`.
    llvm_type: ir.Type  # Corresponding LLVM IR type.
    add: IrBOp  # LLVM IR binary operation for addition.
    mul: IrBOp  # LLVM IR binary operation for multiplication.


# The Boolean constant "One", i.e., "True".
_IrBoolOne: ir.Value = ir.Constant(IrBoolType, 1)


def _bool_and(builder: ir.IRBuilder, x: ir.Value, y: ir.Value) -> ir.Value:
    """
    LLVM IR Boolean "and"
    """
    tmp: ir.Value = ir.IRBuilder.and_(builder, x, y)
    return ir.IRBuilder.and_(builder, tmp, _IrBoolOne)


def _bool_or(builder: ir.IRBuilder, x: ir.Value, y: ir.Value) -> ir.Value:
    """
    LLVM IR Boolean "or"
    """
    tmp: ir.Value = ir.IRBuilder.or_(builder, x, y)
    return ir.IRBuilder.and_(builder, tmp, _IrBoolOne)


def _bool_xor(builder: ir.IRBuilder, x: ir.Value, y: ir.Value) -> ir.Value:
    """
    LLVM IR Boolean "xor"
    """
    tmp: ir.Value = ir.IRBuilder.xor(builder, x, y)
    return ir.IRBuilder.and_(builder, tmp, _IrBoolOne)


def _float_max(builder: ir.IRBuilder, x: ir.Value, y: ir.Value) -> ir.Value:
    """
    LLVM IR floating point "max"
    """
    cond = builder.fcmp_ordered('>', x, y)
    return builder.select(cond, x, y)


def _float_min(builder: ir.IRBuilder, x: ir.Value, y: ir.Value) -> ir.Value:
    """
    LLVM IR floating point "min"
    """
    cond = builder.fcmp_ordered('<', x, y)
    return builder.select(cond, x, y)


# IR operations for TypeInfo: (add, mul)
_float_add: IrBOp = ir.IRBuilder.fadd
_float_mul: IrBOp = ir.IRBuilder.fmul
_int_add: IrBOp = ir.IRBuilder.add
_ind_mul: IrBOp = ir.IRBuilder.mul


class DataType(Enum):
    """
    Predefined TypeInfo objects.

    Each member defines a mathematical ring, i.e., a machine data
    type and the "add" and "mul" arithmetic operations over them.
    """

    FLOAT_32 = TypeInfo(ct.c_float, ir.FloatType(), _float_add, _float_mul)
    FLOAT_64 = TypeInfo(ct.c_double, ir.DoubleType(), _float_add, _float_mul)
    INT_8 = TypeInfo(ct.c_int8, ir.IntType(8), _int_add, _ind_mul)
    INT_16 = TypeInfo(ct.c_int16, ir.IntType(16), _int_add, _ind_mul)
    INT_32 = TypeInfo(ct.c_int32, ir.IntType(32), _int_add, _ind_mul)
    INT_64 = TypeInfo(ct.c_int64, ir.IntType(64), _int_add, _ind_mul)
    BOOL = TypeInfo(ct.c_bool, IrBoolType, _bool_or, _bool_and)
    XBOOL = TypeInfo(ct.c_bool, IrBoolType, _bool_xor, _bool_and)
    MAX_MIN = TypeInfo(ct.c_double, ir.DoubleType(), _float_max, _float_min)
    MAX_MUL = TypeInfo(ct.c_double, ir.DoubleType(), _float_max, _float_mul)
    MAX_SUM = TypeInfo(ct.c_double, ir.DoubleType(), _float_max, _float_add)


class IRFunction:
    """
    Data structure to hold information while building an LLVM IR program function.
    """

    def __init__(self, type_info: TypeInfo):
        """
        Create an LLVM IR program function.

        Actions performed:
        1. LLVM will be initialized.
        2. A IRBuilder will be constructed (field `builder`).
        3. A module will be created (field `module`).
        4. A function will be added to the module (field `function`), the function will
            have the signature (T* in, T* tmp, T* out) -> Void, where T is `type_info.llvm_type`.
        5. A basic block will be added to the function (named "entry").
        """
        _init_llvm()

        # Get important IR types
        self.type_info: TypeInfo = type_info
        self.ret_type: ir.Type = ir.VoidType()
        self.ptr_type: ir.Type = self.type_info.llvm_type.as_pointer()
        function_type = ir.FunctionType(self.ret_type, (self.ptr_type, self.ptr_type, self.ptr_type))

        self.module = ir.Module()
        self.function = ir.Function(self.module, function_type, name=_LVM_FUNCTION_NAME)
        self.builder = ir.IRBuilder()

        # Create a block of code in the function
        bb_entry = self.function.append_basic_block('entry')
        self.builder.position_at_end(bb_entry)

    def llvm_program(self) -> str:
        """
        Get the LLVM source code (i.e., the module as an LLVM string).

        Returns:
            an LLVM program string that can be passed to `compile_llvm_program`.
        """
        return str(self.module)


@dataclass
class LLVMRawProgram(RawProgram):
    llvm_program: Optional[str]
    engine: llvm.ExecutionEngine
    opt: int

    def __getstate__(self):
        """
        Support for pickle.
        """
        if self.llvm_program is None:
            raise ValueError('need to have the LLVM program to pickle a Program object')

        return {
            'dtype': self.dtype,
            'number_of_vars': self.number_of_vars,
            'number_of_tmps': self.number_of_tmps,
            'number_of_results': self.number_of_results,
            'llvm_program': self.llvm_program,
            'opt': self.opt,
        }

    def __setstate__(self, state):
        """
        Support for pickle.
        """
        self.dtype = state['dtype']
        self.number_of_vars = state['number_of_vars']
        self.number_of_tmps = state['number_of_tmps']
        self.number_of_results = state['number_of_results']
        self.llvm_program = state['llvm_program']
        self.opt = state['opt']

        # Compile the LLVM program
        self.engine, self.function = compile_llvm_program(self.llvm_program, self.dtype, self.opt)


def compile_llvm_program(
        llvm_program: str,
        dtype: DType,
        opt: int,
) -> Tuple[llvm.ExecutionEngine, RawProgramFunction]:
    """
    Compile the given LLVM program.

    Returns:
        (engine, function) where
        engine: is an LLVM execution engine, which must remain
            in memory for the returned function to be valid.
        function: is the raw Python callable for the compiled function.
    """
    _init_llvm()

    llvm_module = llvm.parse_assembly(llvm_program)
    llvm_module.verify()

    target = llvm.Target.from_default_triple().create_target_machine(opt=opt)
    engine = llvm.create_mcjit_compiler(llvm_module, target)

    # Calling finalize_object will create native code and make it executable.
    engine.finalize_object()

    engine.run_static_constructors()

    # Get the function entry point
    function_ptr = engine.get_function_address(_LVM_FUNCTION_NAME)
    ctypes_ptr_type = ct.POINTER(dtype)
    function = ct.CFUNCTYPE(None, ctypes_ptr_type, ctypes_ptr_type, ctypes_ptr_type)(function_ptr)

    return engine, function


def _init_llvm() -> None:
    """
    Ensure that LLVM is initialised.
    """
    global __LLVM_INITIALISED
    if not __LLVM_INITIALISED:
        llvm.initialize()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        __LLVM_INITIALISED = True
