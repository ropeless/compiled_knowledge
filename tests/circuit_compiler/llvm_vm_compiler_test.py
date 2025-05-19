import ctypes as ct
from typing import Optional

from llvmlite import ir

from ck.circuit import Circuit, TmpConst, CircuitNode
from ck.circuit_compiler import llvm_vm_compiler
from ck.circuit_compiler.support.input_vars import InputVars, InferVars
from ck.circuit_compiler.support.llvm_ir_function import DataType, TypeInfo
from ck.program import RawProgram
from tests.helpers.circuit_compiler_test_cases import CompilerCases
from tests.helpers.unittest_fixture import Fixture, test_main


class TestLLVMVMCompilerFloat64ArraysTrue(Fixture, CompilerCases):

    def compile_circuit(
            self,
            *result: CircuitNode,
            input_vars: InputVars = InferVars.ALL,
            circuit: Optional[Circuit] = None,
    ) -> RawProgram:
        return llvm_vm_compiler.compile_circuit(
            *result,
            input_vars=input_vars,
            circuit=circuit,
            compile_arrays=True,
        )


class TestLLVMVMCompilerFloat64ArraysFalse(Fixture, CompilerCases):

    def compile_circuit(
            self,
            *result: CircuitNode,
            input_vars: InputVars = InferVars.ALL,
            circuit: Optional[Circuit] = None,
    ) -> RawProgram:
        return llvm_vm_compiler.compile_circuit(
            *result,
            input_vars=input_vars,
            circuit=circuit,
            compile_arrays=False,
        )


class TestLLVMVMCompilerDType(Fixture):

    def assert_constant(self, value, data_type: DataType) -> None:
        cct = Circuit()
        const = cct.const(value)

        program = llvm_vm_compiler.compile_circuit(const, data_type=data_type)

        self.assertEqual(0, program.number_of_vars)
        self.assertEqual(1, program.number_of_results)

        result = program([])

        self.assertEqual(result, value)

        with self.assertRaises(ValueError):
            program(1)  # too many arguments

    def test_constant_float64(self) -> None:
        data_type = DataType.FLOAT_64
        self.assert_constant(value=0, data_type=data_type)
        self.assert_constant(value=123.456, data_type=data_type)
        self.assert_constant(value=-123.456, data_type=data_type)

    def test_constant_float32(self) -> None:
        data_type = DataType.FLOAT_32
        self.assert_constant(value=0, data_type=data_type)
        self.assert_constant(value=123.84375, data_type=data_type)
        self.assert_constant(value=-123.84375, data_type=data_type)

    def test_constant_int8(self) -> None:
        data_type = DataType.INT_8
        self.assert_constant(value=0, data_type=data_type)
        self.assert_constant(value=2 ** 7 - 1, data_type=data_type)
        self.assert_constant(value=-(2 ** 7), data_type=data_type)

    def test_constant_int16(self) -> None:
        data_type = DataType.INT_16
        self.assert_constant(value=0, data_type=data_type)
        self.assert_constant(value=2 ** 15 - 1, data_type=data_type)
        self.assert_constant(value=-(2 ** 15), data_type=data_type)

    def test_constant_int32(self) -> None:
        data_type = DataType.INT_32
        self.assert_constant(value=0, data_type=data_type)
        self.assert_constant(value=2 ** 31 - 1, data_type=data_type)
        self.assert_constant(value=-(2 ** 31), data_type=data_type)

    def test_constant_int64(self) -> None:
        data_type = DataType.INT_64
        self.assert_constant(value=0, data_type=data_type)
        self.assert_constant(value=2 ** 63 - 1, data_type=data_type)
        self.assert_constant(value=-(2 ** 63), data_type=data_type)

    def test_constant_bool(self) -> None:
        data_type = DataType.BOOL
        self.assert_constant(value=True, data_type=data_type)
        self.assert_constant(value=False, data_type=data_type)

    def test_bool(self) -> None:
        cct = Circuit()
        x = cct.new_vars(2)
        add = cct.add(x)  # or
        mul = cct.mul(x)  # and

        program = llvm_vm_compiler.compile_circuit(add, mul, data_type=DataType.BOOL)

        self.assertArrayEqual(program([False, False]), [False, False])
        self.assertArrayEqual(program([False, True]), [True, False])
        self.assertArrayEqual(program([True, False]), [True, False])
        self.assertArrayEqual(program([True, True]), [True, True])

    def test_xbool(self) -> None:
        cct = Circuit()
        x = cct.new_vars(2)
        add = cct.add(x)  # xor
        mul = cct.mul(x)  # and

        program = llvm_vm_compiler.compile_circuit(add, mul, data_type=DataType.XBOOL)

        self.assertArrayEqual(program([False, False]), [False, False])
        self.assertArrayEqual(program([False, True]), [True, False])
        self.assertArrayEqual(program([True, False]), [True, False])
        self.assertArrayEqual(program([True, True]), [False, True])

    def test_max_min(self) -> None:
        cct = Circuit()
        x = cct.new_vars(2)
        add = cct.add(x)  # max
        mul = cct.mul(x)  # min

        program = llvm_vm_compiler.compile_circuit(add, mul, data_type=DataType.MAX_MIN)

        self.assertArrayEqual(program([123.125, 234.625]), [234.625, 123.125])
        self.assertArrayEqual(program([234.625, 123.125]), [234.625, 123.125])

    def test_max_mul(self) -> None:
        cct = Circuit()
        x = cct.new_vars(2)
        add = cct.add(x)  # max
        mul = cct.mul(x)  # mul

        program = llvm_vm_compiler.compile_circuit(add, mul, data_type=DataType.MAX_MUL)

        self.assertArrayEqual(program([123.125, 234.625]), [234.625, 28888.203125])
        self.assertArrayEqual(program([234.625, 123.125]), [234.625, 28888.203125])

    def test_max_sum(self) -> None:
        cct = Circuit()
        x = cct.new_vars(2)
        add = cct.add(x)  # max
        mul = cct.mul(x)  # sum

        program = llvm_vm_compiler.compile_circuit(add, mul, data_type=DataType.MAX_SUM)

        self.assertArrayEqual(program([123.125, 234.625]), [234.625, 357.75])
        self.assertArrayEqual(program([234.625, 123.125]), [234.625, 357.75])

    def test_custom_type_info(self) -> None:
        # This type info deliberately has the wrong IR operations
        type_info = TypeInfo(
            dtype=ct.c_double,
            llvm_type=ir.DoubleType(),
            add=ir.IRBuilder.fmul,
            mul=ir.IRBuilder.fadd,
        )

        cct = Circuit()
        x = cct.new_vars(2)
        # deliberately wrong operations - as per the type info above
        add = cct.mul(x)
        mul = cct.add(x)

        program = llvm_vm_compiler.compile_circuit(add, mul, data_type=type_info)

        self.assertArrayEqual(program([0, 0]), [0, 0])
        self.assertArrayEqual(program([1, 1]), [2, 1])
        self.assertArrayEqual(program([123, 456]), [579, 56088])
        self.assertArrayEqual(program([-123, 456]), [333, -56088])
        self.assertArrayEqual(program([1.25, 4.5]), [5.75, 5.625])

    def test_const_vars(self) -> None:
        cct = Circuit()
        x = cct.new_vars(2)
        top = cct.add(x[0], x[1])

        with TmpConst(cct) as tmp_const:
            tmp_const.set_const(x[1], 123)
            program = llvm_vm_compiler.compile_circuit(top)

        self.assertEqual(program([0, 0]), 123)
        self.assertEqual(program([1, 1]), 124)
        self.assertEqual(program([123, 456]), 246)
        self.assertEqual(program([-123, 456]), 0)


if __name__ == '__main__':
    test_main()
