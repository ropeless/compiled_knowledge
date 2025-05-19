from typing import Optional

import numpy as np

from ck.circuit import Circuit, CircuitNode
from ck.circuit_compiler import interpret_compiler
from ck.circuit_compiler.support.input_vars import InputVars, InferVars
from ck.program import RawProgram
from ck.utils.np_extras import DTypeNumeric
from tests.helpers.circuit_compiler_test_cases import CompilerCases
from tests.helpers.unittest_fixture import Fixture, test_main


class TestInterpreterCompilerFloat64(Fixture, CompilerCases):

    def compile_circuit(
            self,
            *result: CircuitNode,
            input_vars: InputVars = InferVars.ALL,
            circuit: Optional[Circuit] = None,
    ) -> RawProgram:
        return interpret_compiler.compile_circuit(
            *result,
            input_vars=input_vars,
            circuit=circuit,
            dtype=np.float64,
        )


class TestInterpreterCompilerDType(Fixture):

    def assert_constant(self, value, dtype: DTypeNumeric) -> None:
        cct = Circuit()
        const = cct.const(value)

        raw_program = interpret_compiler.compile_circuit(const, dtype=dtype)

        self.assertEqual(raw_program.dtype, dtype)
        self.assertEqual(raw_program.number_of_vars, 0)
        self.assertEqual(raw_program.number_of_results, 1)
        self.assertEqual(raw_program.number_of_tmps, 0)

        result = raw_program([])

        self.assertEqual(result, value)

        with self.assertRaises(ValueError):
            raw_program(1)  # too many arguments

    def test_constant_float64(self) -> None:
        dtype: DTypeNumeric = np.float64
        self.assert_constant(value=0, dtype=dtype)
        self.assert_constant(value=123.456, dtype=dtype)
        self.assert_constant(value=-123.456, dtype=dtype)

    def test_constant_float32(self) -> None:
        dtype: DTypeNumeric = np.float32
        self.assert_constant(value=0, dtype=dtype)
        self.assert_constant(value=123.84375, dtype=dtype)
        self.assert_constant(value=-123.84375, dtype=dtype)

    def test_constant_int8(self) -> None:
        dtype: DTypeNumeric = np.int8
        self.assert_constant(value=0, dtype=dtype)
        self.assert_constant(value=2 ** 7 - 1, dtype=dtype)
        self.assert_constant(value=-(2 ** 7), dtype=dtype)

    def test_constant_int16(self) -> None:
        dtype: DTypeNumeric = np.int16
        self.assert_constant(value=0, dtype=dtype)
        self.assert_constant(value=2 ** 15 - 1, dtype=dtype)
        self.assert_constant(value=-(2 ** 15), dtype=dtype)

    def test_constant_int32(self) -> None:
        dtype: DTypeNumeric = np.int32
        self.assert_constant(value=0, dtype=dtype)
        self.assert_constant(value=2 ** 31 - 1, dtype=dtype)
        self.assert_constant(value=-(2 ** 31), dtype=dtype)

    def test_constant_int64(self) -> None:
        dtype: DTypeNumeric = np.int64
        self.assert_constant(value=0, dtype=dtype)
        self.assert_constant(value=2 ** 63 - 1, dtype=dtype)
        self.assert_constant(value=-(2 ** 63), dtype=dtype)

    def test_constant_bool(self) -> None:
        dtype: DTypeNumeric = np.int8
        self.assert_constant(value=True, dtype=dtype)
        self.assert_constant(value=False, dtype=dtype)


if __name__ == '__main__':
    test_main()
