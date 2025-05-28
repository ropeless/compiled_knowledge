import pickle
from typing import Iterable, Tuple, Optional

from ck.circuit_compiler.support.input_vars import InputVars, InferVars
from ck.circuit_compiler.llvm_compiler import compile_circuit
from ck.circuit import Circuit, CircuitNode
from ck.program import Program
from tests.helpers.unittest_fixture import Fixture, test_main


def _make_program(
        *result: CircuitNode,
        input_vars: InputVars = InferVars.ALL,
        circuit: Optional[Circuit] = None,
) -> Program:
    raw_program = compile_circuit(
        *result,
        input_vars=input_vars,
        circuit=circuit,
    )
    return Program(raw_program)


class TestProgram(Fixture):

    def test_empty(self) -> None:
        program = _make_program()

        self.assertEqual(program.number_of_vars, 0)
        self.assertEqual(program.number_of_results, 0)
        self.assertEqual(program.var_indices, ())

        result = program()

        self.assertEqual(result.shape, (0,))

        with self.assertRaises(ValueError):
            program(1)  # too many arguments

    def test_constant(self) -> None:
        value = 123.456
        cct = Circuit()
        const = cct.const(value)

        program = _make_program(const)

        self.assertEqual(0, program.number_of_vars)
        self.assertEqual(1, program.number_of_results)

        result = program()

        self.assertEqual(result, value)

        with self.assertRaises(ValueError):
            program(1)  # too many arguments

    def test_add(self) -> None:
        cct = Circuit()
        x = cct.new_vars(2)
        top = cct.add(x[0], x[1])
        program = _make_program(top)

        self.assertEqual(program.number_of_vars, 2)
        self.assertEqual(program.number_of_results, 1)
        self.assertEqual(program.var_indices, (0, 1))

        self.assertEqual(program(0, 0), 0)
        self.assertEqual(program(1, 1), 2)
        self.assertEqual(program(123, 456), 579)
        self.assertEqual(program(-123, 456), 333)
        self.assertEqual(program(1.24, 4.51), 5.75)

        with self.assertRaises(ValueError):
            program(1)  # not enough arguments

        with self.assertRaises(ValueError):
            program(1, 2, 3)  # too many arguments

    def test_mul(self) -> None:
        cct = Circuit()
        x = cct.new_vars(2)
        top = cct.mul(x[0], x[1])
        program = _make_program(top)

        self.assertEqual(program.number_of_vars, 2)
        self.assertEqual(program.number_of_results, 1)
        self.assertEqual(program.var_indices, (0, 1))

        self.assertEqual(program(0, 0), 0)
        self.assertEqual(program(1, 1), 1)
        self.assertEqual(program(123, 456), 56088)
        self.assertEqual(program(-123, 456), -56088)
        self.assertEqual(program(1.23, 4.56), 5.6088)

        with self.assertRaises(ValueError):
            program(1)  # not enough arguments

        with self.assertRaises(ValueError):
            program(1, 2, 3)  # too many arguments

    def test_return_array(self) -> None:
        cct = Circuit()
        x = cct.new_vars(2)
        a = cct.add(x[0], x[1])
        m = cct.mul(x[0], x[1])
        program = _make_program(a, m)

        self.assertEqual(program.number_of_vars, 2)
        self.assertEqual(program.number_of_results, 2)
        self.assertEqual(program.var_indices, (0, 1))

        self.assertArrayEqual(program(0, 0), [0, 0])
        self.assertArrayEqual(program(1, 1), [2, 1])

        with self.assertRaises(ValueError):
            program(1)  # not enough arguments

        with self.assertRaises(ValueError):
            program(1, 2, 3)  # too many arguments


class TestProgramPickle(Fixture):

    def assert_pickle_round_trip(
            self,
            program: Program,
            checks: Iterable[Tuple[float, ...]],
            places=None,
            delta=None,
    ) -> None:
        pkl: bytes = pickle.dumps(program)
        clone: Program = pickle.loads(pkl)

        self.assertEqual(program.number_of_vars, clone.number_of_vars)
        self.assertEqual(program.number_of_results, clone.number_of_results)
        self.assertEqual(program.dtype, clone.dtype)

        for check in checks:
            v1 = program(*check)
            v2 = clone(*check)
            if program.number_of_results == 1:
                self.assertAlmostEqual(v1, v2, places=places, delta=delta)
            else:
                self.assertArrayEqual(v1, v2)

    def test_empty(self):
        program = _make_program()
        self.assert_pickle_round_trip(program, [()])

    def test_add(self) -> None:
        cct = Circuit()
        x = cct.new_vars(2)
        top = cct.add(x[0], x[1])
        program = _make_program(top)

        cases = [
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 2),
        ]
        self.assert_pickle_round_trip(program, cases)

    def test_mul(self) -> None:
        cct = Circuit()
        x = cct.new_vars(2)
        top = cct.mul(x[0], x[1])
        program = _make_program(top)

        cases = [
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 2),
        ]
        self.assert_pickle_round_trip(program, cases)


if __name__ == '__main__':
    test_main()
