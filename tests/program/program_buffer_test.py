import pickle
from typing import Iterable, Tuple, Optional

from ck.circuit_compiler.support.input_vars import InputVars, InferVars
from ck.circuit_compiler.llvm_compiler import compile_circuit
from ck.circuit import CircuitNode, Circuit
from ck.program.program_buffer import ProgramBuffer
from tests.helpers.unittest_fixture import Fixture, test_main


def _make_program_buffer(
        *result: CircuitNode,
        input_vars: InputVars = InferVars.ALL,
        circuit: Optional[Circuit] = None,
) -> ProgramBuffer:
    raw_program = compile_circuit(
        *result,
        input_vars=input_vars,
        circuit=circuit,
    )
    return ProgramBuffer(raw_program)


class TestProgramBuffer(Fixture):
    def test_empty(self) -> None:
        program = _make_program_buffer()

        self.assertEqual(program.number_of_vars, 0)
        self.assertEqual(program.number_of_results, 0)
        self.assertEqual(program.var_indices, ())

        program[:] = []
        result = program.compute()

        self.assertEqual(result.shape, (0,))

    def test_constant(self) -> None:
        value = 123.456
        cct = Circuit()
        const = cct.const(value)

        program = _make_program_buffer(const)

        self.assertEqual(0, program.number_of_vars)
        self.assertEqual(1, program.number_of_results)

        program[:] = []
        result = program.compute()

        self.assertEqual(result, value)

    def test_add(self) -> None:
        cct = Circuit()
        x = cct.new_vars(2)
        top = cct.add(x[0], x[1])
        program = _make_program_buffer(top)

        self.assertEqual(program.number_of_vars, 2)
        self.assertEqual(program.number_of_results, 1)
        self.assertEqual(program.var_indices, (0, 1))

        def calc(*args):
            program[:] = args
            return program.compute()

        self.assertEqual(calc(0, 0), 0)
        self.assertEqual(calc(1, 1), 2)
        self.assertEqual(calc(123, 456), 579)
        self.assertEqual(calc(-123, 456), 333)
        self.assertEqual(calc(1.24, 4.51), 5.75)

    def test_mul(self) -> None:
        cct = Circuit()
        x = cct.new_vars(2)
        top = cct.mul(x[0], x[1])
        program = _make_program_buffer(top)

        self.assertEqual(program.number_of_vars, 2)
        self.assertEqual(program.number_of_results, 1)
        self.assertEqual(program.var_indices, (0, 1))

        def calc(*args):
            program[:] = args
            return program.compute()

        self.assertEqual(calc(0, 0), 0)
        self.assertEqual(calc(1, 1), 1)
        self.assertEqual(calc(123, 456), 56088)
        self.assertEqual(calc(-123, 456), -56088)
        self.assertEqual(calc(1.23, 4.56), 5.6088)

    def test_return_array(self) -> None:
        cct = Circuit()
        x = cct.new_vars(2)
        a = cct.add(x[0], x[1])
        m = cct.mul(x[0], x[1])
        program = _make_program_buffer(a, m)

        self.assertEqual(program.number_of_vars, 2)
        self.assertEqual(program.number_of_results, 2)
        self.assertEqual(program.var_indices, (0, 1))

        def calc(*args):
            program[:] = args
            return program.compute()

        self.assertArrayEqual(calc(0, 0), [0, 0])
        self.assertArrayEqual(calc(1, 1), [2, 1])


class TestProgramBufferPickle(Fixture):

    def assert_pickle_round_trip(
            self,
            program: ProgramBuffer,
            checks: Iterable[Tuple[float, ...]],
            places=None,
            delta=None,
    ) -> None:
        pkl: bytes = pickle.dumps(program)
        clone: ProgramBuffer = pickle.loads(pkl)

        self.assertEqual(program.number_of_vars, clone.number_of_vars)
        self.assertEqual(program.number_of_results, clone.number_of_results)
        self.assertEqual(program.dtype, clone.dtype)

        for check in checks:
            program[:] = check
            clone[:] = check

            v1 = program.compute()
            v2 = clone.compute()
            if program.number_of_results == 1:
                self.assertAlmostEqual(v1, v2, places=places, delta=delta)
            else:
                self.assertArrayEqual(v1, v2)

    def test_empty(self):
        program = _make_program_buffer()
        self.assert_pickle_round_trip(program, [()])

    def test_add(self) -> None:
        cct = Circuit()
        x = cct.new_vars(2)
        top = cct.add(x[0], x[1])
        program = _make_program_buffer(top)

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
        program = _make_program_buffer(top)

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
