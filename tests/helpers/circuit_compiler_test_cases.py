import pickle
from abc import ABC, abstractmethod
from typing import List, Optional, Iterable, Tuple

from ck.circuit import TmpConst
from ck.circuit import ConstValue, Circuit, CircuitNode
from ck.circuit_compiler.support.input_vars import InputVars, InferVars
from ck.program import RawProgram


class CompilerCases(ABC):
    """
    This is a test case mix-in for running a circuit compiler through a variety of test cases.
    """

    @abstractmethod
    def compile_circuit(
            self,
            *result: CircuitNode,
            input_vars: InputVars = InferVars.ALL,
            circuit: Optional[Circuit] = None,
    ) -> RawProgram:
        """
        Implementation of the compiler under test.
        """
        ...

    def assert_pickle_round_trip(
            self,
            program: RawProgram,
            checks: Iterable[Tuple],
            places=None,
            delta=None,
    ) -> None:
        """
        Pickle dumps then loads the given program to create a clone. Check
        the close is the same as the given program.
        """
        pkl: bytes = pickle.dumps(program)
        clone: RawProgram = pickle.loads(pkl)

        self.assertEqual(program.number_of_vars, clone.number_of_vars)
        self.assertEqual(program.number_of_results, clone.number_of_results)
        self.assertEqual(program.number_of_tmps, clone.number_of_tmps)
        self.assertEqual(program.dtype, clone.dtype)

        for check in checks:
            v1 = program(check)
            v2 = clone(check)
            if program.number_of_results == 1:
                self.assertAlmostEqual(v1, v2, places=places, delta=delta)
            else:
                self.assertArrayEqual(v1, v2)

    def test_empty(self) -> None:
        program = self.compile_circuit()

        self.assertEqual(program.number_of_vars, 0)
        self.assertEqual(program.number_of_results, 0)
        self.assertEqual(program.var_indices, ())

        result = program([])

        self.assertEqual(result.shape, (0,))

        with self.assertRaises(ValueError):
            program(1)  # too many arguments

    def test_constant(self) -> None:
        # testing the basic constants that should work for any compiled circuit
        test_values: List[ConstValue] = [
            0, 1, 2 ** 7 - 1,
        ]
        for value in test_values:
            cct = Circuit()
            const = cct.const(value)

            program = self.compile_circuit(const)

            self.assertEqual(0, program.number_of_vars)
            self.assertEqual(1, program.number_of_results)

            result = program([])

            self.assertEqual(result, value)

            with self.assertRaises(ValueError):
                program(1)  # too many arguments

    def test_add(self) -> None:
        cct = Circuit()
        x = cct.new_vars(2)
        top = cct.add(x[0], x[1])
        program = self.compile_circuit(top)

        self.assertEqual(program.number_of_vars, 2)
        self.assertEqual(program.number_of_results, 1)
        self.assertEqual(program.var_indices, (0, 1))

        self.assertEqual(program([0, 0]), 0)
        self.assertEqual(program([1, 1]), 2)
        self.assertEqual(program([123, 456]), 579)
        self.assertEqual(program([-123, 456]), 333)
        self.assertEqual(program([1.24, 4.51]), 5.75)

        with self.assertRaises(ValueError):
            program([])  # not enough arguments

        with self.assertRaises(ValueError):
            program([1, 2, 3])  # too many arguments

    def test_mul(self) -> None:
        cct = Circuit()
        x = cct.new_vars(2)
        top = cct.mul(x[0], x[1])
        program = self.compile_circuit(top)

        self.assertEqual(program.number_of_vars, 2)
        self.assertEqual(program.number_of_results, 1)
        self.assertEqual(program.var_indices, (0, 1))

        self.assertEqual(program([0, 0]), 0)
        self.assertEqual(program([1, 1]), 1)
        self.assertEqual(program([123, 456]), 56088)
        self.assertEqual(program([-123, 456]), -56088)
        self.assertEqual(program([1.23, 4.56]), 5.6088)

        with self.assertRaises(ValueError):
            program([1])  # not enough arguments

        with self.assertRaises(ValueError):
            program([1, 2, 3])  # too many arguments

    def test_return_array(self) -> None:
        cct = Circuit()
        x = cct.new_vars(2)
        a = cct.add(x[0], x[1])
        m = cct.mul(x[0], x[1])
        program = self.compile_circuit(a, m)

        self.assertEqual(program.number_of_vars, 2)
        self.assertEqual(program.number_of_results, 2)
        self.assertEqual(program.var_indices, (0, 1))

        self.assertArrayEqual(program([0, 0]), [0, 0])
        self.assertArrayEqual(program([1, 1]), [2, 1])

        with self.assertRaises(ValueError):
            program([1])  # not enough arguments

        with self.assertRaises(ValueError):
            program([1, 2, 3])  # too many arguments


    def test_const_vars(self) -> None:
        cct = Circuit()
        x = cct.new_vars(2)
        top = cct.add(x[0], x[1])

        with TmpConst(cct) as tmp_const:
            tmp_const.set_const(x[1], 123)
            program = self.compile_circuit(top)

        self.assertEqual(program([0, 0]), 123)
        self.assertEqual(program([1, 1]), 124)
        self.assertEqual(program([123, 456]), 246)
        self.assertEqual(program([-123, 456]), 0)

    def test_pickle_empty(self):
        program = self.compile_circuit()
        self.assert_pickle_round_trip(program, [()])

    def test_pickle_add(self) -> None:
        cct = Circuit()
        x = cct.new_vars(2)
        top = cct.add(x[0], x[1])
        program = self.compile_circuit(top)

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

    def test_pickle_mul(self) -> None:
        cct = Circuit()
        x = cct.new_vars(2)
        top = cct.mul(x[0], x[1])
        program = self.compile_circuit(top)

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

