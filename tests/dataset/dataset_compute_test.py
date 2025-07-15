from typing import Sequence

import numpy as np

from ck.circuit import Circuit, VarNode
from ck.circuit_compiler import NamedCircuitCompiler
from ck.dataset import SoftDataset
from ck.dataset.dataset_compute import get_slot_arrays, accumulate_compute
from ck.pgm import PGM, RandomVariable
from ck.pgm_circuit.slot_map import SlotMap
from ck.utils.np_extras import NDArray
from tests.helpers.unittest_fixture import Fixture, test_main


class TestSlotArrays(Fixture):

    def test_empty(self):
        dataset = SoftDataset()
        number_of_slots: int = 0
        slot_map: SlotMap = {}

        slot_arrays = get_slot_arrays(dataset, number_of_slots, slot_map)

        self.assertEqual(slot_arrays.shape, (0, 0))

    def test_no_slots(self):
        dataset = SoftDataset(length=7)
        number_of_slots: int = 0
        slot_map: SlotMap = {}

        slot_arrays = get_slot_arrays(dataset, number_of_slots, slot_map)

        self.assertEqual(slot_arrays.shape, (7, 0))

    def test_no_instances(self):
        pgm = PGM()
        x: RandomVariable = pgm.new_rv('x', 2)
        y: RandomVariable = pgm.new_rv('y', 3)

        # A dataset for rvs x & y, but no instances
        x_data = np.zeros((0, len(x)))
        y_data = np.zeros((0, len(y)))
        dataset = SoftDataset([
            (x, x_data),
            (y, y_data),
        ])

        number_of_slots: int = 5
        slot_map: SlotMap = {
            x[0]: 0,
            x[1]: 1,
            y[0]: 2,
            y[1]: 3,
            y[2]: 4,
        }

        slot_arrays = get_slot_arrays(dataset, number_of_slots, slot_map)

        self.assertEqual(slot_arrays.shape, (0, 5))

    def test_simple(self):
        pgm = PGM()
        x: RandomVariable = pgm.new_rv('x', 2)
        y: RandomVariable = pgm.new_rv('y', 3)

        # A dataset for rvs x & y, with 7 instances
        x_data = np.random.rand(7, len(x))
        y_data = np.random.rand(7, len(y))
        dataset = SoftDataset([
            (x, x_data),
            (y, y_data),
        ])

        number_of_slots: int = 5
        slot_map: SlotMap = {
            x[0]: 0,
            x[1]: 1,
            y[0]: 2,
            y[1]: 3,
            y[2]: 4,
        }

        slot_arrays: NDArray = get_slot_arrays(dataset, number_of_slots, slot_map)

        self.assertEqual(slot_arrays.shape, (7, 5))

        for i, row in enumerate(slot_arrays):
            x_slot_values = [row[0], row[1]]
            y_slot_values = [row[2], row[3], row[4]]
            x_expected_values = x_data[i, :].tolist()
            y_expected_values = y_data[i, :].tolist()
            self.assertArrayEqual(x_slot_values, x_expected_values)
            self.assertArrayEqual(y_slot_values, y_expected_values)

    def test_slot_order(self):
        pgm = PGM()
        x: RandomVariable = pgm.new_rv('x', 2)
        y: RandomVariable = pgm.new_rv('y', 3)

        # A dataset for rvs x & y, with 7 instances
        x_data = np.random.rand(7, len(x))
        y_data = np.random.rand(7, len(y))
        dataset = SoftDataset([
            (x, x_data),
            (y, y_data),
        ])

        number_of_slots: int = 5
        slot_map: SlotMap = {
            x[0]: 4,
            x[1]: 2,
            y[0]: 1,
            y[1]: 0,
            y[2]: 3,
        }

        slot_arrays: NDArray = get_slot_arrays(dataset, number_of_slots, slot_map)

        self.assertEqual(slot_arrays.shape, (7, 5))

        for i, row in enumerate(slot_arrays):
            x_slot_values = [row[4], row[2]]
            y_slot_values = [row[1], row[0], row[3]]
            x_expected_values = x_data[i, :].tolist()
            y_expected_values = y_data[i, :].tolist()
            self.assertArrayEqual(x_slot_values, x_expected_values)
            self.assertArrayEqual(y_slot_values, y_expected_values)

    def test_extra_slot_ignored(self):
        pgm = PGM()
        x: RandomVariable = pgm.new_rv('x', 2)
        y: RandomVariable = pgm.new_rv('y', 3)

        # A dataset for rvs x & y, with 7 instances
        x_data = np.random.rand(7, len(x))
        y_data = np.random.rand(7, len(y))
        dataset = SoftDataset([
            (x, x_data),
            (y, y_data),
        ])

        number_of_slots: int = 4  # deliberately one slot short
        slot_map: SlotMap = {
            x[0]: 0,
            x[1]: 1,
            y[0]: 2,
            y[1]: 3,
            y[2]: 4,  # should be ignored
        }

        slot_arrays: NDArray = get_slot_arrays(dataset, number_of_slots, slot_map)

        self.assertEqual(slot_arrays.shape, (7, 4))

        for i, row in enumerate(slot_arrays):
            x_slot_values = [row[0], row[1]]
            y_slot_values = [row[2], row[3]]
            x_expected_values = x_data[i, :].tolist()
            y_expected_values = y_data[i, :2].tolist()
            self.assertArrayEqual(x_slot_values, x_expected_values)
            self.assertArrayEqual(y_slot_values, y_expected_values)

    def test_multiple_indicators_error(self):
        pgm = PGM()
        x: RandomVariable = pgm.new_rv('x', 2)
        y: RandomVariable = pgm.new_rv('y', 3)

        # A dataset for rvs x & y, with 7 instances
        x_data = np.random.rand(7, len(x))
        y_data = np.random.rand(7, len(y))
        dataset = SoftDataset([
            (x, x_data),
            (y, y_data),
        ])

        number_of_slots: int = 1
        slot_map: SlotMap = {
            x[0]: 0,
            x[1]: 0,
        }
        # Two indicators for slot 0

        with self.assertRaises(ValueError):
            get_slot_arrays(dataset, number_of_slots, slot_map)

    def test_missing_slot_error(self):
        pgm = PGM()
        x: RandomVariable = pgm.new_rv('x', 2)
        y: RandomVariable = pgm.new_rv('y', 3)

        # A dataset for rvs x & y, with 7 instances
        x_data = np.random.rand(7, len(x))
        y_data = np.random.rand(7, len(y))
        dataset = SoftDataset([
            (x, x_data),
            (y, y_data),
        ])

        number_of_slots: int = 3
        slot_map: SlotMap = {
            x[0]: 0,
            x[1]: 1,
        }
        # No indicator for slot 2

        with self.assertRaises(ValueError):
            get_slot_arrays(dataset, number_of_slots, slot_map)


class TestAccumulateCompute(Fixture):
    def test_one_result(self):
        # Make a simple program, with two results
        cct: Circuit = Circuit()
        v: Sequence[VarNode] = cct.new_vars(5)
        top = (v[0] + v[1]) * (v[2] + v[3] + v[4])
        program = NamedCircuitCompiler.INTERPRET(top)

        # Create slot arrays for computation
        number_of_instances: int = 7
        slot_arrays: NDArray = np.random.rand(number_of_instances, len(v))

        result: NDArray = accumulate_compute(program, slot_arrays)
        self.assertEqual(result.shape, (1,))

        # Compute expected result, knowing the program we created
        expect = 0
        for row in slot_arrays:
            expect += (row[0] + row[1]) * (row[2] + row[3] + row[4])

        self.assertEqual(expect, result[0])

    def test_two_results(self):
        # Make a simple program, with two results
        cct: Circuit = Circuit()
        v: Sequence[VarNode] = cct.new_vars(5)
        top0 = v[0] + v[2] + v[4]
        top1 = v[1] * v[3]
        program = NamedCircuitCompiler.INTERPRET(top0, top1)

        # Create slot arrays for computation
        number_of_instances: int = 7
        slot_arrays: NDArray = np.random.rand(number_of_instances, len(v))

        result: NDArray = accumulate_compute(program, slot_arrays)
        self.assertEqual(result.shape, (2,))

        # Compute expected result, knowing the program we used
        expect0 = 0
        expect1 = 0
        for row in slot_arrays:
            expect0 += row[0] + row[2] + row[4]
            expect1 += row[1] * row[3]

        self.assertEqual(expect0, result[0])
        self.assertEqual(expect1, result[1])

    def test_with_accumulator(self):
        # Make a simple program, with two results
        cct: Circuit = Circuit()
        v: Sequence[VarNode] = cct.new_vars(5)
        top0 = v[0] + v[2] + v[4]
        top1 = v[1] * v[3]
        program = NamedCircuitCompiler.INTERPRET(top0, top1)

        # Create slot arrays for computation
        number_of_instances: int = 7
        slot_arrays: NDArray = np.random.rand(number_of_instances, len(v))

        accumulator: NDArray = np.array([123.0, 456.0])
        result: NDArray = accumulate_compute(program, slot_arrays, accumulator=accumulator)
        self.assertIs(result, accumulator)

        # Compute expected result, knowing the program and accumulator we used
        expect0 = 123
        expect1 = 456
        for row in slot_arrays:
            expect0 += row[0] + row[2] + row[4]
            expect1 += row[1] * row[3]

        self.assertEqual(expect0, result[0])
        self.assertEqual(expect1, result[1])

    def test_with_weights(self):
        # Make a simple program, with two results
        cct: Circuit = Circuit()
        v: Sequence[VarNode] = cct.new_vars(5)
        top0 = v[0] + v[2] + v[4]
        top1 = v[1] * v[3]
        program = NamedCircuitCompiler.INTERPRET(top0, top1)

        # Create slot arrays and weights for computation
        number_of_instances: int = 7
        slot_arrays: NDArray = np.random.rand(number_of_instances, len(v))
        weights: NDArray = np.random.rand(number_of_instances)

        result: NDArray = accumulate_compute(program, slot_arrays, weights=weights)
        self.assertEqual(result.shape, (2,))

        # Compute expected result, knowing the program and weights we used
        expect0 = 0
        expect1 = 0
        for weight, row in zip(weights, slot_arrays):
            expect0 += weight * (row[0] + row[2] + row[4])
            expect1 += weight * (row[1] * row[3])

        self.assertEqual(expect0, result[0])
        self.assertEqual(expect1, result[1])

    def test_bad_slot_arrays_shape(self):
        # Make a simple program, with two results
        cct: Circuit = Circuit()
        v: Sequence[VarNode] = cct.new_vars(5)
        top0 = v[0] + v[2] + v[4]
        top1 = v[1] * v[3]
        program = NamedCircuitCompiler.INTERPRET(top0, top1)

        # Create slot arrays for computation
        number_of_instances: int = 7
        slot_arrays: NDArray = np.random.rand(number_of_instances, len(v) + 1)

        with self.assertRaises(ValueError):
            accumulate_compute(program, slot_arrays)

    def test_bad_accumulator_shape(self):
        # Make a simple program, with two results
        cct: Circuit = Circuit()
        v: Sequence[VarNode] = cct.new_vars(5)
        top0 = v[0] + v[2] + v[4]
        top1 = v[1] * v[3]
        program = NamedCircuitCompiler.INTERPRET(top0, top1)

        # Create slot arrays for computation
        number_of_instances: int = 7
        slot_arrays: NDArray = np.random.rand(number_of_instances, len(v))

        accumulator: NDArray = np.array([123.0, 456.0, 789.0])  # extra values

        with self.assertRaises(ValueError):
            accumulate_compute(program, slot_arrays, accumulator=accumulator)

    def test_bad_accumulator_type(self):
        # Make a simple program, with two results
        cct: Circuit = Circuit()
        v: Sequence[VarNode] = cct.new_vars(5)
        top0 = v[0] + v[2] + v[4]
        top1 = v[1] * v[3]
        program = NamedCircuitCompiler.INTERPRET(top0, top1)

        assert program.dtype == np.float64, 'test assumption'

        # Create slot arrays for computation
        number_of_instances: int = 7
        slot_arrays: NDArray = np.random.rand(number_of_instances, len(v))

        accumulator: NDArray = np.array([123, 456], dtype=np.intc)  # wrong dtype

        with self.assertRaises(ValueError):
            accumulate_compute(program, slot_arrays, accumulator=accumulator)

    def test_bad_slot_array_type(self):
        # Make a simple program, with two results
        cct: Circuit = Circuit()
        v: Sequence[VarNode] = cct.new_vars(5)
        top0 = v[0] + v[2] + v[4]
        top1 = v[1] * v[3]
        program = NamedCircuitCompiler.INTERPRET(top0, top1)

        assert program.dtype == np.float64, 'test assumption'

        # Create slot arrays for computation
        number_of_instances: int = 7
        slot_arrays: NDArray = np.random.rand(number_of_instances, len(v)).astype(np.float32)  # wrong dtype

        accumulator: NDArray = np.array([123.0, 456.0])

        with self.assertRaises(ValueError):
            accumulate_compute(program, slot_arrays, accumulator=accumulator)

    def test_bad_weights_shape(self):
        # Make a simple program, with two results
        cct: Circuit = Circuit()
        v: Sequence[VarNode] = cct.new_vars(5)
        top0 = v[0] + v[2] + v[4]
        top1 = v[1] * v[3]
        program = NamedCircuitCompiler.INTERPRET(top0, top1)

        # Create slot arrays and weights for computation
        number_of_instances: int = 7
        slot_arrays: NDArray = np.random.rand(number_of_instances, len(v))
        weights: NDArray = np.random.rand(number_of_instances + 1)  # too many weights

        with self.assertRaises(ValueError):
            accumulate_compute(program, slot_arrays, weights=weights)


if __name__ == '__main__':
    test_main()
