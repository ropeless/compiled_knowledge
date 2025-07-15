import numpy as np

from ck.pgm import PGM
from ck.pgm_circuit import PGMCircuit
from ck.pgm_circuit.program_with_slotmap import ProgramWithSlotmap
from ck.pgm_circuit.support.compile_circuit import compile_results
from ck.pgm_compiler import DEFAULT_PGM_COMPILER
from ck.program import RawProgram, ProgramBuffer
from ck.utils.np_extras import NDArrayNumeric
from tests.helpers.unittest_fixture import Fixture, test_main


def make_program_with_slotmap(pgm: PGM, const_parameters: bool = True) -> ProgramWithSlotmap:
    pgm_circuit: PGMCircuit = DEFAULT_PGM_COMPILER(
        pgm,
        const_parameters=const_parameters,
    )
    raw_program: RawProgram = compile_results(
        pgm_circuit=pgm_circuit,
        results=(pgm_circuit.circuit_top,),
        const_parameters=const_parameters,
    )
    program_with_slotmap = ProgramWithSlotmap(
        program_buffer=ProgramBuffer(raw_program),
        slot_map=pgm_circuit.slot_map,
        rvs=pgm_circuit.rvs,
        precondition=pgm_circuit.conditions,
    )
    return program_with_slotmap


class TestProgramWithSlotMap(Fixture):

    def test_empty(self):
        pgm = PGM()
        program_with_slotmap = make_program_with_slotmap(pgm)

        self.assertEqual(program_with_slotmap.rvs, ())
        self.assertEqual(program_with_slotmap.precondition, ())
        self.assertEmpty(program_with_slotmap.slot_map)
        self.assertEqual(len(program_with_slotmap.vars), 0)

        results: NDArrayNumeric = program_with_slotmap.compute()

        self.assertNDArrayAlmostEqual(results, np.array([1]))
        self.assertNDArrayAlmostEqual(program_with_slotmap.results, np.array([1]))

    def test_set_condition(self):
        pgm = PGM()
        x = pgm.new_rv('x', 2)
        y = pgm.new_rv('y', 2)
        program_with_slotmap = make_program_with_slotmap(pgm)

        self.assertEqual(program_with_slotmap.rvs, (x, y))
        self.assertEqual(program_with_slotmap.precondition, ())
        self.assertEqual(program_with_slotmap.slot_map, {x[0]: 0, x[1]: 1, y[0]: 2, y[1]: 3})
        self.assertEqual(len(program_with_slotmap.vars), 4)

        self.assertNDArrayEqual(program_with_slotmap.vars, np.array([0, 0, 0, 0]))

        program_with_slotmap.set_condition(x[0])
        self.assertNDArrayEqual(program_with_slotmap.vars, np.array([1, 0, 1, 1]))

        program_with_slotmap.set_condition(x[1])
        self.assertNDArrayEqual(program_with_slotmap.vars, np.array([0, 1, 1, 1]))

        program_with_slotmap.set_condition(x[0], x[1])
        self.assertNDArrayEqual(program_with_slotmap.vars, np.array([1, 1, 1, 1]))

        program_with_slotmap.set_condition(x[0], y[1])
        self.assertNDArrayEqual(program_with_slotmap.vars, np.array([1, 0, 0, 1]))

    def test_set_rv(self):
        pgm = PGM()
        x = pgm.new_rv('x', 2)
        y = pgm.new_rv('y', 2)
        program_with_slotmap = make_program_with_slotmap(pgm)

        self.assertEqual(program_with_slotmap.rvs, (x, y))
        self.assertEqual(program_with_slotmap.precondition, ())
        self.assertEqual(program_with_slotmap.slot_map, {x[0]: 0, x[1]: 1, y[0]: 2, y[1]: 3})
        self.assertEqual(len(program_with_slotmap.vars), 4)

        self.assertNDArrayEqual(program_with_slotmap.vars, np.array([0, 0, 0, 0]))

        program_with_slotmap.set_rv(x, 4, 5)
        self.assertNDArrayEqual(program_with_slotmap.vars, np.array([4, 5, 0, 0]))

        program_with_slotmap.set_rv(y, 6, 7)
        self.assertNDArrayEqual(program_with_slotmap.vars, np.array([4, 5, 6, 7]))

    def test_set_rvs_uniform(self):
        pgm = PGM()
        x = pgm.new_rv('x', 2)
        y = pgm.new_rv('y', 2)
        program_with_slotmap = make_program_with_slotmap(pgm)

        self.assertEqual(program_with_slotmap.rvs, (x, y))
        self.assertEqual(program_with_slotmap.precondition, ())
        self.assertEqual(program_with_slotmap.slot_map, {x[0]: 0, x[1]: 1, y[0]: 2, y[1]: 3})
        self.assertEqual(len(program_with_slotmap.vars), 4)

        self.assertNDArrayEqual(program_with_slotmap.vars, np.array([0, 0, 0, 0]))

        program_with_slotmap.set_rvs_uniform(x)
        self.assertNDArrayEqual(program_with_slotmap.vars, np.array([0.5, 0.5, 0, 0]))

        program_with_slotmap.set_rv(x, 0, 0)
        self.assertNDArrayEqual(program_with_slotmap.vars, np.array([0, 0, 0, 0]))

        program_with_slotmap.set_rvs_uniform(y)
        self.assertNDArrayEqual(program_with_slotmap.vars, np.array([0, 0, 0.5, 0.5]))

        program_with_slotmap.set_rv(y, 0, 0)
        self.assertNDArrayEqual(program_with_slotmap.vars, np.array([0, 0, 0, 0]))

        program_with_slotmap.set_all_rvs_uniform()
        self.assertNDArrayEqual(program_with_slotmap.vars, np.array([0.5, 0.5, 0.5, 0.5]))

    def test_set_and_get(self):
        pgm = PGM()
        x = pgm.new_rv('x', 2)
        y = pgm.new_rv('y', 2)
        program_with_slotmap = make_program_with_slotmap(pgm)

        self.assertEqual(program_with_slotmap.rvs, (x, y))
        self.assertEqual(program_with_slotmap.precondition, ())
        self.assertEqual(program_with_slotmap.slot_map, {x[0]: 0, x[1]: 1, y[0]: 2, y[1]: 3})
        self.assertEqual(len(program_with_slotmap.vars), 4)

        self.assertNDArrayEqual(program_with_slotmap.vars, np.array([0, 0, 0, 0]))

        program_with_slotmap[1] = 7
        self.assertNDArrayEqual(program_with_slotmap.vars, np.array([0, 7, 0, 0]))

        program_with_slotmap[1] = 0
        self.assertNDArrayEqual(program_with_slotmap.vars, np.array([0, 0, 0, 0]))

        program_with_slotmap[1:3] = 5
        self.assertNDArrayEqual(program_with_slotmap.vars, np.array([0, 5, 5, 0]))

        program_with_slotmap[1:3] = 3, 2
        self.assertNDArrayEqual(program_with_slotmap.vars, np.array([0, 3, 2, 0]))

        program_with_slotmap[y[1]] = 9
        self.assertNDArrayEqual(program_with_slotmap.vars, np.array([0, 3, 2, 9]))

        self.assertEqual(program_with_slotmap[x[0]], 0)
        self.assertEqual(program_with_slotmap[x[1]], 3)
        self.assertEqual(program_with_slotmap[2], 2)
        self.assertEqual(program_with_slotmap[3], 9)

        self.assertArrayEqual(program_with_slotmap[x], [0, 3])
        self.assertArrayEqual(program_with_slotmap[y], [2, 9])

        program_with_slotmap[y] = 1
        self.assertNDArrayEqual(program_with_slotmap.vars, np.array([0, 3, 1, 1]))


if __name__ == '__main__':
    test_main()
