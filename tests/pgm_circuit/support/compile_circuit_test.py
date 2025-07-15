import numpy as np

from ck.circuit import Circuit
from ck.pgm import PGM, ParamId
from ck.pgm_circuit import PGMCircuit
from ck.pgm_circuit.slot_map import SlotMap
from ck.pgm_circuit.support.compile_circuit import compile_results
from ck.program import RawProgram
from tests.helpers.unittest_fixture import Fixture, test_main


class TestCompileCircuit(Fixture):

    def test_no_parameters(self):
        pgm = PGM()
        x = pgm.new_rv('x', 2)
        y = pgm.new_rv('y', 2)

        parameters = []

        cct = Circuit()
        cct_vars = cct.new_vars(pgm.number_of_indicators + len(parameters))
        top = cct.mul(cct_vars)

        slot_map: SlotMap = {
            x[0]: 0,
            x[1]: 1,
            y[0]: 2,
            y[1]: 3,
        }

        pgm_circuit = PGMCircuit(
            rvs=pgm.rvs,
            conditions=(),
            circuit_top=top,
            number_of_indicators=pgm.number_of_indicators,
            number_of_parameters=len(parameters),
            slot_map=slot_map,
            parameter_values=np.array(parameters),
        )

        program: RawProgram = compile_results(
            pgm_circuit=pgm_circuit,
            results=[top],
            const_parameters=False,  # <<<<<
        )

        self.assertArrayEqual(program([2, 3, 5, 7]), [210])

    def test_no_parameters_const(self):
        pgm = PGM()
        x = pgm.new_rv('x', 2)
        y = pgm.new_rv('y', 2)

        parameters = []

        cct = Circuit()
        cct_vars = cct.new_vars(pgm.number_of_indicators + len(parameters))
        top = cct.mul(cct_vars)

        slot_map: SlotMap = {
            x[0]: 0,
            x[1]: 1,
            y[0]: 2,
            y[1]: 3,
        }

        pgm_circuit = PGMCircuit(
            rvs=pgm.rvs,
            conditions=(),
            circuit_top=top,
            number_of_indicators=pgm.number_of_indicators,
            number_of_parameters=len(parameters),
            slot_map=slot_map,
            parameter_values=np.array(parameters),
        )

        program: RawProgram = compile_results(
            pgm_circuit=pgm_circuit,
            results=[top],
            const_parameters=True,  # <<<<<
        )

        self.assertArrayEqual(program([2, 3, 5, 7]), [210])

    def test_free_parameters(self):
        pgm = PGM()
        x = pgm.new_rv('x', 2)
        y = pgm.new_rv('y', 2)

        parameters = [23, 45]

        cct = Circuit()
        cct_vars = cct.new_vars(pgm.number_of_indicators + len(parameters))
        top = cct.mul(cct_vars)

        slot_map: SlotMap = {
            x[0]: 0,
            x[1]: 1,
            y[0]: 2,
            y[1]: 3,
            ParamId(0, 0): 4,
            ParamId(0, 1): 5,
        }

        pgm_circuit = PGMCircuit(
            rvs=pgm.rvs,
            conditions=(),
            circuit_top=top,
            number_of_indicators=pgm.number_of_indicators,
            number_of_parameters=len(parameters),
            slot_map=slot_map,
            parameter_values=np.array(parameters),
        )

        program: RawProgram = compile_results(
            pgm_circuit=pgm_circuit,
            results=[top],
            const_parameters=False,  # <<<<<
        )

        self.assertArrayEqual(program([2, 3, 5, 7, 13, 17]), [46410])

    def test_const_parameters(self):
        pgm = PGM()
        x = pgm.new_rv('x', 2)
        y = pgm.new_rv('y', 2)

        parameters = [23, 45]

        cct = Circuit()
        cct_vars = cct.new_vars(pgm.number_of_indicators + len(parameters))
        top = cct.mul(cct_vars)

        slot_map: SlotMap = {
            x[0]: 0,
            x[1]: 1,
            y[0]: 2,
            y[1]: 3,
            ParamId(0, 0): 4,
            ParamId(0, 1): 5,
        }

        pgm_circuit = PGMCircuit(
            rvs=pgm.rvs,
            conditions=(),
            circuit_top=top,
            number_of_indicators=pgm.number_of_indicators,
            number_of_parameters=len(parameters),
            slot_map=slot_map,
            parameter_values=np.array(parameters),
        )

        program: RawProgram = compile_results(
            pgm_circuit=pgm_circuit,
            results=[top],
            const_parameters=True,  # <<<<<
        )

        self.assertArrayEqual(program([2, 3, 5, 7, 13, 17]), [217350])


if __name__ == '__main__':
    test_main()
