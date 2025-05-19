from typing import Sequence

import numpy as np

from ck.circuit import Circuit
from ck.circuit_compiler.llvm_compiler import compile_circuit
from ck.pgm import PGM, RandomVariable
from ck.pgm_circuit.program_with_slotmap import ProgramWithSlotmap
from ck.pgm_circuit.slot_map import SlotMap
from ck.probability.probability_space import Condition, check_condition
from ck.program import ProgramBuffer, RawProgram
from ck.sampling.sampler_support import SamplerInfo, get_sampler_info
from ck.utils.iter_extras import flatten
from tests.helpers.unittest_fixture import Fixture, test_main


class DummyProgramWithSlotmap(ProgramWithSlotmap):
    def __init__(
            self,
            rvs: Sequence[RandomVariable],
            precondition: Condition = None,
    ):
        slot_map: SlotMap = {
            ind: i
            for i, ind in enumerate(flatten(rvs))
        }

        circuit: Circuit = Circuit()
        circuit.new_vars(len(slot_map))
        top = circuit.add(circuit.vars)

        program: RawProgram = compile_circuit(top)
        program_buffer: ProgramBuffer = ProgramBuffer(program)
        super().__init__(program_buffer, slot_map, rvs, check_condition(precondition))


class TestSamplerInfo(Fixture):

    def test_yield_single(self):
        pgm: PGM = PGM()
        for i in range(10):
            pgm.new_rv(f'x_{i}', 3)
        rvs = pgm.rvs

        sampler_info: SamplerInfo = get_sampler_info(
            program_with_slotmap=DummyProgramWithSlotmap(rvs=rvs),
            rvs=rvs[0],  # single rv
            condition=(),
        )

        yielded = sampler_info.yield_f(np.array([123]))
        self.assertEqual(yielded, 123)

    def test_yield_singleton(self):
        pgm: PGM = PGM()
        for i in range(10):
            pgm.new_rv(f'x_{i}', 3)
        rvs = pgm.rvs

        sampler_info: SamplerInfo = get_sampler_info(
            program_with_slotmap=DummyProgramWithSlotmap(rvs=rvs),
            rvs=[rvs[0]],  # singleton
            condition=(),
        )

        yielded = sampler_info.yield_f(np.array([123]))
        self.assertArrayEqual(yielded, [123])

    def test_yield_multiple(self):
        pgm: PGM = PGM()
        for i in range(10):
            pgm.new_rv(f'x_{i}', 3)
        rvs = pgm.rvs

        sampler_info: SamplerInfo = get_sampler_info(
            program_with_slotmap=DummyProgramWithSlotmap(rvs=rvs),
            rvs=[rvs[0], rvs[1]],
            condition=(),
        )

        yielded = sampler_info.yield_f(np.array([123, 456]))
        self.assertArrayEqual(yielded, [123, 456])

    def test_precondition_only(self):
        pgm: PGM = PGM()
        for i in range(10):
            pgm.new_rv(f'x_{i}', 3)
        rvs = pgm.rvs

        sampler_info: SamplerInfo = get_sampler_info(
            program_with_slotmap=DummyProgramWithSlotmap(rvs=rvs, precondition=rvs[3][0]),
            rvs=rvs,
            condition=(),
        )

        self.assertArraySetEqual(sampler_info.condition, [rvs[3][0]])

    def test_condition_only(self):
        pgm: PGM = PGM()
        for i in range(10):
            pgm.new_rv(f'x_{i}', 3)
        rvs = pgm.rvs

        sampler_info: SamplerInfo = get_sampler_info(
            program_with_slotmap=DummyProgramWithSlotmap(rvs=rvs),
            rvs=rvs,
            condition=rvs[2][1],
        )

        self.assertArraySetEqual(sampler_info.condition, [rvs[2][1]])

    def test_precondition_with_condition(self):
        pgm: PGM = PGM()
        for i in range(10):
            pgm.new_rv(f'x_{i}', 3)
        rvs = pgm.rvs

        sampler_info: SamplerInfo = get_sampler_info(
            program_with_slotmap=DummyProgramWithSlotmap(rvs=rvs, precondition=rvs[3][0]),
            rvs=rvs,
            condition=rvs[2][1],
        )

        self.assertArraySetEqual(sampler_info.condition, [rvs[2][1], rvs[3][0]])

    def test_conditioning_overlap(self):
        pgm: PGM = PGM()
        for i in range(10):
            pgm.new_rv(f'x_{i}', 3)
        rvs = pgm.rvs

        sampler_info: SamplerInfo = get_sampler_info(
            program_with_slotmap=DummyProgramWithSlotmap(rvs=rvs, precondition=(rvs[3][0], rvs[3][1])),
            rvs=rvs,
            condition=(rvs[3][1], rvs[3][2]),
        )
        self.assertArraySetEqual(sampler_info.condition, [rvs[3][1]])

    def test_conditioning_disjoint(self):
        pgm: PGM = PGM()
        for i in range(10):
            pgm.new_rv(f'x_{i}', 3)
        rvs = pgm.rvs

        with self.assertRaises(ValueError):
            _ = get_sampler_info(
                program_with_slotmap=DummyProgramWithSlotmap(rvs=rvs, precondition=rvs[3][0]),
                rvs=rvs,
                condition=rvs[3][1],
            )


if __name__ == '__main__':
    test_main()
