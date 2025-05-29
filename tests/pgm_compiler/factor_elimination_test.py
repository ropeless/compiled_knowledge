from ck.circuit_compiler.llvm_compiler import compile_circuit
from ck.pgm import PGM
from ck.pgm_compiler import factor_elimination
from ck.pgm_circuit import PGMCircuit
from ck.program.program_buffer import ProgramBuffer
from ck.utils.iter_extras import take
from tests.helpers.pgm_test_cases import PGMTestCases

from tests.helpers.unittest_fixture import Fixture, test_main

LIMIT_INSTANCES_CHECK: int = 1_000_000


class TestFactorElimination(Fixture, PGMTestCases):

    def check_pgm(self, pgm: PGM) -> None:
        """
        Compile the PGM and assert the program value is the same as the PGM value product
        for every possible instance.
        """
        pgm_cct: PGMCircuit = factor_elimination.compile_pgm(pgm)
        prog = ProgramBuffer(compile_circuit(pgm_cct.circuit_top))
        slot_map = pgm_cct.slot_map

        for instance_indicators in take(pgm.instances_as_indicators(), LIMIT_INSTANCES_CHECK):
            prog[:] = 0
            for ind in instance_indicators:
                prog[slot_map[ind]] = 1

            msg = f'{pgm.name!r} {pgm.indicator_str(*instance_indicators)}'
            program_value = prog.compute().item()
            pgm_value = pgm.value_product_indicators(*instance_indicators)

            self.assertAlmostEqual(program_value, pgm_value, msg=msg)


if __name__ == '__main__':
    test_main()
