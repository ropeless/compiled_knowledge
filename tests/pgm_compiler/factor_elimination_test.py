from ck.circuit_compiler.llvm_compiler import compile_circuit
from ck.pgm import PGM
from ck.pgm_compiler import factor_elimination
from ck.pgm_circuit import PGMCircuit
from ck.program.program_buffer import ProgramBuffer
from tests.helpers.pgm_test_cases import PGMTestCases

from tests.helpers.unittest_fixture import Fixture, test_main


class TestFactorElimination(Fixture, PGMTestCases):

    def check_pgm(self, pgm: PGM) -> None:
        """
        Compile the PGM and assert the program value is the same as the PGM value product
        for every possible instance.
        """
        pgm_cct: PGMCircuit = factor_elimination.compile_pgm(pgm)
        prog = ProgramBuffer(compile_circuit(pgm_cct.circuit_top))
        slot_map = pgm_cct.slot_map

        for indicators in pgm.instances_as_indicators():
            prog[:] = 0
            for ind in indicators:
                prog[slot_map[ind]] = 1

            instance_as_str = pgm.indicator_str(*indicators)
            program_value = prog.compute().item()
            pgm_value = pgm.value_product_indicators(*indicators)

            self.assertAlmostEqual(program_value, pgm_value, msg=instance_as_str)


if __name__ == '__main__':
    test_main()
