from ck.circuit_compiler.llvm_compiler import compile_circuit
from ck.pgm import PGM
from ck.pgm_circuit import PGMCircuit
from ck.pgm_compiler.ace import ace
from ck.program.program_buffer import ProgramBuffer
from tests.helpers.pgm_test_cases import BNTestCases
from tests.helpers.unittest_fixture import Fixture, test_main

# If `MUST_HAVE_ACE_INSTALLED` is True, then we expect ACE to be installed at the default location in CK.
MUST_HAVE_ACE_INSTALLED: bool = False

if MUST_HAVE_ACE_INSTALLED:
    class TestAceAvailable(Fixture):
        def test_ace_available(self):
            self.assertTrue(ace.ace_available())

if ace.ace_available():

    class TestAce(Fixture, BNTestCases):

        def check_pgm(self, pgm: PGM) -> None:
            """
            Compile the PGM and assert the program value is the same as the PGM value product
            for every possible instance.
            """

            pgm_cct: PGMCircuit = ace.compile_pgm(pgm)
            prog = ProgramBuffer(compile_circuit(pgm_cct.circuit_top))
            slot_map = pgm_cct.slot_map

            for indicators in pgm.instances_as_indicators():
                prog[:] = 0
                for ind in indicators:
                    prog[slot_map[ind]] = 1

                msg = f'{pgm.name!r} {pgm.indicator_str(*indicators)}'
                program_value = prog.compute().item()
                pgm_value = pgm.value_product_indicators(*indicators)

                self.assertAlmostEqual(program_value, pgm_value, msg=msg)

if __name__ == '__main__':
    test_main()
