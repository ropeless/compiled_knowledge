import io

import numpy as np

from ck.circuit import Circuit
from ck.pgm import PGM
from ck.pgm_circuit import PGMCircuit
from tests.helpers.redirect_out import redirect_out
from tests.helpers.string_extras import unindent
from tests.helpers.unittest_fixture import Fixture, test_main


class TestPGMCircuit(Fixture):
    """
    These are tests for PGMCircuit.
    """

    def test_dump(self):
        # This test merely makes sure that circuit variables get named correctly when dumped.

        pgm = PGM()
        x = pgm.new_rv('x', 2)
        y = pgm.new_rv('y', 2)

        parameters = [23, 45]

        cct = Circuit()
        cct_vars = cct.new_vars(pgm.number_of_indicators + len(parameters))
        top = cct.add(cct_vars)

        slotmap = {
            x[0]: 0,
            x[1]: 1,
            y[0]: 2,
            y[1]: 3,
            'irrelevant': 9999,  # This should be ignored.
        }

        pgm_cct = PGMCircuit(
            pgm.rvs,
            (),
            top,
            pgm.number_of_indicators,
            len(parameters),
            slotmap,
            np.array(parameters),
        )

        expect = unindent(
            """
            number of vars: 6
            number of const nodes: 2
            number of op nodes: 1
            number of operations: 5
            number of arcs: 6
            var nodes: 6
                var[0]: 'x'[0] 0
                var[1]: 'x'[1] 1
                var[2]: 'y'[0] 0
                var[3]: 'y'[1] 1
                var[4]: param[0] = 23
                var[5]: param[1] = 45
            op nodes: 1 (arcs: 6, ops: 5)
                add<0>: var[0] var[1] var[2] var[3] var[4] var[5]
            """
        )

        output = io.StringIO()
        with redirect_out(output):
            pgm_cct.dump()
        capture = output.getvalue()

        self.assertEqual(capture, expect)


if __name__ == '__main__':
    test_main()
