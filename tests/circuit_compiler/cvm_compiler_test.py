from typing import Optional

import numpy as np

from ck.circuit import Circuit, CircuitNode
from ck.circuit_compiler import cython_vm_compiler
from ck.circuit_compiler.support.input_vars import InputVars, InferVars
from ck.program import RawProgram
from tests.helpers.circuit_compiler_test_cases import CompilerCases
from tests.helpers.unittest_fixture import Fixture, test_main


class TestCVMCompilerFloat64(Fixture, CompilerCases):

    def compile_circuit(
            self,
            *result: CircuitNode,
            input_vars: InputVars = InferVars.ALL,
            circuit: Optional[Circuit] = None,
    ) -> RawProgram:
        return cython_vm_compiler.compile_circuit(
            *result,
            input_vars=input_vars,
            circuit=circuit,
            dtype=np.float64,
        )



if __name__ == '__main__':
    test_main()
