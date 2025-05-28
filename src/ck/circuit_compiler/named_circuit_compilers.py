from enum import Enum
from functools import partial
from typing import Optional

from .llvm_compiler import Flavour
from ..circuit import CircuitNode, Circuit
from ..circuit_compiler import interpret_compiler, cython_vm_compiler, llvm_compiler, llvm_vm_compiler, CircuitCompiler
from ..circuit_compiler.support.input_vars import InputVars, InferVars
from ..program import RawProgram


class NamedCircuitCompiler(Enum):
    """
    A standard collection of named circuit compiler functions.

    The `value` of each enum member is tuple containing a compiler function.
    Wrapping in a tuple is needed otherwise Python erases the type of the member, which can cause problems.
    Each member itself is callable, conforming to the CircuitCompiler protocol, delegating to the compiler function.
    """

    LLVM_STACK = (partial(llvm_compiler.compile_circuit, flavour=Flavour.STACK),)
    LLVM_TMPS = (partial(llvm_compiler.compile_circuit, flavour=Flavour.TMPS, opt=0),)
    LLVM_VM = (llvm_vm_compiler.compile_circuit,)
    CYTHON_VM = (cython_vm_compiler.compile_circuit,)
    INTERPRET = (interpret_compiler.compile_circuit,)

    # The following circuit compilers were experimental but are not really useful.
    #
    # Slow compile and execution:
    # LLVM_FUNCS = (partial(llvm_compiler.compile_circuit, flavour=Flavour.FUNCS, opt=0),)
    #
    # Slow compile and same execution as LLVM_VM:
    # LLVM_VM_COMPILED_ARRAYS = (partial(llvm_vm_compiler.compile_circuit, compile_arrays=True),)

    def __call__(
            self,
            *result: CircuitNode,
            input_vars: InputVars = InferVars.ALL,
            circuit: Optional[Circuit] = None,
    ) -> RawProgram:
        """
        Each member of the enum is a CircuitCompiler function.

        This implements the `CircuitCompiler` protocol for each member of the enum.
        """
        return self.compiler(*result, input_vars=input_vars, circuit=circuit)

    @property
    def compiler(self) -> CircuitCompiler:
        """
        Returns:
            The compiler function, conforming to the CircuitCompiler protocol.
        """
        return self.value[0]


DEFAULT_CIRCUIT_COMPILER: NamedCircuitCompiler = NamedCircuitCompiler.CYTHON_VM
