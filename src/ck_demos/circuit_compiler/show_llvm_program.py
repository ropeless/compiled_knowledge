from ck import example
from ck.circuit import CircuitNode
from ck.circuit_compiler import llvm_vm_compiler
from ck.circuit_compiler.support.llvm_ir_function import LLVMRawProgram
from ck.pgm import PGM
from ck.pgm_circuit import PGMCircuit
from ck.pgm_compiler import NamedPGMCompiler, PGMCompiler

EXAMPLE_PGM: PGM = example.Rain()
PGM_COMPILER: PGMCompiler = NamedPGMCompiler.FE_BEST_JOINTREE


def main() -> None:
    pgm_cct: PGMCircuit = PGM_COMPILER(EXAMPLE_PGM)
    top: CircuitNode = pgm_cct.circuit_top

    program: LLVMRawProgram = llvm_vm_compiler.compile_circuit(top)

    print(program.llvm_program)

    print()
    print('Done.')


if __name__ == '__main__':
    main()
