"""
Demo script to show the foundational processes PGM compilation
and inference.
"""

from ck.pgm import PGM
from ck.pgm_circuit import PGMCircuit
from ck.pgm_circuit.support.compile_circuit import DEFAULT_CIRCUIT_COMPILER
from ck.pgm_circuit.wmc_program import WMCProgram
from ck.pgm_compiler import DEFAULT_PGM_COMPILER
from ck.program import RawProgram, ProgramBuffer

pgm = PGM()

# =============================================================
# Define PGM random variables
# =============================================================
A = pgm.new_rv('A', 2)
B = pgm.new_rv('B', 2)
C = pgm.new_rv('C', 2)
D = pgm.new_rv('D', 3)

# =============================================================
# Show the possible worlds
# =============================================================
print()
print("The PGM worlds...")
print(tuple(str(rv) for rv in pgm.rvs))
for world in pgm.instances():
    print(world)

# =============================================================
# Add some PGM factors
# (configure each factor to have a dense potential function)
# =============================================================
factor_AB = pgm.new_factor(A, B)
factor_BCD = pgm.new_factor(B, C, D)
factor_B = pgm.new_factor(B)

f_AB = factor_AB.set_dense()
f_BCD = factor_BCD.set_dense()
f_B = factor_B.set_dense()

f_AB[0, 0] = 0.9
f_AB[1, 0] = 0.8
f_AB[0, 1] = 0.1
f_AB[1, 1] = 0.2

f_BCD[0, 0, 0] = 0
f_BCD[1, 0, 0] = 1
f_BCD[0, 1, 0] = 1
f_BCD[1, 1, 0] = 0
f_BCD[0, 0, 1] = 1
f_BCD[1, 0, 1] = 0
f_BCD[0, 1, 1] = 0
f_BCD[1, 1, 1] = 1
f_BCD[0, 0, 2] = 0
f_BCD[1, 0, 2] = 1
f_BCD[0, 1, 2] = 1
f_BCD[1, 1, 2] = 0

f_B[0] = 2.3
f_B[1] = 4.5

# =============================================================
# Show the weight of each possible world
# =============================================================
print()
print('The PGM world factors and weights...')
z = 0
print(tuple(str(rv) for rv in pgm.rvs), 'weights...')
for world in pgm.instances():
    weight = pgm.value_product(world)
    print(world, list(pgm.factor_values(world)), weight)
    z += weight
print('z =', z)

# =============================================================
# Getting and printing indicators
# =============================================================
print()

A_is_0 = A[0]  # Get the indicator for A=0
print('Indicator for A = 0:', A_is_0)

A_is_0_as_str = pgm.indicator_str(A[0])  # Get the indicator in human-readable form
print('Indicator string for A = 0:', A_is_0_as_str)

print('A possible world:', pgm.indicator_str(A[0], B[1], C[0], D[2]))

# =============================================================
# Convert the PGM into an arithmetic circuit
# =============================================================

pgm_cct: PGMCircuit = DEFAULT_PGM_COMPILER(pgm)  # Compiles the PGM into a circuit

# Get a reference to the circuit object
cct = pgm_cct.circuit_top.circuit

print()
print('Dump of the arithmetic circuit')
cct.dump()

# =============================================================
# Compile the arithmetic circuit into a program
# =============================================================

# Get a reference to the top node of the arithmetic circuit
cct_top = pgm_cct.circuit_top

# Compile the circuit into a program
raw_program: RawProgram = DEFAULT_CIRCUIT_COMPILER(cct_top)
program = ProgramBuffer(raw_program)

# =============================================================
# Use the program to compute the partition function, z
# =============================================================

program[:] = 1  # set all input slots to 1
z = program.compute()  # run the program

print()
print('Program computed z =', z)

# =============================================================
# Use the program to compute the weight of A=0,B=1,C=0,D=0
# =============================================================

program[:] = 0  # set all input slots to 0
program[0] = 1  # A=0
program[3] = 1  # B=1
program[4] = 1  # C=0
program[6] = 1  # D=0
w = program.compute()

print()
print('Program computed w(A=0, B=1, C=0, D=0) =', w)

# =============================================================
# Use the program to compute the probability of A=0 and B=1
# =============================================================

program[:] = 1  # set all input slots to 1
program[1] = 0  # exclude A=1
program[2] = 0  # exclude B=0
w = program.compute()

print()
print('Program computed P(A=0, B=1) =', (w / z))

# =============================================================
# Using a slot map
# =============================================================

slots = pgm_cct.slot_map  # get the slot map

program[:] = 1  # set all input slots to 1
program[slots[B[1]]] = 0  # exclude B=1
program[slots[C[0]]] = 0  # exclude C=0
w = program.compute()

print()
print('Program computed P(B=0, C=1) =', (w / z))

# =============================================================
# Using a WMCProgram object with built in slot map
# =============================================================

# Compile the circuit as a WMCProgram object
wmc = WMCProgram(pgm_cct)

wmc[:] = 0  # set all input slots to 0
wmc[B[1]] = 1  # set B=1 to 1
wmc[D[2]] = 1  # set C=2 to 1
wmc[A] = 1  # set all A indicators to 1
wmc[C] = 1  # set all C indicators to 1

print()
print('Program computed P(B=1, D=2) =', (wmc.compute() / z))

# Here is the easy way to do it
print()
print('Program computed P(B=0, D=2) =', wmc.probability(B[0], D[2]))
print('Program computed P(B=1, D=2) =', wmc.probability(B[1], D[2]))
print('Program computed P(D=2) =', wmc.probability(D[2]))

print()
print('Done.')
