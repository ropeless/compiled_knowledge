from ck import example
from ck.pgm import RVMap
from ck.pgm_circuit.wmc_program import WMCProgram
from ck.pgm_compiler import DEFAULT_PGM_COMPILER

# create the example "Cancer" Bayesian network
pgm = example.Cancer()

# compile the PGM and construct an object for probabilistic queries
wmc = WMCProgram(DEFAULT_PGM_COMPILER(pgm))

# provide easy access to the random variables - not needed but simplifies this demo
rvs = RVMap(pgm)

# get the probability of having cancer given that pollution is high
pr = wmc.probability(rvs.cancer('True'), condition=rvs.pollution('high'))

print('probability of having cancer given that pollution is high =', pr)
