"""
Demonstrate how to do MEP inference on a PGM
"""

from ck.pgm import PGM
from ck.pgm_circuit import PGMCircuit
from ck.pgm_circuit.mpe_program import MPEProgram, MPEResult
from ck.pgm_compiler import DEFAULT_PGM_COMPILER

pgm = PGM()
pollution = pgm.new_rv('pollution', ('low', 'high'))
smoker = pgm.new_rv('smoker', ('true', 'false'))
cancer = pgm.new_rv('cancer', ('true', 'false'))
xray = pgm.new_rv('xray', ('positive', 'negative'))
dyspnoea = pgm.new_rv('dyspnoea', ('true', 'false'))
pgm.new_factor(pollution).set_dense().set_flat(0.9, 0.1)
pgm.new_factor(smoker).set_dense().set_flat(0.3, 0.7)
pgm.new_factor(cancer, pollution, smoker).set_dense().set_flat(0.03, 0.001, 0.05, 0.02, 0.97, 0.999, 0.95, 0.98)
pgm.new_factor(xray, cancer).set_dense().set_flat(0.9, 0.2, 0.1, 0.8)
pgm.new_factor(dyspnoea, cancer).set_dense().set_flat(0.65, 0.3, 0.35, 0.7)

pgm_cct: PGMCircuit = DEFAULT_PGM_COMPILER(pgm)
mpe = MPEProgram(pgm_cct)


# How we will render an mpe_result
def mpe_str(_mpe_result: MPEResult) -> str:
    global pgm
    return ', '.join([
        pgm.indicator_str(rv[state])
        for rv, state in zip(mpe.trace_rvs, _mpe_result.mpe)
    ])


# What is the most likely situation (unconditioned MPE)
mpe_result: MPEResult = mpe.mpe()
print(mpe_str(mpe_result))

# What is the MPE given smoker = true
mpe_result: MPEResult = mpe.mpe(smoker('true'))
print(mpe_str(mpe_result))

# What is the MPE given pollution = high and xray = negative
mpe_result: MPEResult = mpe.mpe(pollution('high'), xray('negative'))
print(mpe_str(mpe_result))
