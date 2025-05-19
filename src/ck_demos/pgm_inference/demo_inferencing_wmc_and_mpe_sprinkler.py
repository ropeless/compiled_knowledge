"""
This is a simple demo creating then using a PGM model.
"""

from ck.pgm import PGM
from ck.pgm_circuit import PGMCircuit
from ck.pgm_circuit.mpe_program import MPEProgram
from ck.pgm_circuit.wmc_program import WMCProgram
from ck.pgm_compiler import DEFAULT_PGM_COMPILER
from ck_demos.utils.stop_watch import timer

pgm = PGM()

rain = pgm.new_rv('rain', ['no', 'yes'])
sprinkler = pgm.new_rv('sprinkler', ['off', 'on'])
grass = pgm.new_rv('grass', ['dry', 'damp', 'wet'])

f_g = pgm.new_factor(grass, rain, sprinkler)  # same as a Conditional Probability Table (CPT)
f_r = pgm.new_factor(rain)
f_s = pgm.new_factor(sprinkler)

# Instead of learning the parameter values from data, in this simple
# demo we will hard code the parameter values.
f_r.set_dense().set_flat(0.8, 0.2)
f_s.set_dense().set_flat(0.9, 0.1)
f_g.set_dense().set_flat(
    # not raining  raining       # rain
    # off on       off   on      # sprinkler
    0.90, 0.01,    0.02, 0.01,   # grass dry
    0.09, 0.01,    0.08, 0.04,   # grass damp
    0.01, 0.98,    0.90, 0.95,   # grass wet
)

# ---------------------------------------
#  Directly use the model
# ---------------------------------------

print('is_structure_bayesian', pgm.is_structure_bayesian)
print('factors_are_cpts', pgm.factors_are_cpts())

for inst in pgm.instances():
    inst_str = ','.join(f'{rv}={state}' for rv, state in zip(pgm.rvs, inst))
    inst_w = pgm.value_product(inst)
    print(f'weight({inst_str}) = {inst_w}')

# ---------------------------------------
#  Compile the PGM for complex queries
# ---------------------------------------

with timer('compiling PGM'):
    pgm_cct: PGMCircuit = DEFAULT_PGM_COMPILER(pgm)
    wmc = WMCProgram(pgm_cct)
circuit = pgm_cct.circuit_top.circuit
print(f'number of ops: {circuit.number_of_op_nodes}')
print(f'number of arcs: {circuit.number_of_arcs}')

# ---------------------------------------
#  Here are some example uses of a model
# ---------------------------------------

print()
print('--------')
print('Show selected probabilities')

# What is the probability of 'not raining' and 'sprinkler off' and 'grass damp'?
pr001 = wmc.probability(rain[0], sprinkler[0], grass[1])
print(pgm.indicator_str(rain[0], sprinkler[0], grass[1]), 'Pr =', pr001)

# What is the probability of 'not raining' and 'sprinkler on' and 'grass damp'?
pr011 = wmc.probability(rain[0], sprinkler[1], grass[1])
print(pgm.indicator_str(rain[0], sprinkler[1], grass[1]), 'Pr =', pr011)

# What is the probability of 'not raining' and 'grass damp'?
pr0_1 = wmc.probability(rain[0], grass[1])
print(pgm.indicator_str(rain[0], grass[1]), 'Pr =', pr0_1)

# What is the probability of 'grass damp'?
pr__1 = wmc.probability(grass[1])
print(pgm.indicator_str(grass[1]), 'Pr =', pr__1)

# What is the probability of 'raining' given 'grass wet'?
pr_val = wmc.probability(rain[1], condition=grass[2])
pr_str  = pgm.indicator_str(rain[1])
pr_cond = pgm.indicator_str(grass[2])
print(f'Pr[{pr_str} | {pr_cond}] = {pr_val}')


# Show the probability of each possible world
print('--------')
print('Show every possible world and its probability')
for inst in pgm.instances_as_indicators():
    pr_str = pgm.indicator_str(*inst)
    pr_val = wmc.probability(*inst)
    print(f'Pr[{pr_str}] = {pr_val}')

print('--------')
print('Show selected marginal distributions')

# What is the marginal probability distribution over sprinkler?
pr_sprinkler = wmc.marginal_distribution(sprinkler)
print(f'Pr[{sprinkler}] = {pr_sprinkler}')

# What is the marginal probability distribution over sprinkler,
# given 'not raining' and 'grass wet'?
condition = (rain[0], grass[2])
print('Pr[{} | {}] = {}'.format(
    sprinkler, pgm.indicator_str(*condition),
    wmc.marginal_distribution(sprinkler, condition=condition)
))

# What is the marginal probability distribution over sprinkler,
# given 'raining' and 'grass wet'?
condition = (rain[1], grass[2])
print('Pr[{} | {}] = {}'.format(
    sprinkler,
    pgm.indicator_str(*condition),
    wmc.marginal_distribution(sprinkler, condition=condition)
))


print('--------')
print('Show selected MAP results')

pr, states = wmc.map(sprinkler, rain)
print(f'MAP[{sprinkler}, {rain}] = {states} with probability = {pr}')

pr, states = wmc.map(sprinkler, grass, condition=[grass[1], rain[0]])
cond_str = pgm.indicator_str(grass[1], rain[0])
print(f'MAP[{sprinkler}, {grass} | {cond_str}] = {states} with probability = {pr}')


print('--------')
print('Show selected MPE results')

z = wmc.z
mpe = MPEProgram(pgm_cct)

mpe_result = mpe.mpe()
print('MPE = {} with probability = {}'.format(mpe_result.mpe, mpe_result.wmc/z))

mpe_result = mpe.mpe(grass[2])
print('given wet grass: MPE = {} with probability = {}'.format(mpe_result.mpe, mpe_result.wmc/z))

print('--------')
print('Draw 20 independent samples from the probability distribution')
num_samples = 20
sampler = wmc.sample_direct()
for i, sample in zip(range(1, 1 + num_samples), sampler):
    sample_indicators = pgm.state_idxs_to_indicators(sample)
    ind_str = pgm.indicator_str(*sample_indicators)
    print(f'{i:2} {sample} {ind_str}')

print('--------')
print('Done.')
