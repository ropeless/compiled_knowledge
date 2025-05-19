"""
This is a simple demo creating then using a PGM model.
"""

from ck.pgm import PGM, rv_instances_as_indicators, Indicator, RandomVariable
from ck.pgm_circuit.wmc_program import WMCProgram
from ck.pgm_compiler import DEFAULT_PGM_COMPILER
from ck.probability.probability_space import Condition, check_condition

# -------------------------------------------------------------------------
# Construct the 'student' network from
# Koller & Friedman, Probabilistic Graphical Models, 2009, Figure 3.4, p53.
# -------------------------------------------------------------------------

pgm = PGM()

difficult = pgm.new_rv('difficult', ('Yes', 'No'))
intelligent = pgm.new_rv('intelligent', ('Yes', 'No'))
grade = pgm.new_rv('grade', ('1', '2', '3'))
sat = pgm.new_rv('sat', ('High', 'Low'))
letter = pgm.new_rv('letter', ('Yes', 'No'))

pgm.new_factor(difficult).set_cpt().set_all(
    (0.6, 0.4),
)
pgm.new_factor(intelligent).set_cpt().set_all(
    (0.7, 0.3),
)
pgm.new_factor(grade, difficult, intelligent).set_cpt().set_all(
    (0.3, 0.4, 0.3),
    (0.05, 0.25, 0.7),
    (0.9, 0.08, 0.02),
    (0.5, 0.3, 0.2)
)
pgm.new_factor(sat, intelligent).set_cpt().set_all(
    (0.95, 0.05),
    (0.2, 0.8),
)
pgm.new_factor(letter, grade).set_cpt().set_all(
    (0.1, 0.9),
    (0.4, 0.6),
    (0.99, 0.01),
)

# ---------------------------------------
#  Compile the PGM
# ---------------------------------------

wmc = WMCProgram(DEFAULT_PGM_COMPILER(pgm))


# ---------------------------------------
#  Here are some example uses of a model
# ---------------------------------------

def show_probability(*indicators: Indicator, condition: Condition = ()):
    """
    Print the probability of the given indicators,
    with an optional set of condition indicators.
    """
    condition = check_condition(condition)
    indicator_str = pgm.indicator_str(*indicators)
    probability = wmc.probability(*indicators, condition=condition)
    if len(condition) == 0:
        print(f'P({indicator_str}) = {probability:.4g}')
    else:
        condition_str = pgm.indicator_str(*condition)
        print(f'P({indicator_str} | {condition_str}) = {probability:.4g}')


def show_marginal_distribution(rv: RandomVariable, condition: Condition = ()):
    """
    Print the marginal probability distribution of the random variable,
    with an optional set of condition indicators.
    """
    condition = check_condition(condition)
    probabilities = wmc.marginal_distribution(rv, condition=condition)
    if len(condition) == 0:
        print(f'P({rv}) = {probabilities}')
    else:
        condition_str = pgm.indicator_str(*condition)
        print(f'P({rv} | {condition_str}) = {probabilities}')


print()
show_probability(difficult('Yes'))
show_probability(difficult('No'))

print()
show_probability(intelligent('Yes'))
show_probability(intelligent('No'))

print()
show_probability(difficult('Yes'), intelligent('No'))
show_probability(grade('3'), condition=intelligent('Yes'))
show_probability(grade('3'), condition=(intelligent('Yes'), letter('Yes')))
show_probability(intelligent('No'), condition=letter('Yes'))
show_probability(intelligent('Yes'), condition=letter('Yes'))

print()
show_marginal_distribution(difficult)
show_marginal_distribution(intelligent)
show_marginal_distribution(grade)

print()
for example_condition in rv_instances_as_indicators(difficult, intelligent):
    show_marginal_distribution(grade, condition=example_condition)

print()
print('Done.')
