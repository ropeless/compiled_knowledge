import pathlib as _pathlib
import sys as _sys
from dataclasses import dataclass
from typing import Optional

from ck.pgm import RandomVariable, Factor, PGM


def render_bayesian_network(
        pgm: PGM,
        out=None,
        *,
        network_name: str = 'network',
        pom: str = 'pom',
        use_variable_names: bool = False,
        bake: bool = True,
        check_structure_bayesian: bool = True,
) -> None:
    """
    Render a Bayesian network PGM to 'out' as a pomegranate Python script.

    Args:
        pgm: is a PGM object.
        out: is an output file or None for stdout.
        network_name: A Python variable name to use for the pomegranate object.
        pom: is the Python import name for pomegranate.
        use_variable_names: If false, then Python variable names for distribution objects will
            be system generated, otherwise the random variable name will be used.
        bake: If True, then {network_name}.bake() is appended to the script.
        check_structure_bayesian: If True, then raise an exception if not pgm.is_structure_bayesian.
    """
    if check_structure_bayesian:
        if not pgm.is_structure_bayesian:
            raise RuntimeError('Attempting to render a PGM with non-Bayesian structure')

    if out is None:
        _render_bayesian_network(pgm, _sys.stdout, network_name, pom, use_variable_names, bake)
    elif isinstance(out, (str, _pathlib.Path)):
        with open(out, 'w') as file:
            _render_bayesian_network(pgm, file, network_name, pom, use_variable_names, bake)
    else:
        _render_bayesian_network(pgm, out, network_name, pom, use_variable_names, bake)


# ============================================================
#  Private support
# ============================================================


@dataclass
class _RVData:
    rv: RandomVariable
    cpt_name: str
    factor: Optional[Factor] = None
    wrote_cpt: bool = False


def _render_bayesian_network(
        pgm: PGM,
        out,
        network_name,
        pom,
        use_variable_names,
        bake
):
    """
    See render_bayesian_network.
    """
    if use_variable_names:
        rv_data_lookup = {rv: _RVData(rv, rv.name) for rv in pgm.rvs}
    else:
        rv_data_lookup = {rv: _RVData(rv, f'cpt_{i}') for i, rv in enumerate(pgm.rvs)}

    def write(*args, sep=' ', end='\n'):
        out.write(sep.join(str(arg) for arg in args))
        out.write(end)

    for factor in pgm.factors:
        child = factor.rvs[0]
        rv_data = rv_data_lookup.get(child)
        if rv_data is None:
            raise RuntimeError(f'lost random variable in factors: {child}')
        if rv_data.factor is not None:
            raise RuntimeError(f'duplicated child random variable in factors: {child}')
        rv_data.factor = factor

    write(f'import pomegranate as {pom}')
    write()

    for rv in pgm.rvs:
        _write_cpt_r(rv, rv_data_lookup, pom, write)

    write()
    for rv in pgm.rvs:
        _write_state(rv, rv_data_lookup, pom, write)

    write()
    write(f'{network_name} = {pom}.BayesianNetwork({pgm.name!r})')

    write()
    for rv in pgm.rvs:
        write(f'{network_name}.add_nodes(s{rv.idx})')

    write()
    for rv in pgm.rvs:
        _write_add_edges(rv, rv_data_lookup, network_name, write)

    if bake:
        write()
        write(f'{network_name}.bake()')


def _write_cpt_r(rv, rv_data_lookup, pom, write):
    """
    Recursively write CPT for the given random variable
    """
    rv_data = rv_data_lookup[rv]
    if rv_data.wrote_cpt:
        return

    factor = rv_data.factor
    if factor is None:
        raise RuntimeError(f'Missing random variable factor: {rv}')

    # first ensure the parent CPTs are written
    for parent in factor.rvs[1:]:
        _write_cpt_r(parent, rv_data_lookup, pom, write)

    # now write our own CPT
    write()
    _write_cpt(rv, rv_data_lookup, pom, write)
    rv_data.wrote_cpt = True


def _write_cpt(rv, rv_data_lookup, pom, write):
    """
    Write the CPT for one random variable.
    This with either write a pomegranate
    DiscreteDistribution or a ConditionalProbabilityTable.
    """
    rv_data = rv_data_lookup[rv]
    cpt_name = rv_data.cpt_name
    factor = rv_data.factor
    function = factor.function
    parents = factor.rvs[1:]

    if len(parents) == 0:
        write(f'{cpt_name} = {pom}.DiscreteDistribution({{')
        for i, state in enumerate(rv_data.rv.states):
            value = function[i]
            write(f'{state!r}: {value},')
        write(f'}})')
    else:
        write(f'{cpt_name} = {pom}.ConditionalProbabilityTable([')
        state_name_lookup = [rv.states for rv in factor.rvs]
        for inst in factor.instances():
            value = function[inst]
            state_names = ', '.join(repr(states[i]) for i, states in zip(inst, state_name_lookup))
            write(f'    [{state_names}, {value}],')
        write(f'], [')
        for p in parents:
            write(f'    {rv_data_lookup[p].cpt_name},')
        write('])')


def _write_state(rv, rv_data_lookup, pom, write):
    rv_data: _RVData = rv_data_lookup[rv]
    rv_idx = rv.idx
    cpt_name = rv_data.cpt_name
    rv_name = rv.name
    write(f's{rv_idx} = {pom}.State({cpt_name}, name={rv_name!r})')


def _write_add_edges(rv, rv_data_lookup, network_name, write):
    rv_data = rv_data_lookup[rv]
    factor = rv_data.factor
    parents = factor.rvs[1:]
    rv_idx = rv.idx

    if parents == 0:
        return

    for parent in parents:
        write(f'{network_name}.add_edge(s{parent.idx}, s{rv_idx})')
