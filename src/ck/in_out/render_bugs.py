import pathlib as _pathlib
import sys as _sys
from typing import Sequence

from ck.pgm import RandomVariable, Factor, PotentialFunction, PGM


def render_bayesian_network(
        pgm: PGM,
        out=None,
        *,
        check_structure_bayesian: bool = True,
) -> None:
    """
    Render a Bayesian network PGM as a BUGS model file in ODC format.

    Args:
        pgm: is a PGM object.
        out: is an output file or None for stdout.
        check_structure_bayesian: If True, then raise an exception if not pgm.is_structure_bayesian.
    """
    if check_structure_bayesian:
        if not pgm.is_structure_bayesian:
            raise RuntimeError('attempting to render a PGM with non-Bayesian structure')

    if out is None:
        _render_bayesian_network(pgm, _sys.stdout)
    elif isinstance(out, (str, _pathlib.Path)):
        with open(out, 'w') as file:
            _render_bayesian_network(pgm, file)
    else:
        _render_bayesian_network(pgm, out)


# ============================================================
#  Private support
# ============================================================


def _render_bayesian_network(pgm: PGM, out):
    """
    See render_bayesian_network.
    """

    def write(*args, sep=' ', end='\n'):
        out.write(sep.join(str(arg) for arg in args))
        out.write(end)

    write('model {')
    seen_child_rvs = set()
    for factor in pgm.factors:
        child = factor.rvs[0]
        parents = factor.rvs[1:]

        if child in seen_child_rvs:
            raise RuntimeError(f'duplicated child random variable in factors: {child}')
        seen_child_rvs.add(child)

        _render_rv(child, parents, write)
    write('}')

    write('list(')
    for factor in pgm.factors[:-1]:
        _render_factor(factor, ',', write)
    _render_factor(pgm.factors[-1], '', write)
    write(')')


def _render_rv(child: RandomVariable, parents: Sequence[RandomVariable], write):
    name = child.name
    number_of_states = len(child)
    write(f'    {name} ~ dcat(p.{name}[', end='')
    for parent in parents:
        write(f'{parent.name},', end='')
    write(f'1:{number_of_states}])')


def _render_factor(factor: Factor, delim, write):
    child = factor.rvs[0]
    name = f'p.{child.name}'
    write(f'    {name} = ', end='')

    if len(factor.rvs) == 1:
        _write_param_values(factor.function, write)
    else:
        parents = factor.rvs[1:]
        dims = [len(rv) for rv in parents] + [len(child)]
        dims_str = ','.join(str(d) for d in dims)
        write('structure(.Data = ', end='')
        _write_param_values(factor.function, write)
        write(f', .Dim = c({dims_str}))', end='')

    write(delim)


def _write_param_values(function: PotentialFunction, write):
    write('c(', end='')

    num_child_states = function.shape[0]
    last_parent_state = function.number_of_parent_states - 1
    last_child_state = num_child_states - 1

    for i, parent_key in enumerate(function.parent_instances(flip=True)):
        for j in range(num_child_states):
            key = (j,) + parent_key
            value = function[key]
            write(value, end='')

            if i != last_parent_state or j != last_child_state:
                write(',', end='')
    write(')', end='')
