import pathlib as _pathlib
import re as _re
import sys as _sys
from typing import Iterable, Set, List

from ck.in_out.parser_utils import escape_string
from ck.pgm import PGM, RandomVariable


def render_bayesian_network(
        pgm: PGM,
        out=None,
        *,
        check_structure_bayesian: bool = True,
) -> List[str]:
    """
    Render a PGM as a Hugin 'net' file.

    Args:
        pgm: is a PGM object.
        out: is an output file or None for stdout.
        check_structure_bayesian: If True, then raise an exception if not pgm.is_structure_bayesian.

    Returns:
        a list of node names used in the Hugin 'net' file, co-indexed with pgm.rvs.

    Raises:
        ValueError: if `check_structure_bayesian` is true and `pgm.is_structure_bayesian` is false.
    """
    if check_structure_bayesian and not pgm.is_structure_bayesian:
        raise ValueError('attempting to render a PGM with non-Bayesian structure')

    node_names: List[str] = _make_node_names(pgm.rvs)

    if out is None:
        _render_bayesian_network(pgm, node_names, _sys.stdout)
    elif isinstance(out, (str, _pathlib.Path)):
        with open(out, 'w') as file:
            _render_bayesian_network(pgm, node_names, file)
    else:
        _render_bayesian_network(pgm, node_names, out)

    return node_names


# ============================================================
#  Private support
# ============================================================

def _make_node_names(rvs: Iterable[RandomVariable]) -> List[str]:
    """
    Make a dictionary from `RandomVariable.idx` to a node label that works in a Hugin 'net' file.
    """
    node_names: List[str] = []
    made_names: Set[str] = set()
    for rv in rvs:
        name = _rv_name(rv)
        if name in made_names:
            prefix = name + '_'
            i = 2
            name = prefix + str(i)
            while name in made_names:
                i += 1
                name = prefix + str(i)
        made_names.add(name)
        node_names.append(name)
    return node_names


def _render_bayesian_network(pgm: PGM, node_names: List[str], out):
    out.write('net{}\n')

    for rv in pgm.rvs:
        _write_node_block(rv, node_names, out)

    out.write('\n')

    for factor in pgm.factors:
        _write_potential_block(factor, node_names, out)


def _write_node_block(rv, node_names: List[str], out):
    out.write('node ' + node_names[rv.idx] + '\n')
    out.write('{\n')
    _write_node_block_label(rv, out)
    _write_node_block_states(rv, out)
    out.write('}\n')


def _write_node_block_label(rv, out):
    out.write('  label = "')
    out.write(_rv_label(rv))
    out.write('";\n')


def _write_node_block_states(rv, out):
    out.write('  states = (')
    for state in rv.states:
        out.write(' "' + _state_label(state) + '"')
    out.write(' );\n')


def _write_potential_block_link(factor, node_names: List[str], out):
    out.write('potential (')

    for rv_count, rv in enumerate(factor.rvs):
        if rv_count == 1:
            out.write(' |')
        out.write(' ' + node_names[rv.idx])

    out.write(' )\n')


def _recursively_write_ordered_data(shape, address_order_map, address_current, current_depth, max_depth, function, out):
    out.write('( ')

    mapped_current_depth = address_order_map[current_depth]

    for i in range(shape[mapped_current_depth]):
        address_current[mapped_current_depth] = i

        if current_depth == max_depth:
            out.write(str(function[address_current]) + ' ')
        else:
            _recursively_write_ordered_data(
                shape, address_order_map, address_current, current_depth + 1, max_depth, function, out
            )

    out.write(") ")


def _write_potential_block_data(factor, out):
    out.write('{\n')
    out.write('  data = ')

    function = factor.function
    shape = factor.shape

    address_current = [0] * len(shape)
    max_depth = len(shape) - 1

    # The ordering of data in a 'net' file is different to the natural order.
    # Consequently, address_order_map will keep track of the required ordering.
    address_order_map = [0] * len(shape)
    for index in range(max_depth):
        address_order_map[index] = index + 1

    _recursively_write_ordered_data(shape, address_order_map, address_current, 0, max_depth, function, out)

    out.write(';\n')
    out.write('}\n')


def _write_potential_block(factor, node_names: List[str], out):
    _write_potential_block_link(factor, node_names, out)
    _write_potential_block_data(factor, out)


def _rv_label(rv):
    """
    make a label for a random variable
    """
    return escape_string(rv.name, double_quotes=True)


def _state_label(state) -> str:
    """
    make a label for a random variable state
    """
    return escape_string(str(state), double_quotes=True)


def _rv_name(rv: RandomVariable) -> str:
    """
    make a name for a random variable
    """
    return _re.sub(r'[^0-9a-zA-Z]+', '_', rv.name)
