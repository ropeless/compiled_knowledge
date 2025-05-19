import datetime
import importlib
import importlib.machinery
import math
import types
from pathlib import Path
from typing import Optional, List, Any, Sequence

from ck.pgm import PGM, DensePotentialFunction, SparsePotentialFunction, State, CompactPotentialFunction, \
    ClausePotentialFunction, CPTPotentialFunction, default_pgm_name


def write_python(
        pgm: PGM,
        pgm_name: str = 'pgm',
        import_module: str = 'ck.pgm',
        package_name: Optional[str] = None,
        use_variable_names: bool = False,
        include_potential_functions: bool = True,
        include_comment: bool = False,
        note: Optional[str] = None,
        author: Optional[str] = None,
        file=None
) -> None:
    """
    Print a Python script that would build the PGM.

    Args:
        pgm: The PGM to write.
        pgm_name: A Python variable name to use for the PGM object.
        import_module: if not None, then an 'import' command will be included to import the named module.
        package_name: if None, then 'PGM()' is used to create the PGM object,
        otherwise '{package_name}.PGM()' is used.
        use_variable_names: If false, then Python variable names for RandomVariable objects will
        be system generated, otherwise the random variable name will be used.
        include_potential_functions: whether to dump the potential functions or not.
        include_comment: include a Python comment or not.
        note: An explicit comment to include.
        author: an optional author name to go in the comment.
        file: optional file argument to the print function.
    """

    def _print(*args, **kwargs) -> None:
        print(*args, file=file, **kwargs)

    constructor_args = '' if pgm.name == default_pgm_name(pgm) else repr(pgm.name)
    class_name = PGM.__name__

    if use_variable_names:
        def rv_name(rv):
            return rv.name
    else:
        def rv_name(rv):
            return f'{pgm_name}_rv{rv.idx}'

    has_a_comment = include_comment or note is not None
    if has_a_comment:
        _print('"""')
        _print(f'PGM name: {pgm.name}')
        _print(f'{datetime.datetime.now()}')
        _print()
    if note is not None:
        _print(str(note))
        _print()
    if include_comment:
        num_states: int = pgm.number_of_states
        number_of_parameters = sum(factor.function.number_of_parameters for factor in pgm.factors)
        number_of_nz_parameters = sum(function.number_of_parameters for function in pgm.non_zero_functions)
        precision = 3
        _print(f'number of random variables: {pgm.number_of_rvs:,}')
        _print(f'number of indicators: {pgm.number_of_indicators:,}')
        _print(f'number of states: {num_states}')
        _print(f'log 2 of states: {math.log2(num_states):,.{precision}f}')
        _print(f'number of factors: {pgm.number_of_factors:,}')
        _print(f'number of functions: {pgm.number_of_functions:,}')
        _print(f'number of parameters: {number_of_parameters:,}')
        _print(f'number of functions (excluding ZeroPotentialFunction): {pgm.number_of_non_zero_functions:,}')
        _print(f'number of parameters (excluding ZeroPotentialFunction): {number_of_nz_parameters:,}')
        _print(f'Bayesian structure: {pgm.is_structure_bayesian}')
        _print(f'CPT factors: {pgm.factors_are_cpts()}')
        _print()
        _print('Usage:')
        rv_list = ', '.join([rv_name(rv) for rv in pgm.rvs])
        sep = '' if pgm.number_of_rvs == 0 else ', '
        _print(f'from {pgm.name} import {pgm_name}{sep}{rv_list}')
        _print()
    if has_a_comment:
        _print('"""')
    if author is not None:
        _print(f'__author__ = {author!r}')
    if has_a_comment or author is not None:
        _print()

    if import_module is not None:
        if package_name is None:
            _print(f'from {import_module} import {class_name}')
        else:
            _print(f'import {import_module} as {package_name}')

    if package_name is None:
        _print(f'{pgm_name} = {class_name}({constructor_args})')
    else:
        _print(f'{pgm_name} = {package_name}.{class_name}({constructor_args})')

    # Print random variables
    for rv in pgm.rvs:
        if rv.is_default_states():
            states = len(rv.states)
        else:
            states = _repr_states(rv.states)
        _print(f'{rv_name(rv)} = {pgm_name}.new_rv({rv.name!r}, {states})')

    # Print factors
    for factor in pgm.factors:
        rvs = ', '.join([rv_name(rv) for rv in factor.rvs])
        factor_name = f'{pgm_name}_factor{factor.idx}'
        _print(f'{factor_name} = {pgm_name}.new_factor({rvs})')

    # Print potential functions
    if include_potential_functions:
        seen_functions = {}
        for factor in pgm.factors:
            if factor.is_zero:
                continue

            _print()
            factor_name = f'{pgm_name}_factor{factor.idx}'
            function = factor.function

            function_name = seen_functions.get(function)
            if function_name is not None:
                _print(f'{factor_name}.function = {function_name}')
                continue

            function_name = f'{pgm_name}_function{len(seen_functions)}'
            seen_functions[function] = function_name

            if isinstance(function, DensePotentialFunction):
                _print(f'{function_name} = {factor_name}.set_dense()')
                _write_python_dense_function(function_name, function, _print)

            elif isinstance(function, SparsePotentialFunction):
                _print(f'{function_name} = {factor_name}.set_sparse()')
                for key, idx, value in function.keys_with_param:
                    _print(f'{function_name}[{key}] = {value}')

            elif isinstance(function, CompactPotentialFunction):
                _print(f'{function_name} = {factor_name}.set_compact()')
                for key, value in function.items():
                    if value != 0:
                        _print(f'{function_name}[{key}] = {value}')

            elif isinstance(function, ClausePotentialFunction):
                states = ', '.join(repr(v) for v in function.clause)
                _print(f'{function_name} = {factor_name}.set_clause({states})')

            elif isinstance(function, CPTPotentialFunction):
                _print(f'{function_name} = {factor_name}.set_cpt()')
                for parent_states, cpd in function.cpds():
                    cpd = ', '.join(repr(v) for v in cpd)
                    _print(f'{function_name}.set_cpd({parent_states}, ({cpd}))')

            else:
                raise RuntimeError(f'unimplemented writing of function type {function.__class__.__name__}')


def read_python(
        source: str | Path,
        var_name: Optional[str] = None,
        module_name: Optional[str] = None,
) -> PGM:
    """
    Load a PGM previously written using `write_python`.

    Args:
        source: The source file name or file path
        var_name: The name of the PGM variable to load, if None, then a name will be found.
        module_name: The name of the module that file will be loaded as, default is the name of the source.

    Returns:
        the loaded PGM.

    Raises:
        RuntimeError: if a unique PGM object is not found (with the given var_name).
    """
    if module_name is None:
        if isinstance(source, str):
            module_name = Path(source).name
        elif isinstance(source, Path):
            module_name = source.name

    loader = importlib.machinery.SourceFileLoader(module_name, source)
    module = types.ModuleType(loader.name)
    loader.exec_module(module)

    if var_name is not None:
        pgm = getattr(module, var_name)
        if not isinstance(pgm, PGM):
            raise RuntimeError(f'object {var_name} is not a PGM')
        return pgm
    else:
        potentials: List[PGM] = [
            value
            for var, value in vars(module).items()
            if not var.startswith('_') and isinstance(value, PGM)
        ]
        if len(potentials) != 1:
            raise RuntimeError(f'unique PGM object not found')
        return potentials[0]


def _write_python_dense_function(function_name: str, function: DensePotentialFunction, _print) -> None:
    """
    Support method for `write_python`.
    """
    num_params = function.number_of_parameters
    if num_params > 0:
        indent = '    '
        wrap_count = 5
        _print(f'{function_name}.set_flat(', end='')

        if num_params >= wrap_count:
            _print('', indent, sep='\n', end='')
        for i in range(num_params):
            _print(repr(function.param_value(i)), end='')
            next_i = i + 1
            if next_i != num_params:
                _print(', ', end='')
                if next_i % wrap_count == 0:
                    _print()
                    _print(indent, end='')
        if num_params >= wrap_count:
            _print()

        _print(')')


def _isnan(value: Any) -> bool:
    """
    Returns:
        True only if the given value is a float and is NaN.
    """
    return isinstance(value, float) and math.isnan(value)


def _repr_states(states: Sequence[State]) -> str:
    """
    If states contain float('nan') then write_python needs to avoid the issue
    that repr(float('nan')) is not parsable by Python.

    See https://bugs.python.org/issue1732212
    """
    return '(' + ', '.join(_repr_state(state) for state in states) + ')'


def _repr_state(state: State) -> str:
    """
    Render a state as a string.

    If states is float('nan') then write_python needs to avoid the issue
    that repr(float('nan')) is not parsable by Python.

    See https://bugs.python.org/issue1732212
    """
    if _isnan(state):
        return "float('nan')"
    else:
        return repr(state)
