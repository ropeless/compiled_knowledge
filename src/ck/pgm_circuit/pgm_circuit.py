from dataclasses import dataclass
from typing import Sequence, List, Dict

from ck.circuit import CircuitNode, Circuit
from ck.pgm import RandomVariable, Indicator
from ck.pgm_circuit.slot_map import SlotMap, SlotKey
from ck.utils.np_extras import NDArray


@dataclass
class PGMCircuit:
    """
    A data structure representing the results of compiling a PGM to a circuit.

    If the circuit contains variables to represent parameter values, then `parameter_values`
    holds the values of the parameters. Specifically, given parameter id `param_id`, then
    `parameter_values[slot_map[param_id] - number_of_indicators]` is the value of the
    identified parameter as it was in the PGM.

    Fields:
        rvs: holds the random variables from the PGM as it was compiled, in order.

        conditions: any conditions on `rvs` that were compiled into the circuit.

        number_of_indicators: is the number of indicators in `rvs` which is
            `sum(len(rv) for rv in rvs`. Specifically, `circuit.vars[i]` is the circuit variable
            corresponding to the ith indicator, where `circuit` is `circuit_top.circuit` and
            indicators are ordered as per `rvs`.

        number_of_parameters: is the number of parameters from the PGM that are
            represented as circuit variables. This may be zero if parameters from the PGM
            were compiled as constants.

        slot_map[x]: gives the index of the circuit variable corresponding to x,
            where x is either a random variable indicator (Indicator) or a parameter id (ParamId).

    """

    rvs: Sequence[RandomVariable]
    conditions: Sequence[Indicator]
    circuit_top: CircuitNode
    number_of_indicators: int
    number_of_parameters: int
    slot_map: SlotMap
    parameter_values: NDArray

    def dump(self, *, prefix: str = '', indent: str = '    ') -> None:
        """
        Print a dump of the circuit.
        This is intended for debugging and demonstration purposes.

        Args:
            prefix: optional prefix for indenting all lines.
            indent: additional prefix to use for extra indentation.
        """

        # We infer names for the circuit variables, either as an indicator or as a parameter.
        # The `var_names` will be passed to `circuit.dump`.

        circuit: Circuit = self.circuit_top.circuit
        var_names: List[str] = [''] * circuit.number_of_vars

        # Name the circuit variables that are indicators
        rvs_by_idx: Dict[int, RandomVariable] = {rv.idx: rv for rv in self.rvs}
        slot_key: SlotKey
        slot: int
        for slot_key, slot in self.slot_map.items():
            if isinstance(slot_key, Indicator):
                rv = rvs_by_idx[slot_key.rv_idx]
                state_idx = slot_key.state_idx
                var_names[slot] = f'{rv.name!r}[{state_idx}] {rv.states[state_idx]!r}'

        # Name the circuit variables that are parameters
        for i, param_value in enumerate(self.parameter_values):
            slot = i + self.number_of_indicators
            var_names[slot] = f'param[{i}] = {param_value}'

        # Dump the circuit
        circuit.dump(prefix=prefix, indent=indent, var_names=var_names)
