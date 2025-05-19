from __future__ import annotations

from typing import Optional, Tuple, List

from ck.circuit import CircuitNode, Circuit, TmpConst
from ck.pgm import RandomVariable
from ck.pgm_circuit import PGMCircuit
from ck.pgm_circuit.program_with_slotmap import ProgramWithSlotmap
from ck.pgm_circuit.slot_map import SlotMap
from ck.pgm_circuit.support.compile_circuit import compile_results
from ck.probability.probability_space import check_condition, Condition
from ck.program.program_buffer import ProgramBuffer
from ck.program.raw_program import RawProgram
from ck.utils.np_extras import NDArray


class TargetMarginalsProgram(ProgramWithSlotmap):
    """
    """

    def __init__(
            self,
            pgm_circuit: PGMCircuit,
            target_rv: RandomVariable,
            const_parameters: bool = True,
    ):
        """
        Construct a TargetMarginalsProgram object.

        Compile the given circuit for computing marginal probabilities over the states of 'target_var'.

        Args:
            pgm_circuit: The circuit representing a PGM.
            target_rv: the random variable to compute marginals for.
            const_parameters: if True then any circuit variable representing a parameter value will
                be made 'const' in the resulting program.
        """
        top_node: CircuitNode = pgm_circuit.circuit_top
        circuit: Circuit = top_node.circuit
        slot_map: SlotMap = pgm_circuit.slot_map
        input_rvs: List[RandomVariable] = list(pgm_circuit.rvs)

        target_vars = [circuit.vars[slot_map[ind]] for ind in target_rv]
        cct_outputs = circuit.partial_derivatives(top_node, target_vars)

        # Remove the target rv from the input rvs.
        target_index = input_rvs.index(target_rv)  # will throw if not found
        del input_rvs[target_index]

        with TmpConst(circuit) as tmp:
            tmp.set_const(target_vars, 1)
            raw_program: RawProgram = compile_results(
                pgm_circuit=pgm_circuit,
                results=cct_outputs,
                const_parameters=const_parameters,
            )

        ProgramWithSlotmap.__init__(self, ProgramBuffer(raw_program), slot_map, input_rvs, pgm_circuit.conditions)

        # additional fields
        self._x_slots: List[List[int]] = [[slot_map[ind] for ind in rv] for rv in input_rvs]
        self._y_size: int = raw_program.number_of_results
        self._target_rv: RandomVariable = target_rv
        self._number_of_indicators: int = pgm_circuit.number_of_indicators
        self._z_cache: Optional[float] = None

        # consistency check
        assert (self._y_size == len(self._target_rv))

        if not const_parameters:
            # set the parameter slots
            self.vars[pgm_circuit.number_of_indicators:] = pgm_circuit.parameter_values

    @property
    def target_rv(self) -> RandomVariable:
        return self._target_rv

    def map(self, condition: Condition = ()) -> Tuple[float, int]:
        """
        Return the maximum a posterior (MAP) state of the target variable.

        Args:
            condition: any conditioning indicators.

        Returns:
            (pr, state_idx) where
            pr is the MAP probability
            state_idx: is the MAP state index of `self.target_rv`.
        """
        self.set_condition(*check_condition(condition))
        self.compute()
        results: NDArray = self.results
        z: float = results.sum()

        max_p = -1
        max_i = -1
        for i in range(self._y_size):
            p = results[i]
            if p > max_p:
                max_p = p
                max_i = i

        return max_p / z, max_i
