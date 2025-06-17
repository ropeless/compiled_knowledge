from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Sequence, Optional, Tuple, List, Dict, Set, assert_never

from ck.circuit import CircuitNode, Circuit, VarNode, OpNode, ADD, MUL
from ck.circuit_compiler import llvm_vm_compiler, CircuitCompiler
from ck.pgm import RandomVariable, Instance
from ck.pgm_circuit import PGMCircuit
from ck.pgm_circuit.program_with_slotmap import ProgramWithSlotmap
from ck.pgm_circuit.slot_map import SlotMap
from ck.pgm_circuit.support.compile_circuit import compile_results
from ck.probability.probability_space import check_condition
from ck.program.program_buffer import ProgramBuffer
from ck.program.raw_program import RawProgram
from ck.utils.np_extras import NDArray, NDArrayNumeric

_NO_TRACE = (-1, -1)  # used as a sentinel value

_CCT_COMPILER = llvm_vm_compiler  # Python module used for compiling an MPE circuit


class MPEProgram(ProgramWithSlotmap):
    """
    A class for computing Most Probable Explanation (MPE). This is equivalent to
    Maximum A Posterior (MAP) inference when there are no latent random variables.
    """

    def __init__(
            self,
            pgm_circuit: PGMCircuit,
            trace_rvs: Optional[Sequence[RandomVariable]] = None,
            const_parameters: bool = True,
            log_parameters: bool = False,
    ):
        """
        Construct a MPEProgram object.

        Compile the circuit for computing Most Probable Explanation (MPE). This is equivalent to
        Maximum A Posterior (MAP) inference when there are no latent variables.

        This will compile a clone of the given circuit with
        'add' nodes replaced with 'max' nodes.

        This will augment the given circuit and compile it to make a program for computing MPE states.
        'trace_vars' is a list random variables, where each random variable is a list of circuit var nodes, each
        var node representing an indicator (i.e., a state) of a random variable.
        Assumes that all operator nodes to compute top are either an add or mul node.

        Args:
            pgm_circuit: The circuit representing a PGM.
            trace_rvs: the random variables to compute MPE for, default is all random variables of the PGM.
            const_parameters: if True then any circuit variable representing a parameter value will
                be made 'const' in the resulting program.
            log_parameters: if true, then parameters are taken to be logs, i.e., uses addition instead
                of multiplication.
        """
        trace_rvs: Tuple[RandomVariable, ...] = pgm_circuit.rvs if trace_rvs is None else tuple(trace_rvs)
        if len(trace_rvs) != len(set(trace_rvs)):
            raise ValueError('duplicated trace random variable detected')

        top: CircuitNode = pgm_circuit.circuit_top
        circuit: Circuit = top.circuit
        slot_map: SlotMap = pgm_circuit.slot_map

        cct_compiler: CircuitCompiler
        if log_parameters:
            cct_compiler = partial(_CCT_COMPILER.compile_circuit, data_type=_CCT_COMPILER.DataType.MAX_SUM)
        else:
            cct_compiler = partial(_CCT_COMPILER.compile_circuit, data_type=_CCT_COMPILER.DataType.MAX_MUL)

        # make inv_trace_blocks
        #
        # inv_trace_blocks[slot] = (rv_trace_idx, state_idx)
        # where
        #   rv_trace_idx is an index into trace_vars,
        #   state_idx is an index into trace_vars[rv_trace_idx] indicators,
        #
        #   slot = slot_map[ind], where ind = trace_vars[rv_trace_idx][state_idx].
        #
        inv_trace_blocks: List[Tuple[int, int]] = [_NO_TRACE] * circuit.number_of_vars
        rv_trace_idx: int
        trace_rv: RandomVariable
        for rv_trace_idx, trace_rv in enumerate(trace_rvs):
            for state_idx in trace_rv.state_range():
                slot: int = slot_map[trace_rv[state_idx]]
                if inv_trace_blocks[slot] is not _NO_TRACE:
                    raise ValueError('unexpected reused circuit slot')
                inv_trace_blocks[slot] = (rv_trace_idx, state_idx)

        used_nodes: List[CircuitNode] = list(circuit.reachable_op_nodes(top))

        mpe_idx: Dict[int, int] = {
            id(used_node): used_node_idx
            for used_node_idx, used_node in enumerate(used_nodes)
        }

        # create a dummy MPE result until compute is called
        dummy_result = MPEResult(float('nan'), tuple(0 for _ in trace_rvs))

        self._trace_rvs: Tuple[RandomVariable, ...] = trace_rvs
        self._inv_trace_blocks = inv_trace_blocks
        self._top: CircuitNode = top
        self._mpe_result: MPEResult = dummy_result

        self._top_idx: Optional[int] = mpe_idx.get(id(top))  # it may be possible that top is not an op node.
        self._used_nodes: List[CircuitNode] = used_nodes
        self._mpe_idx: Dict[int, int] = mpe_idx

        raw_program: RawProgram = compile_results(
            pgm_circuit=pgm_circuit,
            results=used_nodes,
            const_parameters=const_parameters,
            compiler=cct_compiler,
        )
        ProgramWithSlotmap.__init__(self, ProgramBuffer(raw_program), slot_map, pgm_circuit.rvs, pgm_circuit.conditions)

        if not const_parameters:
            # set the parameter slots
            self.vars[pgm_circuit.number_of_indicators:] = pgm_circuit.parameter_values

    def mpe(self, *condition) -> MPEResult:
        """
        What is the MPE, given any conditioning indicators.

        The mpe array may contain None in an element corresponding to a traced random variable where
        all states of that random variable lead to the same wmc value. I.e., the solution is indifferent
        to the state of that random variable. In this case, a caller is at liberty to use any state for that
        random variable as an MPE solution. For example, all 'None' values could be replaced with zero
        and the solution is still a valid MPE solution.

        Returns:
            an MPEResult with field `wmc` and `mpe`.
            wmc: is the value of the weighted model count.
            mpe: is an Instance, co-indexed with trace vars, where mpe[rv_idx] = state_idx.
        """
        condition = check_condition(condition)
        self.compute_conditioned(*condition)
        return self._mpe_result

    @property
    def trace_rvs(self) -> Sequence[RandomVariable]:
        """
        What are the random variables used in an MPE trace.
        """
        return self._trace_rvs

    def compute(self) -> NDArrayNumeric:
        """
        Execute the program to compute and return the result. As per `ProgramBuffer.compute`.

        Warning:
            when returning an array, the array is backed by the program buffer memory, not a copy.
        """
        program_result: NDArray = self._program_buffer.compute()
        self._trace()
        return program_result

    @property
    def mpe_result(self) -> MPEResult:
        """
        Get the MPEResult of the last program computation.

        Returns:
            an MPEResult object.
        """
        return self._mpe_result

    def _trace(self) -> None:
        """
        Trace the last program computation to determine the wmc and the mpe states.
        """
        if self._top_idx is not None:
            wmc: float = self.results.item(self._top_idx)
            states: List[Optional[int]] = [None for _ in self._trace_rvs]
            seen: Set[int] = set()
            self._trace_r(self._top, wmc, states, seen)
            mpe = tuple(
                0 if state_idx is None else state_idx
                for state_idx in states
            )
            self._mpe_result = MPEResult(wmc, mpe)

    def _trace_r(self, node: CircuitNode, node_value: float, states: List[Optional[int]], seen: Set[int]) -> None:

        # A circuit is a DAG, not necessarily a tree.
        # No need to revisit nodes.
        if id(node) in seen:
            return
        seen.add(id(node))

        if isinstance(node, VarNode):
            self._trace_var(node, states)
        elif isinstance(node, OpNode):
            if node.symbol == ADD:
                # Find which child node led to the max result, then recurse though it only.
                for child in node.args:
                    if isinstance(child, OpNode):
                        child_value: float = self.results.item(self._mpe_idx[id(child)])
                        if child_value == node_value:
                            self._trace_r(child, child_value, states, seen)
                            return
                    elif isinstance(child, VarNode):
                        child_value: float = self.vars.item(child.idx)
                        if child_value == node_value:
                            self._trace_var(child, states)
                            return
                # No child value equaled the value for node! We should never get here
                assert_never('not reached')
            elif node.symbol == MUL:
                # Recurse though each child node
                for child in node.args:
                    if isinstance(child, OpNode):
                        child_value: float = self.results.item(self._mpe_idx[id(child)])
                        self._trace_r(child, child_value, states, seen)
                    elif isinstance(child, VarNode):
                        self._trace_var(child, states)

    def _trace_var(self, node: VarNode, states: List[Optional[int]]) -> None:
        trace = self._inv_trace_blocks[node.idx]
        if trace is not _NO_TRACE:
            rv_trace_idx, state_idx = trace
            states[rv_trace_idx] = state_idx


@dataclass
class MPEResult:
    """
    An MPE result is the result of MPE inference.

    Fields:
        wmc: the weighted model count value of the MPE solution.
        mpe: The MPE solution instance. If there are ties then this will just be once instance.
    """
    wmc: float
    mpe: Instance
