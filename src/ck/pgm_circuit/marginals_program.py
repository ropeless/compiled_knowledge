from __future__ import annotations

import random
from typing import Sequence, Optional, Tuple, List, Iterable, Dict

import numpy as np

from ck.circuit import CircuitNode, Circuit
from ck.pgm import RandomVariable, number_of_states, rv_instances_as_indicators
from ck.pgm_circuit import PGMCircuit
from ck.pgm_circuit.program_with_slotmap import ProgramWithSlotmap
from ck.pgm_circuit.slot_map import SlotMap
from ck.pgm_circuit.support.compile_circuit import compile_results
from ck.probability.probability_space import ProbabilitySpace, check_condition, Condition
from ck.program.program_buffer import ProgramBuffer
from ck.program.raw_program import RawProgram
from ck.sampling.marginals_direct_sampler import MarginalsDirectSampler
from ck.sampling.sampler import Sampler
from ck.sampling.sampler_support import SamplerInfo, get_sampler_info
from ck.sampling.uniform_sampler import UniformSampler
from ck.utils.np_extras import NDArray, NDArrayNumeric
from ck.utils.random_extras import Random


class MarginalsProgram(ProgramWithSlotmap, ProbabilitySpace):
    """
    A class for computing marginal probability distributions over states of selected output random variables.
    This class provides, for each indicator, the product of indicator value with the derivative
    of the network function with respect to the indicator.

    Compile the circuit for computing marginal probability distributions using the
    so-called 'differential' approach.

    Reference: Darwiche, A. (2003). A differential approach to inference in Bayesian
    networks. Journal of the ACM (JACM), 50(3), 280-305.

    A note about samplers
    ---------------------

    When creating a sampler, a client may request that samples are conditioned
    on provided condition indicators. Also, the WMCProgram may have been
    produced with compile-in conditions, e.g., using const_conditions with
    a call to PGM_cct.wmc(...).

    The conditions respected by a sampler are the conjunction of the compiled
    conditions and the sampler conditions. For example, with compiled condition
    (A[0], A[1], A[2]) and sampler condition (A[1], A[2], A[3]) the effective
    condition is (A[1], A[2]), i.e., a sample of A may be 1 or 2.

    Warning:
        if the sampled random variables include conditions, those conditions
        must be provided to the sampler. If a sampled random variable is conditioned
        at compile time, but not passed to the sampler, then the sample will not
        be aware of the conditions, and unexpected sample values may be produced.
    """

    def __init__(
            self,
            pgm_circuit: PGMCircuit,
            output_rvs: Optional[Sequence[RandomVariable]] = None,
            const_parameters: bool = True,
    ):
        """
        Construct a MarginalsProgram object.

        The given program should produce marginal outputs in the order
        of output_rvs indicators, followed by the wmc output.

        Args:
            pgm_circuit: The circuit representing a PGM.
            output_rvs: if None, the output rvs are all rvs, otherwise the given rvs.
            const_parameters: if True then any circuit variable representing a parameter value will
                be made 'const' in the resulting program.
        """
        top_node: CircuitNode = pgm_circuit.circuit_top
        circuit: Circuit = top_node.circuit
        slot_map: SlotMap = pgm_circuit.slot_map
        input_rvs: Sequence[RandomVariable] = pgm_circuit.rvs

        output_rvs: Sequence[RandomVariable] = tuple(output_rvs) if output_rvs is not None else input_rvs

        output_rvs_slots = [[slot_map[ind] for ind in rv] for rv in output_rvs]
        flat_out_rv_vars = [circuit.vars[slot] for slots in output_rvs_slots for slot in slots]
        derivatives = circuit.partial_derivatives(top_node, flat_out_rv_vars, self_multiply=True)

        raw_program: RawProgram = compile_results(
            pgm_circuit=pgm_circuit,
            results=derivatives + [top_node],
            const_parameters=const_parameters,
        )

        program_buffer = ProgramBuffer(raw_program)
        ProgramWithSlotmap.__init__(self, program_buffer, slot_map, input_rvs, pgm_circuit.conditions)

        # cache the input slots for the output rvs
        output_rvs_slots = [[slot_map[ind] for ind in rv] for rv in output_rvs]

        # cache the output offsets for the derivatives.
        # A map from `RandomVariable.idx` to offset into the result buffer
        self._rv_idx_to_result_offset: Dict[int, int] = {}
        prev_offset: int = 0
        for rv in output_rvs:
            self._rv_idx_to_result_offset[rv.idx] = prev_offset
            prev_offset += len(rv)

        # cached a map from output rv to its position in the marginals result
        self._rv_idx_to_output_index: Dict[int, int] = {rv.idx: i for i, rv in enumerate(output_rvs)}

        self._marginals: List[NDArrayNumeric] = []
        start = 0
        for rv_slots in output_rvs_slots:
            end = start + len(rv_slots)
            result_part = program_buffer.results[start:end]  # gets a view onto the same data.
            self._marginals.append(result_part)
            start = end

        # additional fields
        self._raw_program: RawProgram = raw_program
        self._program_buffer: ProgramBuffer = program_buffer
        self._number_of_indicators: int = pgm_circuit.number_of_indicators
        self._output_rvs = output_rvs
        self._output_rvs_slots = output_rvs_slots
        self._z_cache: Optional[float] = None

        if not const_parameters:
            # set the parameter slots
            self.vars[pgm_circuit.number_of_indicators:] = pgm_circuit.parameter_values

    @property
    def output_rvs(self):
        """
        What random variables are included in the marginal probabilities calculations.
        """
        return self._output_rvs

    def wmc(self, *condition: Condition) -> float:
        """
        What is the weight of the world with the given indicators.
        If multiple indicators from the same random variable ar mentioned, then it is treated as a disjunction.
        If a random variable is not mentioned in the indicators, that random variable is marginalised out.
        """
        self.set_condition(*condition)
        self._program_buffer.compute()
        return self.result_wmc

    @property
    def z(self):
        if self._z_cache is None:
            number_of_indicators: int = self._number_of_indicators
            slots: NDArray = self.vars
            old_vals: NDArray = slots[:number_of_indicators].copy()
            slots[:number_of_indicators] = 1
            self._program_buffer.compute()
            self._z_cache = self.result_wmc
            slots[:number_of_indicators] = old_vals
        return self._z_cache

    def marginal_distribution(self, *rvs: RandomVariable, condition: Condition = ()):
        # Check for easy cases.
        if len(rvs) == 0:
            if self.wmc(*condition) == 0:
                return np.array([np.nan])
            return np.array([1.0])
        if len(rvs) == 1:
            return self.marginal_for_rv(rvs[0], condition=condition)

        # We try to eliminate searching combinations of probabilities where marginals are zero.
        # If there are no marginal probabilities = 0, then this is equivalent to
        # ProbabilitySpace.marginal_distribution

        condition = check_condition(condition)
        rvs_marginals = self.marginal_for_rvs(rvs, condition=condition)
        zero_indicators = set(
            ind
            for rv, rv_marginal in zip(rvs, rvs_marginals)
            for ind, marginal in zip(rv, rv_marginal)
            if marginal == 0
        )
        raw_wmc = self._get_wmc_for_marginals(rvs, condition)

        if len(zero_indicators) == 0:
            wmc = raw_wmc
        else:
            def wmc(indicators):
                for ind in indicators:
                    if ind in zero_indicators:
                        return 0
                return raw_wmc(indicators)

        result = np.fromiter(
            (wmc(indicators) for indicators in rv_instances_as_indicators(*rvs)),
            count=number_of_states(*rvs),
            dtype=np.float64
        )
        _normalise_marginal(result)
        return result

    def marginal_for_rv(self, rv: RandomVariable, condition: Condition = ()) -> NDArrayNumeric:
        """
        Compute and return marginal distribution over the given random variable.
        The random variable is assumed to be in self.rvs.

        Returns:
            a numpy array representing the marginal distribution over the states of 'rv'.
        """
        self.compute_conditioned(*condition)
        return self.result_for_rv(rv)

    def marginal_for_rvs(self, rvs: Iterable[RandomVariable], condition: Condition = ()) -> List[NDArrayNumeric]:
        """
        Compute and return marginal distribution over the given random variables.
        Each random variable is assumed to be in self.rvs.

        Returns:
            a list of numpy arrays representing the marginal distribution over the
            states of each rv in the given random variables, `rvs`.
        """
        self.compute_conditioned(*condition)
        marginals = self._marginals
        rv_idx_to_output_index = self._rv_idx_to_output_index
        return list(marginals[rv_idx_to_output_index[rv.idx]] for rv in rvs)

    def compute(self) -> NDArrayNumeric:
        self._program_buffer.compute()
        for part in self._marginals:
            _normalise_marginal(part)
        return self._program_buffer.results

    @property
    def result_wmc(self) -> float:
        """
        Assuming the result has been computed,
        return the WMC value.
        """
        return self._program_buffer.results.item(-1)

    @property
    def result_marginals(self) -> List[NDArrayNumeric]:
        """
        Assuming the result has been computed,
        return the marginal distributions of each random variable, co-indexed with the
        output random variables, `self.output_rvs`.

        Returns:
            a list of numpy arrays, the list co-indexed with `self.output_rvs`, each numpy array
            representing the marginal distribution over the states of the co-indexed random variable.
        """
        return self._marginals

    def result_for_rv(self, rv: RandomVariable) -> NDArrayNumeric:
        """
        Assuming the result has been computed,
        return marginal distribution over the given random variable.
        The random variable is assumed to be in self.output_rvs.

        Returns:
            a numpy array representing the marginal distribution over the states of 'rv'.
        """
        return self._marginals[self._rv_idx_to_output_index[rv.idx]]

    def sample_uniform(
            self,
            rvs: Optional[RandomVariable | Sequence[RandomVariable]] = None,
            *,
            condition: Condition = (),
            rand: Random = random,
    ) -> Sampler:
        """
        Create a sampler that performs uniform sampling of
        the state space of the given random variables, rvs.

        The sampler will yield state lists, where the state
        values are co-indexed with rvs, or self.rvs if rvs is None.

        This sampler is not affected by and does not affect
        the state of input slots.

        Args:
            rvs: the list of random variables to sample; the
                yielded state vectors are co-indexed with rvs; if None,
                then the self.rvs are used; if rvs is a single
                random variable, then single samples are yielded.
            condition: is a collection of zero or more conditioning indicators.
            rand: provides the stream of random numbers.

        Returns:
            a Sampler object (UniformSampler).
        """
        return UniformSampler(
            rvs=(self.rvs if rvs is None else rvs),
            condition=condition,
            rand=rand,
        )

    def sample_direct(
            self,
            rvs: Optional[RandomVariable | Sequence[RandomVariable]] = None,
            *,
            condition: Condition = (),
            rand: Random = random,
            chain_pairs: Sequence[Tuple[RandomVariable, RandomVariable]] = (),
            initial_chain_condition: Condition = (),
    ) -> Sampler:
        """
        Create an inverse-transform sampler, which uses the fact that marginal
        probabilities are exactly computable with a single execution of the program.

        The sampler will yield state lists, where the state
        values are co-indexed with rvs, or self.rvs if rvs is None.

        Args:
            rvs: the list of random variables to sample; the
                yielded state vectors are co-indexed with rvs; if None,
                then the WMC rvs are used; if rvs is a single
                random variable, then single samples are yielded.
            condition: is a collection of zero or more conditioning indicators.
            rand: provides the stream of random numbers.
            chain_pairs: is a collection of pairs of random variables, each random variable
                must be in the given rvs. Given a pair (from_rv, to_rv) the state of from_rv is used
                as a condition for to_rv prior to generating a sample.
            initial_chain_condition: are condition indicators (just like condition)
                for the initialisation of the 'to_rv' random variables mentioned in chain_pairs.

        Returns:
            a Sampler object (MarginalsDirectSampler).
        """
        sampler_info: SamplerInfo = get_sampler_info(
            program_with_slotmap=self,
            rvs=rvs,
            condition=condition,
            chain_pairs=chain_pairs,
            initial_chain_condition=initial_chain_condition,
        )

        return MarginalsDirectSampler(
            sampler_info=sampler_info,
            raw_program=self._raw_program,
            rand=rand,
            rv_idx_to_result_offset=self._rv_idx_to_result_offset,
        )


def _normalise_marginal(distribution: NDArrayNumeric) -> None:
    """
    Update the values in the given distribution to
    properly represent a marginal distribution.
    """
    total = np.sum(distribution)
    if total <= 0:
        distribution[:] = np.nan
    elif total != 1:
        distribution /= total
