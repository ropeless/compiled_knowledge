from __future__ import annotations

import random
from typing import Sequence, Optional, Tuple

from ck.circuit_compiler import CircuitCompiler
from ck.pgm import RandomVariable
from ck.pgm_circuit import PGMCircuit
from ck.pgm_circuit.program_with_slotmap import ProgramWithSlotmap
from ck.pgm_circuit.support.compile_circuit import compile_results, DEFAULT_CIRCUIT_COMPILER
from ck.probability.probability_space import ProbabilitySpace, Condition
from ck.program.program_buffer import ProgramBuffer
from ck.program.raw_program import RawProgram
from ck.sampling.sampler import Sampler
from ck.sampling.sampler_support import SamplerInfo, get_sampler_info
from ck.sampling.uniform_sampler import UniformSampler
from ck.sampling.wmc_direct_sampler import WMCDirectSampler
from ck.sampling.wmc_gibbs_sampler import WMCGibbsSampler
from ck.sampling.wmc_metropolis_sampler import WMCMetropolisSampler
from ck.sampling.wmc_rejection_sampler import WMCRejectionSampler
from ck.utils.np_extras import NDArray
from ck.utils.random_extras import Random


class WMCProgram(ProgramWithSlotmap, ProbabilitySpace):
    """
    A class for computing Weighted Model Count (WMC).
    """

    def __init__(
            self,
            pgm_circuit: PGMCircuit,
            const_parameters: bool = True,
            compiler: CircuitCompiler = DEFAULT_CIRCUIT_COMPILER,
    ):
        """
        Construct a WMCProgram object.

        Args:
            pgm_circuit: The circuit representing a PGM.
            const_parameters: if True then any circuit variable representing a parameter value will
                be made 'const' in the resulting program.
        """
        raw_program: RawProgram = compile_results(
            pgm_circuit=pgm_circuit,
            results=(pgm_circuit.circuit_top,),
            const_parameters=const_parameters,
            compiler=compiler,
        )
        ProgramWithSlotmap.__init__(
            self,
            ProgramBuffer(raw_program),
            pgm_circuit.slot_map,
            pgm_circuit.rvs,
            pgm_circuit.conditions,
        )
        self._raw_program: RawProgram = raw_program
        self._number_of_indicators: int = pgm_circuit.number_of_indicators
        self._z_cache: Optional[float] = None

        if not const_parameters:
            # set the parameter slots
            self.vars[pgm_circuit.number_of_indicators:] = pgm_circuit.parameter_values

    def wmc(self, *condition: Condition) -> float:
        self.set_condition(*condition)
        return self.compute().item()

    @property
    def z(self) -> float:
        if self._z_cache is None:
            number_of_indicators: int = self._number_of_indicators
            slots: NDArray = self.vars
            old_vals: NDArray = slots[:number_of_indicators].copy()
            slots[:number_of_indicators] = 1
            self._z_cache = self.compute().item()
            slots[:number_of_indicators] = old_vals

        return self._z_cache

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
        Create an inverse-transform sampler, which uses the fact that
        probabilities are exactly computable using a WMC.

        The sampler will yield state lists, where the state
        values are co-indexed with rvs, or self.rvs if rvs is None.

        Given 'n' random variables, and 'm' number of indicators, for each yielded sample, this method:
        * calls rand.random() once and rand.randrange(...) n times,
        * calls self.program().compute_result() at least once and <= 1 + m.

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
            a Sampler object (WMCDirectSampler).
        """
        sampler_info: SamplerInfo = get_sampler_info(
            program_with_slotmap=self,
            rvs=rvs,
            condition=condition,
            chain_pairs=chain_pairs,
            initial_chain_condition=initial_chain_condition,
        )

        return WMCDirectSampler(
            sampler_info=sampler_info,
            raw_program=self._raw_program,
            rand=rand,
        )

    def sample_rejection(
            self,
            rvs: Optional[RandomVariable | Sequence[RandomVariable]] = None,
            *,
            condition: Condition = (),
            rand: Random = random,
    ) -> Sampler:
        """
        Create a sampler to perform rejection sampling.

        The sampler will yield state lists, where the state
        values are co-indexed with rvs, or self.rvs if rvs is None.

        The method uniformly samples states and uses an adaptive 'max weight'
        to reduce unnecessary rejection.

        After each sample is yielded, the WMC indicator variables will
        be left set as per the yielded states of rvs and conditions.

        Args:
            rvs: the list of random variables to sample; the
                yielded state vectors are co-indexed with rvs; if None,
                then the WMC rvs are used; if rvs is a single
                random variable, then single samples are yielded.
            condition: is a collection of zero or more conditioning indicators.
            rand: provides the stream of random numbers.

        Returns:
            a Sampler object (WMCRejectionSampler).
        """
        sampler_info: SamplerInfo = get_sampler_info(
            program_with_slotmap=self,
            rvs=rvs,
            condition=condition,
        )
        z = self.wmc(*condition)

        return WMCRejectionSampler(
            sampler_info=sampler_info,
            raw_program=self._raw_program,
            rand=rand,
            z=z,
        )

    def sample_gibbs(
            self,
            rvs: Optional[RandomVariable | Sequence[RandomVariable]] = None,
            *,
            condition: Condition = (),
            skip: int = 0,
            burn_in: int = 0,
            pr_restart: float = 0,
            rand: Random = random,
    ) -> Sampler:
        """
        Create a sampler to perform Gibbs sampling.

        The sampler will yield state lists, where the state
        values are co-indexed with rvs, or self.rvs if rvs is None.

        After each sample is yielded, the WMC indicator vars will
        be left set as per the yielded states of rvs and conditions.

        Args:
            rvs: the list of random variables to sample; the
                yielded state vectors are co-indexed with rvs; if None,
                then the WMC rvs are used; if rvs is a single
                random variable, then single samples are yielded.
            condition: is a collection of zero or more conditioning indicators.
            skip: is an integer >= 0 specifying how may samples to discard
                for each sample provided. Values > 0 can be used to de-correlate adjacent samples.
            burn_in: how many iterations to perform after
                initialisation before yielding a sample.
            pr_restart: the chance of re-initialising each
                iteration. If restarted then burn-in is performed again.
            rand: provides the stream of random numbers.

        Returns:
            a Sampler object (WMCGibbsSampler).
        """
        if skip < 0:
            raise RuntimeError('skip must be non-negative')
        if burn_in < 0:
            raise RuntimeError('burn_in must be non-negative')

        sampler_info: SamplerInfo = get_sampler_info(
            program_with_slotmap=self,
            rvs=rvs,
            condition=condition,
        )

        return WMCGibbsSampler(
            sampler_info=sampler_info,
            raw_program=self._raw_program,
            rand=rand,
            skip=skip,
            burn_in=burn_in,
            pr_restart=pr_restart,
        )

    def sample_metropolis(
            self,
            rvs: Optional[RandomVariable | Sequence[RandomVariable]] = None,
            *,
            condition: Condition = (),
            skip: Optional[int] = None,
            burn_in: int = 0,
            pr_restart: float = 0,
            rand: Random = random,
    ) -> Sampler:
        """
        Create a sampler to perform Metropolis-Hastings sampling.

        The sampler will yield state lists, where the state
        values are co-indexed with rvs, or self.rvs if rvs is None.

        After each sample is yielded, the WMC indicator vars will
        be left set as per the yielded states of rvs and conditions.

        Args:
            rvs: the list of random variables to sample; the
                yielded state vectors are co-indexed with rvs; if None,
                then the WMC rvs are used; if rvs is a single
                random variable, then single samples are yielded.
            condition: is a collection of zero or more conditioning indicators.
            skip: is an optional integer >= 0 specifying how may samples to discard
                for each sample provided. Values > 0 can be used to de-correlate adjacent samples.
                Default value = len(rvs)
            burn_in: how many iterations to perform after initialisation
                before yielding a sample.
            pr_restart: the chance of re-initialising each iteration. If
                restarted then burn-in is performed again.
            rand: provides the stream of random numbers.

        Returns:
            a Sampler object (WMCMetropolisSampler).
        """
        if skip is not None and skip < 0:
            raise RuntimeError('skip must be non-negative')
        if burn_in < 0:
            raise RuntimeError('burn_in must be non-negative')

        sampler_info: SamplerInfo = get_sampler_info(
            program_with_slotmap=self,
            rvs=rvs,
            condition=condition,
        )

        if skip is None:
            skip = len(sampler_info.sample_rvs)

        return WMCMetropolisSampler(
            sampler_info=sampler_info,
            raw_program=self._raw_program,
            rand=rand,
            skip=skip,
            burn_in=burn_in,
            pr_restart=pr_restart,
        )
