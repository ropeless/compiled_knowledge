from __future__ import annotations

from typing import List, Sequence

from ck.circuit import CircuitNode
from ck.pgm import PGM
from ck.pgm_circuit import PGMCircuit
from ck.pgm_compiler.support import clusters
from ck.pgm_compiler.support.circuit_table import CircuitTable, product, sum_out
from ck.pgm_compiler.support.clusters import ClusterAlgorithm
from ck.pgm_compiler.support.factor_tables import make_factor_tables, FactorTables

# Standard cluster algorithms.
MIN_DEGREE: ClusterAlgorithm = clusters.min_degree
MIN_FILL: ClusterAlgorithm = clusters.min_fill
MIN_DEGREE_THEN_FILL: ClusterAlgorithm = clusters.min_degree_then_fill
MIN_FILL_THEN_DEGREE: ClusterAlgorithm = clusters.min_fill_then_degree
MIN_WEIGHTED_DEGREE: ClusterAlgorithm = clusters.min_weighted_degree
MIN_WEIGHTED_FILL: ClusterAlgorithm = clusters.min_weighted_fill
MIN_TRADITIONAL_WEIGHTED_FILL: ClusterAlgorithm = clusters.min_traditional_weighted_fill


def compile_pgm(
        pgm: PGM,
        const_parameters: bool = True,
        *,
        algorithm: ClusterAlgorithm = MIN_FILL_THEN_DEGREE,
        pre_prune_factor_tables: bool = False,
) -> PGMCircuit:
    """
    Compile the PGM to an arithmetic circuit, using variable elimination.

    Conforms to the `PGMCompiler` protocol.

    Args:
        pgm: The PGM to compile.
        const_parameters: If true, the potential function parameters will be circuit
            constants, otherwise they will be circuit variables.
        algorithm: algorithm to get an elimination order.
        pre_prune_factor_tables: if true, then heuristics will be used to remove any provably zero row.

    Returns:
        a PGMCircuit object.
    """
    factor_tables: FactorTables = make_factor_tables(
        pgm=pgm,
        const_parameters=const_parameters,
        multiply_indicators=True,
        pre_prune_factor_tables=pre_prune_factor_tables,
    )

    elimination_order: Sequence[int] = algorithm(pgm).eliminated

    # Eliminate rvs from the factor tables according to the
    # elimination order.
    cur_tables: List[CircuitTable] = list(factor_tables.tables)
    for rv_idx in elimination_order:
        next_tables: List[CircuitTable] = []
        tables_with_rv: List[CircuitTable] = []
        for table in cur_tables:
            if rv_idx in table.rv_idxs:
                tables_with_rv.append(table)
            else:
                next_tables.append(table)
        if len(tables_with_rv) > 0:
            while len(tables_with_rv) > 1:
                # product the two smallest tables
                tables_with_rv.sort(key=lambda _t: -len(_t))
                x = tables_with_rv.pop()
                y = tables_with_rv.pop()
                tables_with_rv.append(product(x, y))
            next_tables.append(sum_out(tables_with_rv[0], (rv_idx,)))
            cur_tables = next_tables

    # All rvs are now eliminated - all tables should have a single top.
    tops: List[CircuitNode] = [
        table.top()
        for table in cur_tables
    ]
    top: CircuitNode = factor_tables.circuit.optimised_mul(tops)
    top.circuit.remove_unreachable_op_nodes(top)

    return PGMCircuit(
        rvs=tuple(pgm.rvs),
        conditions=(),
        circuit_top=top,
        number_of_indicators=factor_tables.number_of_indicators,
        number_of_parameters=factor_tables.number_of_parameters,
        slot_map=factor_tables.slot_map,
        parameter_values=factor_tables.parameter_values,
    )
