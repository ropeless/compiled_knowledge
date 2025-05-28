from ck import example
from ck.circuit import CircuitNode, Circuit
from ck.circuit_compiler import DEFAULT_CIRCUIT_COMPILER
from ck.pgm import PGM
from ck.pgm_circuit import PGMCircuit
from ck.pgm_compiler.factor_elimination import DEFAULT_PRODUCT_SEARCH_LIMIT, _circuit_tables_from_join_tree
from ck.pgm_compiler.support.circuit_table import CircuitTable
from ck.pgm_compiler.support.clusters import min_degree, Clusters
from ck.pgm_compiler.support.factor_tables import FactorTables, make_factor_tables
from ck.pgm_compiler.support.join_tree import JoinTree, clusters_to_join_tree
from ck.program import ProgramBuffer, RawProgram
from ck_demos.utils.stop_watch import timer


def main() -> None:
    """
    Time components of the compilation chain for factor elimination.

    Process:
        example -> PGM
        min_degree -> Clusters
        clusters_to_join_tree -> JoinTree
        join_tree_to_circuit -> PGMCircuit
        default circuit compiler -> RawProgram
        execute program
    """
    with timer('make PGM') as make_pgm_time:
        pgm: PGM = example.Mildew()

    with timer('make clusters') as make_clusters_time:
        clusters: Clusters = min_degree(pgm)

    with timer('make join tree') as make_join_tree_time:
        join_tree: JoinTree = clusters_to_join_tree(clusters)

    with timer('make factor tables') as make_factor_tables_time:
        factor_tables: FactorTables = make_factor_tables(
            pgm=pgm,
            const_parameters=True,
            multiply_indicators=True,
            pre_prune_factor_tables=False,
        )

    with timer('make circuit tables') as make_circuit_tables_time:
        top_table: CircuitTable = _circuit_tables_from_join_tree(
            factor_tables,
            join_tree,
            DEFAULT_PRODUCT_SEARCH_LIMIT,
        )
    top: CircuitNode = top_table.top()
    circuit: Circuit = top.circuit

    orig_size = circuit.number_of_op_nodes
    with timer('remove unreachable nodes') as remove_unreachable_time:
        circuit.remove_unreachable_op_nodes(top)
    print(f'    saving  {orig_size - circuit.number_of_op_nodes:10,}')
    print(f'    leaving {circuit.number_of_op_nodes:10,}')

    with timer('make PGMCircuit') as make_pgm_time:
        pgm_circuit = PGMCircuit(
            rvs=tuple(pgm.rvs),
            conditions=(),
            circuit_top=top,
            number_of_indicators=factor_tables.number_of_indicators,
            number_of_parameters=factor_tables.number_of_parameters,
            slot_map=factor_tables.slot_map,
            parameter_values=factor_tables.parameter_values,
        )

    with timer('make program') as make_program_time:
        program: RawProgram = DEFAULT_CIRCUIT_COMPILER(pgm_circuit.circuit_top)

    program_buffer = ProgramBuffer(program)
    with timer('execute program') as execute_program_time:
        program_buffer.compute()

    print()
    print(f'make PGM            {make_pgm_time.seconds():5.2f}')
    print(f'make clusters       {make_clusters_time.seconds():5.2f}')
    print(f'make join_tree      {make_join_tree_time.seconds():5.2f}')
    print(f'make factor tables  {make_factor_tables_time.seconds():5.2f}')
    print(f'make circuit tables {make_circuit_tables_time.seconds():5.2f}')
    print(f'remove unreachables {remove_unreachable_time.seconds():5.2f}')
    print(f'make PGM circuit    {make_pgm_time.seconds():5.2f}')
    print(f'make program        {make_program_time.seconds():5.2f}')
    print(f'execute program     {execute_program_time.seconds():5.2f}')

    print()
    print('Done.')


if __name__ == '__main__':
    main()
