from ck import example
from ck.circuit import Circuit
from ck.pgm import PGM
from ck.pgm_circuit import PGMCircuit
from ck.pgm_circuit.wmc_program import WMCProgram
from ck.pgm_compiler.factor_elimination import join_tree_to_circuit
from ck.pgm_compiler.support.clusters import min_degree, Clusters
from ck.pgm_compiler.support.join_tree import JoinTree, clusters_to_join_tree


def main() -> None:
    """
    This demo shows the full compilation chain for factor elimination.

    Process:
        Rain example -> PGM
        min_degree -> Clusters
        clusters_to_join_tree -> JoinTree
        join_tree_to_circuit -> PGMCircuit
        default circuit compiler -> WMCProgram
    """
    pgm: PGM = example.Rain()

    print(f'PGM {pgm.name!r}')
    print()

    clusters: Clusters = min_degree(pgm)

    clusters.dump()

    join_tree: JoinTree = clusters_to_join_tree(clusters)

    print('Join Tree:')
    join_tree.dump()
    print()

    pgm_circuit: PGMCircuit = join_tree_to_circuit(join_tree)
    circuit: Circuit = pgm_circuit.circuit_top.circuit

    print('Circuit:')
    circuit.dump()
    print()

    wmc = WMCProgram(pgm_circuit)

    print()
    print('Showing Program Results:')
    print(f'  {"State":80} {"WMC":8} {"PGM":8}')
    for indicators in pgm.instances_as_indicators():
        instance_as_str = pgm.indicator_str(*indicators)
        wmc_value = wmc.wmc(*indicators)
        pgm_value = pgm.value_product_indicators(*indicators)
        print(f'  {instance_as_str:80} {wmc_value:8.6f} {pgm_value:8.6f}')

    print()
    print('Done.')


if __name__ == '__main__':
    main()
