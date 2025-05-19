from ck import example
from ck.pgm_compiler.support.clusters import min_degree, Clusters


def main() -> None:
    pgm = example.Alarm()

    clusters: Clusters = min_degree(pgm)

    print('Elimination order:')
    for rv_idx in clusters.eliminated:
        print(f'    {pgm.rvs[rv_idx]}')
    print()

    print('Clusters:')
    for cluster in clusters.clusters:
        cluster_str = ', '.join(str(pgm.rvs[rv_idx]) for rv_idx in cluster)
        print(f'    {cluster_str}')

    print()
    print('Done.')


if __name__ == '__main__':
    main()
