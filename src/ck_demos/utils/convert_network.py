from pathlib import Path

from ck.in_out.parse_net import read_network
from ck.in_out.pgm_python import write_python
from ck.pgm import PGM


def convert_network(network_path: Path, file=None) -> None:
    """
    Convert a Hugin 'net' format to our PGM format.

    Args:
        network_path: path to a Hugin 'net' file.
        file: destination, as per the `print` function.
    """
    # Read the Hugin 'net' file.
    with open(network_path) as in_file:
        pgm: PGM = read_network(in_file)

    # Replace functions that may be better being sparse
    for factor in pgm.factors:
        function = factor.function
        total_params: int = function.number_of_parameters
        zero_params: int = sum(1 for _, value in function.params if value == 0)
        if zero_params > 10 and zero_params / total_params > 0.1:
            new_function = factor.set_sparse()
            for key, _, value in function.keys_with_param:
                new_function[key] = value

    # Write the PGM Python code.
    write_python(pgm, file=file)


def main() -> None:
    """
    Demo of `convert_network`.
    """
    network_directory = r'E:\Dropbox\Research\data\BN\networks'
    network_name = 'pathfinder'

    convert_network(Path(network_directory) / f'{network_name}.net')


if __name__ == '__main__':
    main()
