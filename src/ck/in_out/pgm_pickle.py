import pickle
from pathlib import Path

from ck.pgm import PGM


def write_pickle(pgm: PGM, file) -> None:
    """
    Write the PGM as a pickle file.

    Args:
        pgm: The PGM to write.
        file: Either a file name, file Path, or an object with a write method.
    """
    if isinstance(file, (str, Path)):
        with open(file, 'wb') as f:
            pickle.dump(pgm, f)
    else:
        pickle.dump(pgm, file)


def read_pickle(file) -> PGM:
    """
    Read a PGM as a pickle file.

    Args:
        file: Either a file name, file Path, or an object with a read method.
    
    Returns:
        A PGM object.
        
    Raises:
        RuntimeError: if the unpickled object is not a PGM.
    """
    if isinstance(file, (str, Path)):
        with open(file, 'rb') as f:
            pgm = pickle.load(f)
    else:
        pgm = pickle.load(file)
    if not isinstance(pgm, PGM):
        raise RuntimeError('not a PGM object')
    return pgm
