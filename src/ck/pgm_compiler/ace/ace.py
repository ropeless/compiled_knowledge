import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

from ck.circuit import CircuitNode, Circuit
from ck.in_out.parse_ace_lmap import read_lmap, LiteralMap
from ck.in_out.parse_ace_nnf import read_nnf_with_literal_map
from ck.in_out.render_net import render_bayesian_network
from ck.pgm import PGM
from ck.pgm_circuit import PGMCircuit
from ck.pgm_circuit.slot_map import SlotMap
from ck.utils.local_config import config
from ck.utils.np_extras import NDArrayFloat64
from ck.utils.tmp_dir import tmp_dir


def compile_pgm(
        pgm: PGM,
        const_parameters: bool = True,
        *,
        ace_dir: Optional[Path | str] = None,
        jar_dir: Optional[Path | str] = None,
        print_output: bool = False,
        m_bytes: int = 1512,
        check_is_bayesian_network: bool = True,
) -> PGMCircuit:
    """
    Compile the PGM to an arithmetic circuit, using Ace.

    This is a wrapper for Ace.
    Ace compiles a Bayesian network into an Arithmetic Circuit.
    Provided by the Automated Reasoning Group, University of California Los Angeles.
    Ace requires the Java Runtime Environment (JRE) version 8 or higher.
    See http://reasoning.cs.ucla.edu/ace/

    Conforms to the `PGMCompiler` protocol.

    Args:
        pgm: The PGM to compile.
        const_parameters: If true, the potential function parameters will be circuit
            constants, otherwise they will be circuit variables.
        ace_dir: Directory containing Ace. If not provided then the directory this module is in is used.
        jar_dir: Directory containing Ace jar files. If not provided, then `ace_dir` is used.
        print_output: if true, the output from Ace is printed.
        m_bytes: requested megabytes for the Java Virtual Machine (using the java "-Xmx" argument).
        check_is_bayesian_network: if true, then the PGM will be checked to confirm it is a Bayesian network.

    Returns:
        a PGMCircuit object.

    Raises:
        RuntimeError: if Ace files are not found, including a helpful message.
        ValueError: if `check_is_bayesian_network` is true and the PGM is not a Bayesian network.
        CalledProcessError: if executing Ace failed.
    """
    if check_is_bayesian_network and not pgm.check_is_bayesian_network():
        raise ValueError('the given PGM is not a Bayesian network')

    # ACE cannot deal with the empty PGM even though it is a valid Bayesian network
    if pgm.number_of_factors == 0:
        circuit = Circuit()
        circuit.new_vars(pgm.number_of_indicators)
        parameter_values = np.array([], dtype=np.float64)
        slot_map = {indicator: i for i, indicator in enumerate(pgm.indicators)}
        return PGMCircuit(
            rvs=pgm.rvs,
            conditions=(),
            circuit_top=circuit.const(1),
            number_of_indicators=pgm.number_of_indicators,
            number_of_parameters=0,
            slot_map=slot_map,
            parameter_values=parameter_values,
        )

    java: str
    classpath_separator: str
    java, classpath_separator = _find_java()
    files: _AceFiles = _find_ace_files(ace_dir, jar_dir)
    net_file_name = 'to_compile.net'
    main_class = 'edu.ucla.belief.ace.AceCompile'
    class_path: str = classpath_separator.join(
        str(f) for f in [files.ace_jar, files.inflib_jar, files.jdom_jar]
    )
    ace_cmd: List[str] = [
        java,
        '-cp',
        class_path,
        f'-DACEC2D={files.c2d}',
        f'-Xmx{int(m_bytes)}m',
        main_class,
        net_file_name,
    ]

    with tmp_dir():
        # Render the PGM to a .net file to be read by Ace
        with open(net_file_name, 'w') as file:
            node_names: List[str] = render_bayesian_network(pgm, file, check_structure_bayesian=False)

        # Run Ace
        ace_result: subprocess.CompletedProcess = subprocess.run(ace_cmd, capture_output=(not print_output), text=True)
        if ace_result.returncode != 0:
            raise subprocess.CalledProcessError(
                returncode=ace_result.returncode,
                cmd=' '.join(ace_cmd),
                output=None if print_output else ace_result.stdout,
                stderr=None if print_output else ace_result.stderr,
            )

        # Parse the literal map output from Ace
        with open(f'{net_file_name}.lmap', 'r') as file:
            literal_map: LiteralMap = read_lmap(file, node_names=node_names)

        # Parse the arithmetic circuit output from Ace
        with open(f'{net_file_name}.ac', 'r') as file:
            circuit_top: CircuitNode
            slot_map: SlotMap
            parameter_values: NDArrayFloat64
            circuit_top, slot_map, parameter_values = read_nnf_with_literal_map(
                file,
                indicators=pgm.indicators,
                literal_map=literal_map,
                const_parameters=const_parameters,
            )

    # Consistency checking
    number_of_indicators: int = pgm.number_of_indicators
    number_of_parameters: int = parameter_values.shape[0]
    assert circuit_top.circuit.number_of_vars == number_of_indicators + number_of_parameters, 'consistency check'

    return PGMCircuit(
        rvs=pgm.rvs,
        conditions=(),
        circuit_top=circuit_top,
        number_of_indicators=number_of_indicators,
        number_of_parameters=number_of_parameters,
        slot_map=slot_map,
        parameter_values=parameter_values,
    )


def ace_available(
        ace_dir: Optional[Path | str] = None,
        jar_dir: Optional[Path | str] = None,
) -> bool:
    """
    Returns:
        True if it looks like ACE is available, False otherwise.
        ACE is available if ACE files are in the default location and Java is available.
    """
    try:
        java: str
        java, _ = _find_java()
        _: _AceFiles = _find_ace_files(ace_dir, jar_dir)

        java_cmd: List[str] = [java, '--version',]
        java_result: subprocess.CompletedProcess = subprocess.run(java_cmd, capture_output=True, text=True)

        return java_result.returncode == 0

    except RuntimeError:
        return False


def copy_ace_to_default_location(
        ace_dir: Path | str,
        jar_dir: Optional[Path | str] = None,
) -> None:
    """
    Copy Ace files from the given directories into the default directory.

    Args:
        ace_dir: Directory containing Ace.
        jar_dir: Directory containing Ace jar files. If not provided, then `ace_dir` is used.

    Raises:
        RuntimeError: if Ace files are not found, including a helpful message .
        IOError: if the copy fails.

    Assumes:
        ace_dir exists and is not the same as the installation directory.
    """
    install_location: Path = default_ace_location()

    if ace_dir is None or ace_dir == install_location:
        raise RuntimeError(f'Ace directory cannot be the default directory')

    files: _AceFiles = _find_ace_files(ace_dir, jar_dir)

    to_copy = [files.ace_jar, files.inflib_jar, files.jdom_jar] + files.c2d_options

    for file in to_copy:
        shutil.copyfile(file, install_location / file.name)


def default_ace_location() -> Path:
    """
    Get the default location for Ace files.

    This function checks the local config for the variable
    CK_ACE_LOCATION. If that is not available, then the
    directory that this Python module is in will be used.
    """
    return Path(config.get('CK_ACE_LOCATION', Path(__file__).parent))


@dataclass
class _AceFiles:
    ace_jar: Path
    inflib_jar: Path
    jdom_jar: Path
    c2d: Path
    c2d_options: List[Path]


def _find_java() -> Tuple[str, str]:
    """
    What to call the Java executable and classpath separator.

    Returns:
        (java, classpath_separator)

    Raises:
        RuntimeError: if not found, including a helpful message.
    """
    if sys.platform == 'win32':
        return 'java.exe', ';'
    elif sys.platform == 'darwin':
        return 'java', ':'
    elif sys.platform.startswith('linux'):
        return 'java', ':'
    else:
        raise RuntimeError(f'cannot infer java for platform {sys.platform!r}')


def _find_ace_files(
        ace_dir: Optional[Path | str],
        jar_dir: Optional[Path | str],
) -> _AceFiles:
    """
    Look for the needed Ace files.

    Raises:
        RuntimeError: if not found, including a helpful message.
    """
    ace_dir: Path = default_ace_location() if ace_dir is None else Path(ace_dir)
    jar_dir: Path = ace_dir if jar_dir is None else Path(jar_dir)

    if not ace_dir.is_dir():
        raise RuntimeError(f'Ace directory does not exist: {ace_dir}')
    if not jar_dir.is_dir():
        raise RuntimeError(f'Ace jar directory does not exist: {jar_dir}')

    ace_jar = jar_dir / 'ace.jar'
    inflib_jar = jar_dir / 'inflib.jar'
    jdom_jar = jar_dir / 'jdom.jar'

    missing: List[str] = [
        jar.name
        for jar in [ace_jar, inflib_jar, jdom_jar]
        if not jar.is_file()
    ]
    if len(missing) > 0:
        raise RuntimeError(f'Ace jars missing (ensure Ace is properly installed): {", ".join(missing)}')

    c2d_options: List[Path] = [
        file
        for file in ace_dir.iterdir()
        if file.is_file() and file.name.startswith('c2d')
    ]
    c2d: Path
    if len(c2d_options) == 0:
        raise RuntimeError(f'cannot find c2d in the Ace directory: {ace_dir}')
    if len(c2d_options) == 1:
        c2d = next(iter(c2d_options))
    else:
        if sys.platform == 'win32':
            c2d = ace_dir / 'c2d_windows.exe'
        elif sys.platform == 'darwin':
            c2d = ace_dir / 'c2d_osx'
        elif sys.platform.startswith('linux'):
            c2d = ace_dir / 'c2d_linux'
        else:
            raise RuntimeError(f'cannot infer c2d executable name for platform {sys.platform!r}')

    if not c2d.is_file():
        raise RuntimeError(f'cannot find c2d: {c2d}')

    return _AceFiles(
        c2d=c2d,
        c2d_options=c2d_options,
        ace_jar=ace_jar,
        inflib_jar=inflib_jar,
        jdom_jar=jdom_jar,
    )
