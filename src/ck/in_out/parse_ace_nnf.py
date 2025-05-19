from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, List

import numpy as np

from ck.circuit import Circuit, CircuitNode, VarNode, ConstValue, ConstNode
from ck.in_out.parse_ace_lmap import LiteralMap
from ck.in_out.parser_utils import ParseError, ParserInput
from ck.pgm_circuit.slot_map import SlotKey, SlotMap
from ck.utils.np_extras import NDArrayFloat64

VAR_NODE = {"l", "L"}
ADD_NODE = {"O", "+"}
MUL_NODE = {"A", "*"}


def read_nnf_with_literal_map(
        input_stream,
        literal_map: LiteralMap,
        *,
        const_parameters: bool = True,
        optimise_ops: bool = True,
        circuit: Optional[Circuit] = None,
        check_header: bool = False,
) -> Tuple[CircuitNode, SlotMap, NDArrayFloat64]:
    """
    Parse an input, as per `read_nnf`, using the given literal map to
    create a slot map with indicator entries.

    See: `read_nnf` and `read_lmap` for more information.

    Args:
        input_stream: to parse, as per `ParserInput` argument.
        literal_map: mapping from literal code to indicators.
        circuit: an optional circuit to reuse.
        check_header: if true, an exception is raised if the number of nodes or arcs is not as expected.
        const_parameters: if true, the potential function parameters will be circuit
            constants, otherwise they will be circuit variables.
        optimise_ops: if true then circuit optimised operations will be used.

    Returns:
        (circuit_top, slot_map, params)
        circuit_top: is the resulting top node from parsing the input.
        slot_map: is a map from indicator to a circuit var index (int).
        params: is a numpy array of parameter values, co-indexed with `circuit.vars[num_indicators:]`
    """

    # Set the `const_literals` parameter for `read_nnf`
    const_literals: Optional[Dict[int, ConstValue]]
    if const_parameters:
        indicator_literal_code = literal_map.indicators.keys()
        const_literals = {
            literal_code: value
            for literal_code, value in literal_map.params.items()
            if literal_code not in indicator_literal_code
        }
    else:
        const_literals = {}

    top_node, literal_slot_map = read_nnf(
        input_stream,
        const_literals=const_literals,
        circuit=circuit,
        check_header=check_header,
        optimise_ops=optimise_ops,
    )

    # Build the slot map from indicator to slot
    slot_map: Dict[SlotKey, int] = {
        indicator: literal_slot_map[literal_code]
        for literal_code, indicator in literal_map.indicators.items()
    }

    # Get the parameter values
    num_indicators: int = len(literal_map.indicators)
    num_parameters: int = top_node.circuit.number_of_vars - num_indicators
    assert num_parameters == 0 or not const_parameters, 'const_parameters -> num_parameters == 0'

    params: NDArrayFloat64 = np.zeros(num_parameters, dtype=np.float64)
    for literal_code, value in literal_map.params.items():
        literal_slot: Optional[int] = literal_slot_map.get(literal_code)
        if literal_slot is not None and literal_slot >= num_indicators:
            params[literal_slot - num_indicators] = value

    return top_node, slot_map, params


def read_nnf(
        input_stream,
        *,
        const_literals: Optional[Dict[int, ConstValue]] = None,
        circuit: Optional[Circuit] = None,
        check_header: bool = False,
        optimise_ops: bool = True,
) -> Tuple[CircuitNode, Dict[int, int]]:
    """
    Parse the input_stream (file or string) as "nnf" or "ac" file format describing a circuit.

    The input consists of propositional logical sentences in negative normal form (NNF).
    This covers both ".ac" and ".nnf" files produced by the software ACE.

    The returned slot map will have entries to get circuit vars for literal codes.
    E.g., given a literal code (int), use `circuit.vars[slot_map[literal_code]]` to get its var node.

    Software may simplify an NNF file by removing arcs, but it may not update the header.
    Although the resulting file is not conformant, it is still parsable.
    Parameter `check_header` can be set to false, which prevents an exception being raised.

    Args:
        input_stream: to parse, as per `ParserInput` argument.
        const_literals: an optional mapping from literal code to constant value.
        circuit: an optional empty circuit to reuse.
        check_header: if true, an exception is raised if the number of nodes or arcs is not as expected.
        optimise_ops: if true then circuit optimised operations will be used.

    Returns:
        (circuit_top, literal_slot_map)
        circuit_top: is the resulting top node from parsing the input.
        literal_slot_map: is a mapping from literal code (int) to a circuit var index (int).

    Assumes:
        If a circuit is provided, it is empty.
    """
    if circuit is None:
        circuit: Circuit = Circuit()

    if circuit.number_of_vars != 0:
        raise ValueError('the given circuit must be empty')

    if const_literals is None:
        const_literals: Dict[int, ConstValue] = {}

    parser = CircuitParser(circuit, check_header, const_literals, optimise_ops)
    parser.parse(input_stream)

    nodes = parser.nodes
    cct_top = circuit.zero if len(nodes) == 0 else nodes[-1]

    return cct_top, parser.literal_slot_map


class Parser(ABC):

    def parse(self, input_stream):
        input_stream = ParserInput(input_stream)
        raise_f = lambda msg: input_stream.raise_error(msg)
        try:
            state = 0
            line = input_stream.readline()
            while line and state < 999:
                line = line.strip()
                if len(line) > 0:
                    if state == 0:
                        if line[0] == 'c':
                            self.comment(raise_f, line)
                        else:
                            line = line.split()
                            if line[0] == 'nnf':
                                if len(line) != 4:
                                    raise_f('expected: nnf <num-nodes> <num-edges> <num-?>')
                                self.header(raise_f, int(line[1]), int(line[2]), int(line[3]))
                                state = 1
                            else:
                                raise_f('expected: "nnf"')
                    elif state == 1:
                        if line[0] == '%':
                            state = 999
                        else:
                            line = line.split()
                            code = line[0]
                            if code in VAR_NODE:
                                if len(line) != 2:
                                    raise_f(f'expected: {code} <literal-code>')
                                self.literal(raise_f, int(line[1]))
                            else:
                                is_add = code in ADD_NODE
                                is_mul = code in MUL_NODE
                                if not (is_add or is_mul):
                                    raise_f(f'unexpected line starting with: {code}')
                                if len(line) < 2:
                                    raise_f(f'expected: {code} <num_args> <arguments>...')
                                num_args_idx = 2 if code == 'O' else 1
                                num_args = int(line[num_args_idx])
                                args = [int(arg) for arg in line[num_args_idx + 1:]]
                                if len(args) != num_args:
                                    raise_f(f'unexpected number of args for: {code}')
                                if is_add:
                                    self.add_node(raise_f, args)
                                else:
                                    self.mul_node(raise_f, args)
                line = input_stream.readline()
            self.done(raise_f)
        except ParseError as e:
            raise e
        except Exception as e:
            input_stream.raise_error(e)

    @abstractmethod
    def comment(self, raise_f, message: str) -> None:
        ...

    @abstractmethod
    def header(self, raise_f, num_nodes: int, num_edges: int, num_: int):
        ...

    @abstractmethod
    def literal(self, raise_f, literal_code: int) -> None:
        ...

    @abstractmethod
    def add_node(self, raise_f, args: List[int]) -> None:
        ...

    @abstractmethod
    def mul_node(self, raise_f, args: List[int]) -> None:
        ...

    @abstractmethod
    def done(self, raise_f) -> None:
        ...


class CircuitParser(Parser):

    def __init__(
            self,
            circuit: Circuit,
            check_header: bool,
            const_literals: Dict[int, ConstValue],
            optimise_ops: bool,
    ):
        self.check_header: bool = check_header
        self.literal_slot_map: Dict[SlotKey, int] = {}
        self.optimise_ops = optimise_ops
        self.circuit = circuit
        self.const_literals: Dict[int, ConstValue] = const_literals
        self.nodes: List[CircuitNode] = []
        self.num_nodes = None
        self.num_edges = None

    def comment(self, raise_f, message: str) -> None:
        pass

    def literal(self, raise_f, literal_code: int) -> None:
        """
        Makes either a VarNode or a ConstNode.
        """
        const_value: Optional[ConstValue] = self.const_literals.get(literal_code)
        if const_value is not None:
            node: ConstNode = self.circuit.const(const_value)
        elif literal_code in self.literal_slot_map.keys():
            raise_f(f'duplicated literal code: {literal_code}')
            return
        else:
            node: VarNode = self.circuit.new_var()
            self.literal_slot_map[literal_code] = node.idx
        self.nodes.append(node)

    def add_node(self, raise_f, args: List[int]) -> None:
        """
        Makes a AddNode (or other if optimised).
        """
        arg_nodes = [self.nodes[arg] for arg in args]
        if self.optimise_ops:
            self.nodes.append(self.circuit.optimised_add(arg_nodes))
        else:
            self.nodes.append(self.circuit.add(arg_nodes))

    def mul_node(self, raise_f, args: List[int]) -> None:
        """
        Makes a MulNode (or other if optimised).
        """
        arg_nodes = [self.nodes[arg] for arg in args]
        if self.optimise_ops:
            self.nodes.append(self.circuit.optimised_mul(arg_nodes))
        else:
            self.nodes.append(self.circuit.mul(arg_nodes))

    def header(self, raise_f, num_nodes: int, num_edges: int, num_: int) -> None:
        self.num_nodes = num_nodes
        self.num_edges = num_edges

    def done(self, raise_f) -> None:
        if self.check_header:
            if len(self.nodes) != self.num_nodes:
                raise_f(f'unexpected number of nodes: {len(self.nodes)} expected: {self.num_nodes}')
            if self.circuit.number_of_arcs != self.num_edges:
                raise_f(f'unexpected number of arcs: {self.circuit.number_of_arcs} expected: {self.num_edges}')
