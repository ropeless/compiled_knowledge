"""
This is a pure Python implementation of Circuits (for testing and development)

For more documentation on this module, refer to the Jupyter notebook docs/6_circuits_and_programs.ipynb.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import chain
from typing import Dict, Tuple, Optional, Iterable, Sequence, List, overload, Iterator, Set

# Type for values of ConstNode objects
ConstValue = float | int | bool

# Symbols for op nodes
ADD: int = 0
MUL: int = 1


class Circuit:
    """
    An arithmetic circuit defines an arithmetic function from input variables (`VarNode` objects)
    and constant values (`ConstNode` objects) to one or more result values. Computation is defined
    over a mathematical ring, with two operations: addition and multiplication (represented
    by `OpNode` objects).

    An arithmetic circuit needs to be compiled to a program to execute the function.

    All nodes belong to a circuit. All nodes are immutable, with the exception that a
    `VarNode` may be temporarily be set to a constant value.
    """

    def __init__(self, zero: ConstValue = 0, one: ConstValue = 1):
        """
        Construct a new, empty circuit.

        Args:
            zero: The constant value for zero. mul(x, zero) = zero, add(x, zero) = x.
            one: The constant value for one. mul(x, one) = x.
        """
        self._vars: List[VarNode] = []
        self._ops: List[OpNode] = []
        self._const_map: Dict[ConstValue, ConstNode] = {}
        self.__derivatives: Optional[_DerivativeHelper] = None  # cache for partial derivatives calculations.
        self._zero: ConstNode = self.const(zero)
        self._one: ConstNode = self.const(one)

    @property
    def number_of_vars(self) -> int:
        """
        Returns:
            the number of "var" nodes.
        """
        return len(self._vars)

    @property
    def number_of_consts(self) -> int:
        """
        Returns:
            the number of "const" nodes.
        """
        return len(self._const_map)

    @property
    def number_of_op_nodes(self) -> int:
        """
        Returns:
            the number of "op" nodes.
        """
        return len(self._ops)

    @property
    def number_of_arcs(self) -> int:
        """
        Returns:
            the number of arcs in the circuit, i.e., the sum of the
            number of arguments for all op nodes.
        """
        return sum(len(op.args) for op in self._ops)

    @property
    def number_of_operations(self):
        """
        How many op nodes are in the circuit.
        This is number_of_arcs - number_of_op_nodes.
        """
        return self.number_of_arcs - self.number_of_op_nodes

    @property
    def vars(self) -> Sequence[VarNode]:
        """
        Returns:
            the var nodes, in index order.

        Ensures:
            `self.vars[i].idx == i`.
        """
        return self._vars

    @property
    def ops(self) -> Sequence[OpNode]:
        """
        Returns:
            the op nodes, in the order they were added to this circuit.
        """
        return self._ops

    @property
    def zero(self) -> ConstNode:
        """
        Get the constant representing zero.
        """
        return self._zero

    @property
    def one(self) -> ConstNode:
        """
        Get the constant representing one.
        """
        return self._one

    def new_var(self) -> VarNode:
        """
        Create and return a new variable node.
        """
        node = VarNode(self, len(self._vars))
        self._vars.append(node)
        return node

    def new_vars(self, num_of_vars: int) -> Sequence[VarNode]:
        """
        Create and return multiple variable nodes.
        """
        offset = self.number_of_vars
        new_vars = tuple(VarNode(self, i) for i in range(offset, offset + num_of_vars))
        self._vars.extend(new_vars)
        return new_vars

    def const(self, value: ConstValue | ConstNode) -> ConstNode:
        """
        Return a const node for the given value.
        If a const node for that value already exists, then it will be returned,
        otherwise a new const node will be created.
        """
        if isinstance(value, ConstNode):
            value = value.value

        node = self._const_map.get(value)
        if node is None:
            node = ConstNode(self, value)
            self._const_map[value] = node
        return node

    def _op(self, symbol: int, args: Tuple[CircuitNode, ...]) -> OpNode:
        """
        Create and return a new op node, applied to the given arguments.
        """
        node = OpNode(self, symbol, args)
        self._ops.append(node)
        return node

    def add(self, *args: Args) -> OpNode:
        """
        Create and return a new 'addition' node, applied to the given arguments.
        """
        return self._op(ADD, self._check_nodes(args))

    def mul(self, *args: Args) -> OpNode:
        """
        Create and return a new 'multiplication' node, applied to the given arguments.
        """
        return self._op(MUL, self._check_nodes(args))

    def optimised_add(self, *args: Args) -> CircuitNode:
        """
        Optimised circuit node addition.

        Performs the following optimisations:
        * addition to zero is avoided: add(x, 0) = x,
        * singleton addition is avoided: add(x) = x,
        * empty addition is avoided: add() = 0,
        """
        to_add = tuple(n for n in self._check_nodes(args) if not n.is_zero)
        match len(to_add):
            case 0:
                return self.zero
            case 1:
                return to_add[0]
            case _:
                return self._op(ADD, to_add)

    def optimised_mul(self, *args: Args) -> CircuitNode:
        """
        Optimised circuit node multiplication.

        Performs the following optimisations:
        * multiplication by zero is avoided: mul(x, 0) = 0,
        * multiplication by one is avoided: mul(x, 1) = x,
        * singleton multiplication is avoided: mul(x) = x,
        * empty multiplication is avoided: mul() = 1,
        """
        to_mul = tuple(n for n in self._check_nodes(args) if not n.is_one)
        if any(n.is_zero for n in to_mul):
            return self.zero
        match len(to_mul):
            case 0:
                return self.one
            case 1:
                return to_mul[0]
            case _:
                return self._op(MUL, to_mul)

    def cartesian_product(self, xs: Sequence[CircuitNode], ys: Sequence[CircuitNode]) -> List[CircuitNode]:
        """
        Add multiply operations, one for each possible combination of x from xs and y from ys.

        Args:
            xs: first list of circuit nodes, may be either a Node object or a list of Nodes.
            ys: second list of circuit nodes, may be either a Node object or a list of Nodes.

        Returns:
            a list of 'mul' nodes, one for each combination of xs and ys. The results are in the order
            given by `[mul(x, y) for x in xs for y in ys]`.
        """
        xs: Sequence[CircuitNode] = self._check_nodes(xs)
        ys: Sequence[CircuitNode] = self._check_nodes(ys)
        return [
            self.optimised_mul(x, y)
            for x in xs
            for y in ys
        ]

    @overload
    def partial_derivatives(
            self,
            f: CircuitNode,
            args: Sequence[CircuitNode],
            *,
            self_multiply: bool = False,
    ) -> List[CircuitNode]:
        ...

    @overload
    def partial_derivatives(
            self,
            f: CircuitNode,
            args: CircuitNode,
            *,
            self_multiply: bool = False,
    ) -> CircuitNode:
        ...

    def partial_derivatives(
            self,
            f: CircuitNode,
            args,
            *,
            self_multiply: bool = False,
    ):
        """
        Add to the circuit the operations required to calculate the partial derivative of f
        with respect to each of the given nodes. If self_multiple is True, then this is
        equivalent to calculating the marginal probability at each var that represents
        an indicator.

        This method will cache partial derivative calculations for `f` so that subsequent calls
        to this method with the same `f` will not cause duplicated calculations to be added to
        the circuit. To release this cache, call `self.release_derivatives_cache()`.

        Args:
            f: is the circuit node that defines the function to differentiate.
            args: nodes that are the arguments to f (typically VarNode objects).
                The value may be either a CircuitNode object or a list of CircuitNode objects.
            self_multiply: if true then each partial derivative df/dx will be multiplied by x.

        Returns:
            the results nodes for the partial derivatives, co-indexed with the given arg nodes.
            If  `args` is a single CircuitNode, then a single CircuitNode will be returned, otherwise
            a list of CircuitNode is returned.
        """
        single_result: bool = isinstance(args, CircuitNode)

        args: Sequence[CircuitNode] = self._check_nodes([args])
        if len(args) == 0:
            # Trivial case
            return []

        derivatives: _DerivativeHelper = self._derivatives(f)
        result: List[CircuitNode]
        if self_multiply:
            result = [
                derivatives.derivative_self_mul(arg)
                for arg in args
            ]
        else:
            result = [
                derivatives.derivative(arg)
                for arg in args
            ]

        if single_result:
            return result[0]
        else:
            return result

    def remove_derivatives_cache(self) -> None:
        """
        Release the memory held for partial derivative calculations, as per `partial_derivatives`.
        """
        self.__derivatives = None

    def remove_unreachable_op_nodes(self, *nodes: Args) -> None:
        """
        Find all op nodes reachable from the given nodes, via op arguments.
        (using `self.reachable_op_nodes`). Remove all other op nodes from this circuit.

        If any external object holds a reference to a removed node, that node will be unusable.

        Args:
            *nodes: may be either a node or a list of nodes.
        """
        seen: Set[int] = set()  # set of object ids for all reachable op nodes.
        for node in self._check_nodes(nodes):
            _reachable_op_nodes_seen_r(node, seen)

        if len(seen) < len(self._ops):
            # Invalidate unreadable op nodes
            for op_node in self._ops:
                if id(op_node) not in seen:
                    op_node.circuit = None
                    op_node.args = ()

            # Keep only reachable op nodes
            self._ops = tuple(op_node for op_node in self._ops if id(op_node) in seen)

    def reachable_op_nodes(self, *nodes: Args) -> List[OpNode]:
        """
        Iterate over all op nodes reachable from the given nodes, via op arguments.

        Args:
            *nodes: may be either a node or a list of nodes.

        Returns:
            An list of all op nodes reachable from the given nodes.

        Ensures:
            Returned nodes are not repeated.
            The result is ordered such that if result[i] is referenced by result[j] then i < j.
        """
        seen: Set[int] = set()  # set of object ids for all reachable op nodes.
        return [
            reachable
            for node in self._check_nodes(nodes)
            for reachable in _reachable_op_nodes_r(node, seen)
        ]

    def dump(
            self,
            *,
            prefix: str = '',
            indent: str = '  ',
            var_names: Optional[List[str]] = None,
            include_consts: bool = False,
    ) -> None:
        """
        Print a dump of the Circuit.
        This is intended for debugging and demonstration purposes.

        Args:
            prefix: optional prefix for indenting all lines.
            indent: additional prefix to use for extra indentation.
            var_names: optional variable names to show.
            include_consts: if true, then constant values are dumped.
        """

        next_prefix: str = prefix + indent

        node_name: Dict[int, str] = {}

        print(f'{prefix}number of vars: {self.number_of_vars:,}')
        print(f'{prefix}number of const nodes: {self.number_of_consts:,}')
        print(f'{prefix}number of op nodes: {self.number_of_op_nodes:,}')
        print(f'{prefix}number of operations: {self.number_of_operations:,}')
        print(f'{prefix}number of arcs: {self.number_of_arcs:,}')

        print(f'{prefix}var nodes: {self.number_of_vars}')
        for var in self.vars:
            node_name[id(var)] = f'var[{var.idx}]'
            var_name: str = '' if var_names is None or var.idx >= len(var_names) else var_names[var.idx]
            if var_name != '':
                if var.is_const():
                    print(f'{next_prefix}var[{var.idx}]: {var_name}, {var.const.value}')
                else:
                    print(f'{next_prefix}var[{var.idx}]: {var_name}')
            elif var.is_const():
                print(f'{next_prefix}var[{var.idx}]: {var.const.value}')

        if include_consts:
            print(f'{prefix}const nodes: {self.number_of_consts}')
            for const in self._const_map.values():
                print(f'{next_prefix}{const.value!r}')

        # Add const nodes to the node_name dict
        for const in self._const_map.values():
            node_name[id(const)] = repr(const.value)

        # Add op nodes to the node_name dict
        for i, op in enumerate(self.ops):
            node_name[id(op)] = f'{op.op_str()}<{i}>'

        print(
            f'{prefix}op nodes: {self.number_of_op_nodes} '
            f'(arcs: {self.number_of_arcs}, ops: {self.number_of_operations})'
        )
        for op in reversed(self.ops):
            op_name = node_name[id(op)]
            args_str = ' '.join(node_name[id(arg)] for arg in op.args)
            print(f'{next_prefix}{op_name}: {args_str}')

    def _check_nodes(self, nodes: Iterable[Args]) -> Tuple[CircuitNode, ...]:
        """
        Convert the given circuit nodes to a tuple, flattening nested iterables as needed.

        Args:
            nodes: some circuit nodes of constant values.

        Raises:
            RuntimeError: if any node does not belong to this circuit.
        """
        return tuple(self.__check_nodes(nodes))

    def __check_nodes(self, nodes: Iterable[Args]) -> Iterable[CircuitNode]:
        """
        Convert the given circuit nodes to a tuple, flattening nested iterables as needed.

        Args:
            nodes: some circuit nodes of constant values.

        Raises:
            RuntimeError: if any node does not belong to this circuit.
        """

        for node in nodes:
            if isinstance(node, CircuitNode):
                if node.circuit is not self:
                    raise RuntimeError('node does not belong to this circuit')
                else:
                    yield node
            elif isinstance(node, ConstValue):
                yield self.const(node)
            else:
                # Assume it's iterable
                for n in self._check_nodes(node):
                    yield n

    def _derivatives(self, f: CircuitNode) -> _DerivativeHelper:
        """
        Get a _DerivativeHelper for `f`.
        Checking the derivative cache.
        """
        derivatives: Optional[_DerivativeHelper] = self.__derivatives
        if derivatives is None or derivatives.f is not f:
            derivatives = _DerivativeHelper(f)
            self.__derivatives = derivatives
        return derivatives


class CircuitNode:
    """
    A node in an arithmetic circuit.
    Each node is either an op, var, or const node.

    Each op node is either a mul, add or sub node. Each op
    node has zero or more arguments. Each argument is another node.

    Every var node has an index, `idx`, which is an integer counting from zero, and denotes
    its creation order.

    A var node may be temporarily set to be a constant node, which may
    be useful for optimising a compiled circuit.
    """
    __slots__ = ('circuit',)

    def __init__(self, circuit: Circuit):
        self.circuit = circuit

    @property
    def is_zero(self) -> bool:
        """
        Does this node represent the constant zero.
        """
        return False

    @property
    def is_one(self) -> bool:
        """
        Does this node represent the constant one.
        """
        return False

    def __add__(self, other: CircuitNode | ConstValue):
        return self.circuit.add(self, other)

    def __mul__(self, other: CircuitNode | ConstValue):
        return self.circuit.mul(self, other)


# A type representing a flexible representation of multiple CircuitNode objects.
Args = CircuitNode | ConstValue | Iterable[CircuitNode | ConstValue]


class ConstNode(CircuitNode):
    """
    A node in a circuit representing a constant value.
    """
    __slots__ = ('_value',)

    def __init__(self, circuit: Circuit, value: ConstValue):
        super().__init__(circuit)
        self._value: ConstValue = value

    @property
    def value(self) -> ConstValue:
        return self._value

    @property
    def is_zero(self) -> bool:
        # noinspection PyProtectedMember
        return self is self.circuit.zero

    @property
    def is_one(self) -> bool:
        # noinspection PyProtectedMember
        return self is self.circuit.one

    def __str__(self) -> str:
        return f'const({self.value})'

    def __lt__(self, other) -> bool:
        if isinstance(other, ConstNode):
            return self._value < other._value
        else:
            return False


class VarNode(CircuitNode):
    """
    A node in a circuit representing an input variable.
    """
    __slots__ = ('_idx', '_const')

    def __init__(self, circuit: Circuit, idx: int):
        super().__init__(circuit)
        self._idx: int = idx
        self._const: Optional[ConstNode] = None

    @property
    def idx(self) -> int:
        return self._idx

    def is_const(self) -> bool:
        return self._const is not None

    @property
    def const(self) -> Optional[ConstNode]:
        return self._const

    @const.setter
    def const(self, value: ConstValue | ConstNode | None) -> None:
        if value is None:
            self._const = None
        else:
            self._const = self.circuit.const(value)

    @property
    def is_zero(self) -> bool:
        return self._const is not None and self._const.is_zero

    @property
    def is_one(self) -> bool:
        return self._const is not None and self._const.is_one

    def __lt__(self, other) -> bool:
        if isinstance(other, VarNode):
            return self._idx < other.idx
        else:
            return False

    def __str__(self) -> str:
        if self._const is None:
            return f'var[{self.idx}]'
        else:
            return f'var[{self.idx}] = {self._const.value}'


class OpNode(CircuitNode):
    """
    A node in a circuit representing an arithmetic operation.
    """
    __slots__ = ('args', 'symbol')

    def __init__(self, circuit: Circuit, symbol: int, args: Tuple[CircuitNode, ...]):
        super().__init__(circuit)
        self.args: Tuple[CircuitNode, ...] = args
        self.symbol: int = symbol

    def __str__(self) -> str:
        return f'{self.op_str()}\\{len(self.args)}'

    def op_str(self) -> str:
        """
        Returns the op node operation as a string.
        """
        if self.symbol == MUL:
            return 'mul'
        elif self.symbol == ADD:
            return 'add'
        else:
            return '?' + str(self.symbol)


@dataclass
class _DNode:
    """
    A data structure supporting derivative calculations.
    A DNode holds all information needed to calculate the partial derivative at `node`.
    """
    node: CircuitNode
    derivative: Optional[CircuitNode]
    derivative_self_mul: Optional[CircuitNode] = None
    sum_prod: List[_DNodeProduct] = field(default_factory=list)
    processed: bool = False

    def __str__(self) -> str:
        """
        for debugging
        """
        dots: str = '...'
        return (
            f'DNode({self.node}, '
            f'{None if self.derivative is None else dots}, '
            f'{None if self.derivative_self_mul is None else dots}, '
            f'{len(self.sum_prod)}, '
            f'{self.processed})'
        )


@dataclass
class _DNodeProduct:
    """
    A data structure supporting derivative calculations.

    The represents a product of `parent` and `prod`.
    """
    parent: _DNode
    prod: List[CircuitNode]

    def __str__(self) -> str:
        """
        for debugging
        """
        return f'DNodeProduct({self.parent}, {self.prod})'


class _DerivativeHelper:
    """
    A data structure to support efficient calculation of partial derivatives
    with respect to some function node `f`.
    """

    def __init__(self, f: CircuitNode):
        """
        Prepare to calculate partial derivatives with respect to `f`.
        """
        self.f: CircuitNode = f
        self.circuit: Circuit = f.circuit
        self.d_nodes: Dict[int, _DNode] = {}  # map id(CircuitNode) to its DNode
        self.zero = self.circuit.zero
        self.one = self.circuit.one
        top_d_node: _DNode = _DNode(f, self.one)
        self.d_nodes[id(f)] = top_d_node
        self._mk_derivative_r(top_d_node)

    def derivative(self, node: CircuitNode) -> CircuitNode:
        d_node: Optional[_DNode] = self.d_nodes.get(id(node))
        if d_node is None:
            return self.zero
        else:
            return self._derivative(d_node)

    def derivative_self_mul(self, node: CircuitNode) -> CircuitNode:
        d_node: Optional[_DNode] = self.d_nodes.get(id(node))
        if d_node is None:
            return self.zero

        if d_node.derivative_self_mul is None:
            d: CircuitNode = self._derivative(d_node)
            if d is self.zero:
                d_node.derivative_self_mul = self.zero
            elif d is self.one:
                d_node.derivative_self_mul = node
            else:
                d_node.derivative_self_mul = self.circuit.optimised_mul(d, node)

        return d_node.derivative_self_mul

    def _derivative(self, d_node: _DNode) -> CircuitNode:
        if d_node.derivative is not None:
            return d_node.derivative

        # Get the list of circuit nodes that must be added together.
        to_add: Sequence[CircuitNode] = tuple(
            value
            for value in (self._derivative_prod(prods) for prods in d_node.sum_prod)
            if not value.is_zero
        )
        # We can release the temporary memory at this DNode now
        # Warning disabled as we will never use this field again - doing so would be an error.
        # noinspection PyTypeChecker
        d_node.sum_prod = None

        # Construct the addition operation
        d_node.derivative = self.circuit.optimised_add(*to_add)

        return d_node.derivative

    def _derivative_prod(self, prods: _DNodeProduct) -> CircuitNode:
        """
        Support `_derivative` by constructing the derivative product for the given _DNodeProduct.
        """
        # Get the derivative of the parent node.
        parent: CircuitNode = self._derivative(prods.parent)

        # Multiply the parent derivative with all other nodes recorded at prod.
        to_mul: List[CircuitNode] = []
        for arg in chain((parent,), prods.prod):
            if arg is self.zero:
                # Multiplication by zero is zero
                return self.zero
            if arg is not self.one:
                to_mul.append(arg)

        # Construct the multiplication operation
        return self.circuit.optimised_mul(*to_mul)

    def _mk_derivative_r(self, d_node: _DNode) -> None:
        """
        Construct a DNode for each argument of the given DNode.
        """
        if d_node.processed:
            return
        d_node.processed = True
        node: CircuitNode = d_node.node

        if isinstance(node, OpNode):
            if node.symbol == ADD:
                for arg in node.args:
                    child_d_node = self._add(arg, d_node, [])
                    self._mk_derivative_r(child_d_node)
            elif node.symbol == MUL:
                for arg in node.args:
                    prod = [arg2 for arg2 in node.args if arg is not arg2]
                    child_d_node = self._add(arg, d_node, prod)
                    self._mk_derivative_r(child_d_node)
            else:
                raise RuntimeError(f'unknown op node symbol: {node.symbol!r}')

    def _add(self, node: CircuitNode, parent: _DNode, prod: List[CircuitNode]) -> _DNode:
        """
        Support for `_mk_derivative_r`.

        Add a _DNodeProduct(parent, negate, prod) to the DNode for the given circuit node.

        If the DNode for `node` does not yet exist, one will be created.

        The given circuit node may have multiple parents (i.e., a shared sub-expression). Therefore,
        this method may be called multiple times for a given node. Each time a new _DNodeProduct will be added.

        Args:
            node: the CircuitNode that the returned DNode is for.
            parent: the DNode of the parent node, i.e., `node` is an argument to the parent node.
            prod: other circuit nodes that need to be multiplied with the parent derivative when
                constructing a derivative for `node`.

        Returns:
            the DNode for `node`.
        """
        child_d_node: _DNode = self._get(node)
        child_d_node.sum_prod.append(_DNodeProduct(parent, prod))
        return child_d_node

    def _get(self, node: CircuitNode) -> _DNode:
        """
        Get the DNode for the given circuit node.
        If no DNode exist for it yet, then one will be constructed.
        """
        node_id: int = id(node)
        d_node: Optional[_DNode] = self.d_nodes.get(node_id)
        if d_node is None:
            d_node = _DNode(node, None)
            self.d_nodes[node_id] = d_node
        return d_node


def _reachable_op_nodes_r(node: CircuitNode, seen: Set[int]) -> Iterator[OpNode]:
    """
    Recursive helper for `reachable_op_nodes`. Performs a depth-first search.

    Args:
        node: the current node being checked.
        seen: keep track of seen op node ids (to avoid returning multiple of the same node).

    Returns:
        An iterator over all op nodes reachable from the given node.
    """
    if isinstance(node, OpNode) and id(node) not in seen:
        seen.add(id(node))
        for arg in node.args:
            for reachable in _reachable_op_nodes_r(arg, seen):
                yield reachable
        yield node


def _reachable_op_nodes_seen_r(node: CircuitNode, seen: Set[int]) -> None:
    """
    Recursive helper for `remove_unreachable_op_nodes`. Performs a depth-first search.

    Args:
        node: the current node being checked.
        seen: set of seen op node ids.
    """
    if isinstance(node, OpNode) and id(node) not in seen:
        seen.add(id(node))
        for arg in node.args:
            _reachable_op_nodes_seen_r(arg, seen)
