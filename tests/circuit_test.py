import ctypes as ct
import pickle
from typing import Sequence, Tuple, List, overload

from ck.circuit import Circuit, CircuitNode, ConstNode, VarNode, OpNode, TmpConst, MUL, ADD, ConstValue
from ck.circuit_compiler.interpret_compiler import compile_circuit
from ck.utils.np_extras import NDArrayNumeric, DType
from tests.helpers.unittest_fixture import Fixture, test_main


class CircuitInterpreter:
    def __init__(self, circuit: Circuit, dtype: DType = ct.c_double):
        self.circuit: Circuit = circuit
        self.vars: List[ConstValue] = [0] * len(circuit.vars)
        self.dtype: DType = dtype

    def __setitem__(self, key, value):
        if isinstance(key, VarNode):
            key = key.idx
        self.vars[key] = value

    def __getitem__(self, key):
        if isinstance(key, VarNode):
            key = key.idx
        return self.vars[key]

    @overload
    def __call__(self, result: CircuitNode) -> int | float:
        ...

    @overload
    def __call__(self, result: Sequence[CircuitNode]) -> NDArrayNumeric :
        ...

    def __call__(self, result):
        if isinstance(result, CircuitNode):
            program = compile_circuit(result, input_vars=self.circuit.vars, dtype=self.dtype)
            return program(self.vars).item()
        else:
            program = compile_circuit(*result, input_vars=self.circuit.vars, dtype=self.dtype)
            return program(self.vars)


class CircuitFixture(Fixture):

    def assertCircuitsEqual(
            self,
            lhs: Circuit,
            rhs: Circuit
    ) -> None:
        self.assertEqual(lhs.number_of_arcs, rhs.number_of_arcs)
        self.assertEqual(lhs.number_of_consts, rhs.number_of_consts)
        self.assertEqual(lhs.number_of_op_nodes, rhs.number_of_op_nodes)
        self.assertEqual(lhs.number_of_vars, rhs.number_of_vars)

        seen_1 = {}
        seen_2 = {}

        # It does not really make sense to check that constants are co-indexed.
        # for i in range(lhs.number_of_consts):
        #     self._assertNodesEqual(lhs.get_const_at(i), rhs.get_const_at(i), seen_1, seen_2)

        # This assertion is not trivially true even though lhs.number_of_vars == rhs.number_of_vars.
        # This is because a variable may be set to a constant.
        for i in range(lhs.number_of_vars):
            self._assertNodesEqual(lhs.vars[i], rhs.vars[i], seen_1, seen_2)

        # This assertion requires that operation nodes are added to each circuit in the
        # same order.
        for i in range(rhs.number_of_op_nodes):
            self._assertNodesEqual(lhs.ops[i], rhs.ops[i], seen_1, seen_2)

    def assertNodesEqual(self, c1: CircuitNode, c2: CircuitNode) -> None:
        self._assertNodesEqual(c1, c2, {}, {})

    def _assertNodesEqual(
            self,
            n1: CircuitNode,
            n2: CircuitNode,
            seen_1: dict,
            seen_2: dict
    ) -> None:
        self.assertIs(n1.__class__, n2.__class__)

        # Need to treat constants differently
        if isinstance(n1, ConstNode):
            assert isinstance(n2, ConstNode)
            self.assertEqual(n1.value, n2.value)
            return

        saw_2 = seen_1.get(id(n1))
        saw_1 = seen_2.get(id(n2))
        if saw_2 == n2 and saw_1 == n1:
            return
        if not (saw_2 is None and saw_1 is None):
            raise self.failureException(f'Circuit nodes differ: {n1}, {n1}')
        seen_1[id(n1)] = n2
        seen_2[id(n2)] = n1

        if isinstance(n1, VarNode):
            assert isinstance(n2, VarNode)
            self.assertEqual(n1.idx, n2.idx)
            self.assertEqual(n1.is_const(), n2.is_const())
            if n1.is_const():
                self.assertIsNotNone(n1.const)
                self.assertIsNotNone(n2.const)
                self._assertNodesEqual(n1.const, n2.const, seen_1, seen_2)

        elif isinstance(n1, OpNode):
            assert isinstance(n2, OpNode)
            for child1, child2 in zip(n1.args, n2.args):
                self._assertNodesEqual(child1, child2, seen_1, seen_2)
        else:
            raise NotImplementedError(f'unknown type of node: {n1!r}')

    def assertEmptyCct(self, cct):
        num_vars = cct.number_of_vars
        self.assertEqual(0, num_vars)

        num_of_consts = cct.number_of_consts
        self.assertEqual(2, num_of_consts)  # a circuit is created with constants zero and one

        num_of_op_nodes = cct.number_of_op_nodes
        self.assertEqual(0, num_of_op_nodes)

        num_of_arcs = cct.number_of_arcs
        self.assertEqual(0, num_of_arcs)


class TestCircuitBasic(CircuitFixture):

    def test_circuit_empty(self):
        cct = Circuit()
        self.assertEmptyCct(cct)

    def test_new_var(self):
        cct = Circuit()
        var = cct.new_var()
        self.assertTrue(isinstance(var, CircuitNode))

        num_vars = cct.number_of_vars
        self.assertEqual(1, num_vars)

        num_of_op_nodes = cct.number_of_op_nodes
        self.assertEqual(0, num_of_op_nodes)

        num_of_arcs = cct.number_of_arcs
        self.assertEqual(0, num_of_arcs)

        # add another one
        var = cct.new_var()
        self.assertTrue(isinstance(var, VarNode))

        num_vars = cct.number_of_vars
        self.assertEqual(2, num_vars)

        num_of_op_nodes = cct.number_of_op_nodes
        self.assertEqual(0, num_of_op_nodes)

    def test_new_vars(self):
        cct = Circuit()
        vars_a = cct.new_vars(2)
        self.assertTrue(isinstance(vars_a, Tuple))
        for var in vars_a:
            self.assertTrue(isinstance(var, VarNode))

        num_vars = cct.number_of_vars
        self.assertEqual(2, num_vars)

        num_of_op_nodes = cct.number_of_op_nodes
        self.assertEqual(0, num_of_op_nodes)

        num_of_arcs = cct.number_of_arcs
        self.assertEqual(0, num_of_arcs)

        self.assertEqual(2, len(vars_a))

        # add more

        vars_b = cct.new_vars(3)

        num_vars = cct.number_of_vars
        self.assertEqual(5, num_vars)

        num_of_op_nodes = cct.number_of_op_nodes
        self.assertEqual(0, num_of_op_nodes)

        self.assertEqual(3, len(vars_b))

    def test_get_const_zero(self):
        cct = Circuit()

        self.assertEqual(cct.number_of_consts, 2)  # a circuit is created with constants zero and one

        const_0 = cct.const(0)
        self.assertTrue(isinstance(const_0, ConstNode))

        self.assertEqual(cct.number_of_consts, 2)  # a circuit is created with constants zero and one

        const_again = cct.const(0.0)

        self.assertEqual(cct.number_of_consts, 2)  # a circuit is created with constants zero and one
        self.assertIs(const_0, const_again)

    def test_get_const_one(self):
        cct = Circuit()

        self.assertEqual(cct.number_of_consts, 2)  # a circuit is created with constants zero and one

        const_1 = cct.const(1)
        self.assertTrue(isinstance(const_1, ConstNode))

        self.assertEqual(cct.number_of_consts, 2)

        const_again = cct.const(1.0)

        self.assertEqual(cct.number_of_consts, 2)
        self.assertIs(const_1, const_again)

    def test_get_const(self):
        cct = Circuit()

        # a circuit is created with constants zero and one
        self.assertEqual(2, cct.number_of_consts)

        const_0 = cct.const(0)
        self.assertEqual(const_0, cct.const(0))
        self.assertEqual(2, cct.number_of_consts)

        const_1 = cct.const(1)
        self.assertEqual(const_1, cct.const(1))
        self.assertEqual(2, cct.number_of_consts)

        const_2 = cct.const(2)
        self.assertNotEqual(const_0, const_2)
        self.assertNotEqual(const_1, const_2)
        self.assertEqual(const_2.value, 2.0)
        self.assertEqual(3, cct.number_of_consts)

        const_2_again = cct.const(2)
        self.assertEqual(const_2, const_2_again)
        self.assertEqual(const_2_again.value, 2.0)
        self.assertEqual(3, cct.number_of_consts)

    def test_mul(self):
        cct = Circuit()

        vars_a = cct.new_vars(2)
        mul_a = cct.mul(vars_a)
        self.assertTrue(isinstance(mul_a, CircuitNode))

        num_of_op_nodes = cct.number_of_op_nodes
        self.assertEqual(1, num_of_op_nodes)

        num_of_arcs = cct.number_of_arcs
        self.assertEqual(2, num_of_arcs)

        vars_b = [vars_a[0], vars_a[1]]
        mul_b = cct.mul(vars_b)

        self.assertIsInstance(mul_a, OpNode)
        self.assertEqual(mul_a.symbol, MUL)
        self.assertIsInstance(mul_b, OpNode)
        self.assertEqual(mul_b.symbol, MUL)

        num_of_op_nodes = cct.number_of_op_nodes
        self.assertEqual(2, num_of_op_nodes)

        num_of_arcs = cct.number_of_arcs
        self.assertEqual(4, num_of_arcs)

    def test_add(self):
        cct = Circuit()

        vars_a = cct.new_vars(2)
        add_a = cct.add(vars_a)
        self.assertTrue(isinstance(add_a, CircuitNode))

        num_of_op_nodes = cct.number_of_op_nodes
        self.assertEqual(1, num_of_op_nodes)

        num_of_arcs = cct.number_of_arcs
        self.assertEqual(2, num_of_arcs)

        vars_b = [vars_a[0], vars_a[1]]
        add_b = cct.add(vars_b)

        self.assertIsInstance(add_a, OpNode)
        self.assertEqual(add_a.symbol, ADD)
        self.assertIsInstance(add_b, OpNode)
        self.assertEqual(add_b.symbol, ADD)

        num_of_op_nodes = cct.number_of_op_nodes
        self.assertEqual(2, num_of_op_nodes)

        num_of_arcs = cct.number_of_arcs
        self.assertEqual(4, num_of_arcs)

    def test_clear_const_vars(self):
        cct = Circuit()

        v = cct.new_vars(2)

        self.assertIsNone(v[0].const)
        self.assertIsNone(v[1].const)

        v[0].const = 0
        v[1].const = 1

        self.assertEqual(v[0].const, cct.const(0))
        self.assertEqual(v[1].const, cct.const(1))

        v[0].const = None
        v[1].const = None

        self.assertIsNone(v[0].const)
        self.assertIsNone(v[1].const)

    def test_op_number_of_args(self):
        cct = Circuit()

        v = cct.new_vars(3)
        add_op = cct.add(v)

        self.assertEqual(3, len(add_op.args))

    def test_get_arg(self):
        cct = Circuit()

        v = cct.new_vars(2)
        add_op = cct.add(v)

        self.assertEqual(2, len(add_op.args))
        self.assertEqual(v[0], add_op.args[0])
        self.assertEqual(v[1], add_op.args[1])

    def test_op_args(self):
        cct = Circuit()

        v = cct.new_vars(2)
        add_op = cct.add(v)

        args = add_op.args

        self.assertEqual(2, len(args))
        self.assertEqual(v[0], args[0])
        self.assertEqual(v[1], args[1])


class TestCircuitSimpleCalculations(CircuitFixture):

    def test_add(self):
        cct = Circuit()

        v0, v1 = cct.new_vars(2)
        top = v0 + v1

        self.assertEqual(1, cct.number_of_op_nodes)
        self.assertEqual(2, cct.number_of_arcs)
        self.assertEqual(2, cct.number_of_vars)
        self.assertEqual(2, cct.number_of_consts)  # a circuit is created with constants zero and one

        calculator = CircuitInterpreter(cct)

        calculator[0] = 3
        calculator[1] = 4

        result = calculator(top)
        self.assertEqual(result, 7)

    def test_mul(self):
        cct = Circuit()

        v0, v1 = cct.new_vars(2)
        top = v0 * v1

        self.assertEqual(1, cct.number_of_op_nodes)
        self.assertEqual(2, cct.number_of_arcs)
        self.assertEqual(2, cct.number_of_vars)
        self.assertEqual(2, cct.number_of_consts)  # a circuit is created with constants zero and one

        calculator = CircuitInterpreter(cct)

        calculator[0] = 3
        calculator[1] = 4

        result = calculator(top)
        self.assertEqual(result, 12)

    def test_tree(self):
        cct = Circuit()

        v0, v1, v2, v3 = cct.new_vars(4)
        top = (v0 + v1) * (v2 + v3)

        self.assertEqual(3, cct.number_of_op_nodes)
        self.assertEqual(6, cct.number_of_arcs)
        self.assertEqual(4, cct.number_of_vars)
        self.assertEqual(2, cct.number_of_consts)  # a circuit is created with constants zero and one

        calculator = CircuitInterpreter(cct)

        calculator[0] = 3
        calculator[1] = 4
        calculator[2] = 5
        calculator[3] = 6

        result = calculator(top)
        self.assertEqual(result, 77)

    def test_dup_ref(self):
        cct = Circuit()

        c = cct.const(7)
        top = cct.mul(c, c)

        self.assertEqual(1, cct.number_of_op_nodes)
        self.assertEqual(2, cct.number_of_arcs)
        self.assertEqual(0, cct.number_of_vars)
        self.assertEqual(3, cct.number_of_consts)  # a circuit is created with constants zero and one

        calculator = CircuitInterpreter(cct)

        result = calculator(top)
        self.assertEqual(result, 49)

    def test_graph(self):
        cct = Circuit()

        v0, v1 = cct.new_vars(2)

        a = v0 + v1
        top = a * a

        self.assertEqual(2, cct.number_of_op_nodes)
        self.assertEqual(4, cct.number_of_arcs)
        self.assertEqual(2, cct.number_of_vars)
        self.assertEqual(2, cct.number_of_consts)  # a circuit is created with constants zero and one

        calculator = CircuitInterpreter(cct)

        calculator[0] = 3
        calculator[1] = 4

        result = calculator(top)
        self.assertEqual(result, 49)

    def test_mult_args(self):
        cct = Circuit()
        top = cct.mul(cct.const(2), cct.const(3), cct.const(5), cct.const(7))

        self.assertEqual(1, cct.number_of_op_nodes)
        self.assertEqual(4, cct.number_of_arcs)
        self.assertEqual(0, cct.number_of_vars)
        self.assertEqual(6, cct.number_of_consts)  # a circuit is created with constants zero and one

        calculator = CircuitInterpreter(cct)

        result = calculator(top)
        self.assertEqual(result, 210)

    def test_constants(self):
        cct = Circuit()
        top = cct.add(cct.const(123), cct.const(456))

        self.assertEqual(1, cct.number_of_op_nodes)
        self.assertEqual(2, cct.number_of_arcs)
        self.assertEqual(0, cct.number_of_vars)
        self.assertEqual(4, cct.number_of_consts)  # a circuit is created with constants zero and one

        calculator = CircuitInterpreter(cct)

        result = calculator(top)
        self.assertEqual(result, 579)

    def test_constants_and_vars(self):
        cct = Circuit()

        a = cct.add(cct.const(123), cct.const(456))
        top = cct.mul(a, cct.new_var())

        self.assertEqual(2, cct.number_of_op_nodes)
        self.assertEqual(4, cct.number_of_arcs)
        self.assertEqual(1, cct.number_of_vars)
        self.assertEqual(4, cct.number_of_consts)  # a circuit is created with constants zero and one

        calculator = CircuitInterpreter(cct)

        calculator[0] = 2

        result = calculator(top)
        self.assertEqual(result, 1158)

    def test_fractional_values_add(self):
        cct = Circuit()

        vs = cct.new_vars(2)
        top = cct.add(vs)

        self.assertEqual(1, cct.number_of_op_nodes)
        self.assertEqual(2, cct.number_of_arcs)
        self.assertEqual(2, cct.number_of_vars)
        self.assertEqual(2, cct.number_of_consts)  # a circuit is created with constants zero and one

        calculator = CircuitInterpreter(cct)

        calculator[0] = 3.1
        calculator[1] = 4.2

        result = calculator(top)
        self.assertAlmostEqual(result, 7.3, 10)

    def test_fractional_values_mul(self):
        cct = Circuit()

        vs = cct.new_vars(2)
        top = cct.mul(vs)

        self.assertEqual(1, cct.number_of_op_nodes)
        self.assertEqual(2, cct.number_of_arcs)
        self.assertEqual(2, cct.number_of_vars)
        self.assertEqual(2, cct.number_of_consts)  # a circuit is created with constants zero and one

        calculator = CircuitInterpreter(cct)

        calculator[0] = 3.1
        calculator[1] = 4.2

        result = calculator(top)
        self.assertAlmostEqual(result, 13.02, 10)

    def test_small_fractional_values(self):
        cct = Circuit()

        v = cct.new_vars(2)
        top = cct.add(v[0], v[1])

        self.assertEqual(1, cct.number_of_op_nodes)
        self.assertEqual(2, cct.number_of_arcs)
        self.assertEqual(2, cct.number_of_vars)
        self.assertEqual(2, cct.number_of_consts)  # a circuit is created with constants zero and one

        calculator = CircuitInterpreter(cct)

        calculator[0] = 0.031
        calculator[1] = 0.042

        result = calculator(top)
        self.assertAlmostEqual(result, 0.073, 10)

    def test_single_arg_op(self):
        cct = Circuit()

        v = cct.new_vars(1)
        top = cct.add(v[0])

        self.assertEqual(1, cct.number_of_op_nodes)
        self.assertEqual(1, cct.number_of_arcs)
        self.assertEqual(1, cct.number_of_vars)
        self.assertEqual(2, cct.number_of_consts)  # a circuit is created with constants zero and one

        calculator = CircuitInterpreter(cct)

        calculator[0] = 1234

        result = calculator(top)
        self.assertEqual(result, 1234)

    def test_no_op_var(self):
        cct = Circuit()

        top = cct.new_var()

        self.assertEqual(0, cct.number_of_op_nodes)
        self.assertEqual(0, cct.number_of_arcs)
        self.assertEqual(1, cct.number_of_vars)
        self.assertEqual(2, cct.number_of_consts)  # a circuit is created with constants zero and one

        calculator = CircuitInterpreter(cct)

        calculator[0] = 1234

        result = calculator(top)
        self.assertEqual(result, 1234)

    def test_no_op_const(self):
        cct = Circuit()

        top = cct.const(1234)

        self.assertEqual(0, cct.number_of_op_nodes)
        self.assertEqual(0, cct.number_of_arcs)
        self.assertEqual(0, cct.number_of_vars)
        self.assertEqual(3, cct.number_of_consts)  # a circuit is created with constants zero and one

        calculator = CircuitInterpreter(cct)

        result = calculator(top)
        self.assertEqual(result, 1234)

    def test_round_trip_values_int(self):
        cct = Circuit()

        values = (
            2 ** 63 - 1,
            - (2 ** 63 - 1),

            2 ** 31 + 1,
            2 ** 31,
            2 ** 31 - 1,

            123000,
            123,
            1,

            0,

            -123000,
            -123,
            -1,
        )

        nodes = []
        for v in values:
            const_node = cct.const(v)
            node = cct.add(const_node)
            nodes.append(node)

        calculator = CircuitInterpreter(cct, dtype=ct.c_int64)

        for node, value in zip(nodes, values):
            result = calculator(node)
            self.assertEqual(result, value)

    def test_round_trip_values_float(self):
        cct = Circuit()

        values = (
            123000,
            123.456,
            123,
            1,
            0.123,
            0.00123,

            0,

            -123000,
            -123.456,
            -123,
            -1,

            -0.123,
            -0.00123,
        )

        nodes = []
        for v in values:
            const_node = cct.const(v)
            node = cct.add(const_node)
            nodes.append(node)

        calculator = CircuitInterpreter(cct, dtype=ct.c_double)

        for node, value in zip(nodes, values):
            result = calculator(node)
            self.assertEqual(result, value)

    def test_shared_node(self):
        cct = Circuit()

        v = cct.new_vars(2)
        a = cct.add(v)
        b = cct.mul(a, a)

        self.assertEqual(2, cct.number_of_op_nodes)
        self.assertEqual(4, cct.number_of_arcs)
        self.assertEqual(2, cct.number_of_vars)
        self.assertEqual(2, cct.number_of_consts)  # a circuit is created with constants zero and one

        calculator = CircuitInterpreter(cct)

        calculator[0] = 1
        calculator[1] = 2
        result = calculator(b)

        self.assertEqual(result, 9)

    def test_cartesian_product(self):
        cct = Circuit()

        x = cct.new_vars(2)
        y = cct.new_vars(2)
        prods = cct.cartesian_product(x, y)

        calculator = CircuitInterpreter(cct)
        calculator[:] = (2, 3, 5, 7)

        expected_result = (2 * 5, 2 * 7, 3 * 5, 3 * 7)
        self.assertEqual(len(prods), len(expected_result))

        result = calculator(prods)

        self.assertArrayEqual(result, expected_result)


class TestCircuitDerivative(CircuitFixture):

    def test_derivative_const(self):
        #        f
        #    |   |
        #    x  123
        #
        cct = Circuit()
        x = cct.new_var()
        f = cct.const(123)
        der = cct.partial_derivatives(f, x)

        calculator = CircuitInterpreter(cct)

        calculator[0] = 456
        result = calculator(der)

        self.assertEqual(result, 0)

    def test_derivative_const_self_multiply(self):
        #        f
        #    |   |
        #    x  123
        #
        cct = Circuit()
        x = cct.new_var()
        f = cct.const(123)
        der = cct.partial_derivatives(f, x, self_multiply=True)

        calculator = CircuitInterpreter(cct)

        calculator[0] = 456
        result = calculator(der)

        self.assertEqual(result, 0)

    def test_derivative_add(self):
        #        f
        #        |
        #        +
        #       / \
        #     x0  x1
        #
        cct = Circuit()
        xs = cct.new_vars(2)
        f = cct.add(xs)
        der = cct.partial_derivatives(f, xs)

        self.assertEqual(len(der), 2)

        calculator = CircuitInterpreter(cct)

        calculator[0] = 123
        calculator[1] = 456

        result = calculator(der[0])
        self.assertEqual(result, 1)

        result = calculator(der[1])
        self.assertEqual(result, 1)

    def test_derivative_add_self_multiply(self):
        #        f
        #        |
        #        +
        #       / \
        #     x0  x1
        #
        cct = Circuit()
        xs = cct.new_vars(2)
        f = cct.add(xs)
        der = cct.partial_derivatives(f, xs, self_multiply=True)

        self.assertEqual(len(der), 2)

        calculator = CircuitInterpreter(cct)

        calculator[0] = 123
        calculator[1] = 456

        result = calculator(der[0])
        self.assertEqual(result, 123)

        result = calculator(der[1])
        self.assertEqual(result, 456)

    def test_derivative_mul(self):
        #        f
        #        |
        #        *
        #       / \
        #     x0  x1
        #
        cct = Circuit()
        xs = cct.new_vars(2)
        f = cct.mul(xs)
        der = cct.partial_derivatives(f, xs)

        self.assertEqual(len(der), 2)

        calculator = CircuitInterpreter(cct)

        calculator[0] = 123
        calculator[1] = 456

        result = calculator(der[0])
        self.assertEqual(result, 456)

        result = calculator(der[1])
        self.assertEqual(result, 123)

    def test_derivative_mul_self_multiply(self):
        #        f
        #        |
        #        *
        #       / \
        #     x0  x1
        #
        cct = Circuit()
        xs = cct.new_vars(2)
        f = cct.mul(xs)
        der = cct.partial_derivatives(f, xs, self_multiply=True)

        self.assertEqual(len(der), 2)

        calculator = CircuitInterpreter(cct)

        calculator[0] = 123
        calculator[1] = 456
        result = calculator(der)

        self.assertEqual(result[0], 56088)
        self.assertEqual(result[1], 56088)

    def test_derivative_complex(self):
        #         f
        #         *
        #        / \
        #       /   \
        #     n2    n3
        #     +     +
        #    /  \  /  \
        #   x0   n1   x3
        #         *
        #        / \
        #       x1  x2
        #
        cct = Circuit()
        xs = cct.new_vars(4)
        n1 = cct.mul(xs[1], xs[2])
        n2 = cct.add(xs[0], n1)
        n3 = cct.add(n1, xs[3])
        f = cct.mul(n2, n3)
        der = cct.partial_derivatives(f, xs)
        der.append(f)

        self.assertEqual(len(der), 5)

        calculator = CircuitInterpreter(cct)

        calculator[0] = 12
        calculator[1] = 34
        calculator[2] = 56
        calculator[3] = 78
        result = calculator(der)

        self.assertEqual(result[0], 1982)  # df / dx[0]
        self.assertEqual(result[1], 218288)  # df / dx[1]
        self.assertEqual(result[2], 132532)  # df / dx[2]
        self.assertEqual(result[3], 1916)  # df / dx[3]
        self.assertEqual(result[4], 3797512)  # f

    def test_derivative_complex_self_multiply(self):
        #         f
        #         *
        #        / \
        #       /   \
        #     n2    n3
        #     +     +
        #    /  \  /  \
        #   x0   n1   x3
        #         *
        #        / \
        #       x1  x2
        #
        cct = Circuit()
        xs = cct.new_vars(4)
        n1 = cct.mul(xs[1], xs[2])
        n2 = cct.add(xs[0], n1)
        n3 = cct.add(n1, xs[3])
        f = cct.mul(n2, n3)
        der = cct.partial_derivatives(f, xs, self_multiply=True)
        der.append(f)

        self.assertEqual(len(der), 5)

        calculator = CircuitInterpreter(cct)

        calculator[0] = 12
        calculator[1] = 34
        calculator[2] = 56
        calculator[3] = 78
        result = calculator(der)

        self.assertEqual(result[0], 23784)  # df / dx[0]
        self.assertEqual(result[1], 7421792)  # df / dx[1]
        self.assertEqual(result[2], 7421792)  # df / dx[2]
        self.assertEqual(result[3], 149448)  # df / dx[3]
        self.assertEqual(result[4], 3797512)  # f

    def test_still_get_f_mul(self):
        # This is exercising a found bug.
        cct = Circuit()

        x = cct.new_var()
        y = cct.new_var()
        net = cct.add(
            cct.mul(x, y),
            cct.mul(x, y),
        )

        cct.partial_derivatives(net, y)

        calculator = CircuitInterpreter(cct)
        calculator[x] = 1
        calculator[y] = 0.5

        result = calculator(net)
        self.assertEqual(result, 1)

    def test_still_get_f_add(self):
        # This is exercising a found bug.
        cct = Circuit()

        x = cct.new_var()
        y = cct.new_var()
        net = cct.mul(
            cct.add(x, y),
            cct.add(x, y),
        )

        cct.partial_derivatives(net, y)

        calculator = CircuitInterpreter(cct)
        calculator[x] = 0.5
        calculator[y] = 0.5
        result = calculator(net)
        self.assertEqual(result, 1)

    def test_derivative_caching(self):
        #         f
        #         *
        #        / \
        #       /   \
        #     n2    n3
        #     +     +
        #    /  \  /  \
        #   x0   n1   x3
        #        *
        #       / \
        #     x1  x2
        #
        cct = Circuit()
        xs = cct.new_vars(4)
        n1 = cct.mul(xs[1], xs[2])
        n2 = cct.add(xs[0], n1)
        n3 = cct.add(n1, xs[3])
        f = cct.mul(n2, n3)

        cct.partial_derivatives(f, xs[0:2])

        num_nodes = cct.number_of_op_nodes
        num_arcs = cct.number_of_arcs

        #  requesting the same derivatives again should not add any extra nodes
        cct.partial_derivatives(f, xs[0:2])

        self.assertEqual(num_nodes, cct.number_of_op_nodes)
        self.assertEqual(num_arcs, cct.number_of_arcs)

    def test_derivative_caching_self_multiply(self):
        cct = Circuit()
        xs = cct.new_vars(4)
        n1 = cct.mul(xs[1], xs[2])
        n2 = cct.add(xs[0], n1)
        n3 = cct.add(n1, xs[3])
        f = cct.mul(n2, n3)

        cct.partial_derivatives(f, xs[0:2], self_multiply=True)

        num_nodes = cct.number_of_op_nodes
        num_arcs = cct.number_of_arcs

        #  requesting plain derivatives should not add any extra nodes
        cct.partial_derivatives(f, xs[0:2])

        self.assertEqual(num_nodes, cct.number_of_op_nodes)
        self.assertEqual(num_arcs, cct.number_of_arcs)

        #  requesting self-multiple derivatives should not add any extra nodes
        cct.partial_derivatives(f, xs[0:2], self_multiply=True)

        self.assertEqual(num_nodes, cct.number_of_op_nodes)
        self.assertEqual(num_arcs, cct.number_of_arcs)


class TestCircuitPickle(CircuitFixture):

    def assert_pickle_round_trip_cct(self, cct: Circuit) -> None:
        pkl: bytes = pickle.dumps(cct)
        clone = pickle.loads(pkl)
        self.assertCircuitsEqual(cct, clone)

    def assert_pickle_round_trip_node(self, node: CircuitNode) -> None:
        pkl: bytes = pickle.dumps(node)
        clone = pickle.loads(pkl)
        self.assertNodesEqual(node, clone)

    def test_pickle_node(self):
        cct = Circuit()
        nodes: Sequence[VarNode] = cct.new_vars(10)
        cct.add(nodes[0], nodes[1])
        cct.mul(nodes[8], nodes[9])
        nodes[0].const = 1

        for node in nodes:
            self.assert_pickle_round_trip_node(node)

    def test_empty(self):
        cct = Circuit()
        self.assert_pickle_round_trip_cct(cct)

    def test_circuit(self):
        cct = Circuit()
        nodes: Sequence[VarNode] = cct.new_vars(10)
        cct.add(nodes[0], nodes[1])
        cct.mul(nodes[8], nodes[9])
        nodes[0].const = 1

        self.assert_pickle_round_trip_cct(cct)

    def test_const_vars(self):
        cct = Circuit()
        x = cct.new_var()
        y = cct.new_var()
        x.const = None  # should already be clear.
        y.const = 123

        self.assert_pickle_round_trip_cct(cct)


class TestReachableNodes(CircuitFixture):

    def test_empty(self):
        cct = Circuit()

        result = list(cct.reachable_op_nodes(()))

        self.assertEqual(0, len(result))

    def test_empty_2(self):
        cct = Circuit()

        cct.add(cct.new_vars(2))

        result = list(cct.reachable_op_nodes(()))

        self.assertEqual(0, len(result))

    def test_const(self):
        cct = Circuit()

        n = cct.const(234)
        result = list(cct.reachable_op_nodes(n))

        self.assertEqual(0, len(result))

    def test_var(self):
        cct = Circuit()

        n = cct.new_var()
        result = list(cct.reachable_op_nodes(n))

        self.assertEqual(0, len(result))

    def test_single_op(self):
        cct = Circuit()

        op = cct.add(cct.new_var(), cct.const(123))
        result = list(cct.reachable_op_nodes(op))

        self.assertEqual(1, len(result))
        self.assertEqual(op, result[0])

    def test_two_ops(self):
        cct = Circuit()

        op1 = cct.add(cct.new_var(), cct.new_var())
        op2 = cct.add(op1, op1)
        result = list(cct.reachable_op_nodes(op2))

        self.assertEqual(2, len(result))
        self.assertEqual(op1, result[0])
        self.assertEqual(op2, result[1])


class TestTmpConst(CircuitFixture):

    def test_set_var(self):
        cct = Circuit()
        var = cct.new_var()

        self.assertFalse(var.is_const())
        with TmpConst(cct) as tmp:
            tmp.set_const(var, 1)
            self.assertTrue(var.is_const())
            self.assertEqual(var.const.value, 1)
        self.assertFalse(var.is_const())

    def test_set_var_idx(self):
        cct = Circuit()
        var = cct.new_var()

        self.assertFalse(var.is_const())
        with TmpConst(cct) as tmp:
            tmp.set_const(0, 1)
            self.assertTrue(var.is_const())
            self.assertEqual(var.const.value, 1)
        self.assertFalse(var.is_const())

    def test_unset_var(self):
        cct = Circuit()
        var = cct.new_var()
        var.const = 1

        self.assertTrue(var.is_const())
        with TmpConst(cct) as tmp:
            tmp.set_const(var, None)
            self.assertFalse(var.is_const())
        self.assertTrue(var.is_const())

    def test_unset_var_idx(self):
        cct = Circuit()
        var = cct.new_var()
        var.const = 1

        self.assertTrue(var.is_const())
        with TmpConst(cct) as tmp:
            tmp.set_const(0, None)
            self.assertFalse(var.is_const())
        self.assertTrue(var.is_const())

    def test_set_var_iter(self):
        cct = Circuit()
        x = cct.new_var()
        y = cct.new_var()

        self.assertFalse(x.is_const())
        self.assertFalse(y.is_const())
        with TmpConst(cct) as tmp:
            tmp.set_const((x, y), 1)
            self.assertTrue(x.is_const())
            self.assertTrue(y.is_const())
        self.assertFalse(x.is_const())
        self.assertFalse(y.is_const())

    def test_unset_var_iter(self):
        cct = Circuit()
        x = cct.new_var()
        y = cct.new_var()
        x.const = 1
        y.const = 2

        self.assertTrue(x.is_const())
        self.assertTrue(y.is_const())
        with TmpConst(cct) as tmp:
            tmp.set_const((x, y), None)
            self.assertFalse(x.is_const())
            self.assertFalse(y.is_const())
        self.assertTrue(x.is_const())
        self.assertTrue(y.is_const())
        self.assertEqual(x.const.value, 1)
        self.assertEqual(y.const.value, 2)


if __name__ == '__main__':
    test_main()
