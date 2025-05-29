from typing import Tuple

from ck.circuit import Circuit, CircuitNode
from ck.circuit_compiler.interpret_compiler import compile_circuit
from ck.pgm_compiler.support.circuit_table import CircuitTable, sum_out, sum_out_all, product

from tests.helpers.unittest_fixture import Fixture, test_main


class TestCircuitTable(Fixture):

    def test_empty(self):
        circuit = Circuit()
        table = CircuitTable(circuit, ())

        self.assertIs(table.circuit, circuit)
        self.assertEqual(table.rv_idxs, ())
        self.assertEqual(len(table), 0)
        self.assertIs(table.top(), circuit.zero)
        self.assertEqual(list(table.keys()), [])

    def test_one(self):
        circuit = Circuit()
        var = circuit.new_var()

        table = CircuitTable(circuit, ())
        table[()] = var

        self.assertIs(table.circuit, circuit)
        self.assertEqual(table.rv_idxs, ())
        self.assertEqual(len(table), 1)
        self.assertIs(table.top(), var)
        self.assertEqual(list(table.keys()), [()])
        self.assertIs(table[()], var)

    def test_sum_out_all(self):
        circuit = Circuit()
        table = CircuitTable(circuit, (0, 1, 2))

        table[(0, 0, 0)] = circuit.const(1)
        table[(0, 0, 1)] = circuit.const(2)
        table[(0, 1, 0)] = circuit.const(3)
        table[(0, 1, 1)] = circuit.const(4)
        table[(1, 0, 0)] = circuit.const(5)
        table[(1, 0, 1)] = circuit.const(6)
        table[(1, 1, 0)] = circuit.const(7)
        table[(1, 1, 1)] = circuit.const(8)

        new_table: CircuitTable = sum_out_all(table)

        self.assertIs(new_table.circuit, circuit)
        self.assertEqual(new_table.rv_idxs, ())
        self.assertEqual(len(new_table), 1)
        self.assertEqual(list(new_table.keys()), [()])

        self.assertEqual(_reduce(new_table.top()), 36)
        self.assertEqual(_reduce(new_table[()]), 36)

    def test_sum_out_one(self):
        circuit = Circuit()
        table = CircuitTable(circuit, (0, 1, 2))

        table[(0, 0, 0)] = circuit.const(1)
        table[(0, 0, 1)] = circuit.const(2)
        table[(0, 1, 0)] = circuit.const(3)
        table[(0, 1, 1)] = circuit.const(4)
        table[(1, 0, 0)] = circuit.const(5)
        table[(1, 0, 1)] = circuit.const(6)
        table[(1, 1, 0)] = circuit.const(7)
        table[(1, 1, 1)] = circuit.const(8)

        new_table: CircuitTable = sum_out(table, [1])

        self.assertIs(new_table.circuit, circuit)
        self.assertEqual(new_table.rv_idxs, (0, 2))
        self.assertEqual(len(new_table), 4)

        self.assertEqual(_reduce(new_table[(0, 0)]), 1 + 3)
        self.assertEqual(_reduce(new_table[(0, 1)]), 2 + 4)
        self.assertEqual(_reduce(new_table[(1, 0)]), 5 + 7)
        self.assertEqual(_reduce(new_table[(1, 1)]), 6 + 8)

    def test_sum_out_two(self):
        circuit = Circuit()
        table = CircuitTable(circuit, (0, 1, 2))

        table[(0, 0, 0)] = circuit.const(1)
        table[(0, 0, 1)] = circuit.const(2)
        table[(0, 1, 0)] = circuit.const(3)
        table[(0, 1, 1)] = circuit.const(4)
        table[(1, 0, 0)] = circuit.const(5)
        table[(1, 0, 1)] = circuit.const(6)
        table[(1, 1, 0)] = circuit.const(7)
        table[(1, 1, 1)] = circuit.const(8)

        new_table: CircuitTable = sum_out(table, [0, 2])

        self.assertIs(new_table.circuit, circuit)
        self.assertEqual(new_table.rv_idxs, (1,))
        self.assertEqual(len(new_table), 2)

        self.assertEqual(_reduce(new_table[(0,)]), 1 + 2 + 5 + 6)
        self.assertEqual(_reduce(new_table[(1,)]), 3 + 4 + 7 + 8)

    def test_sum_out_everything(self):
        circuit = Circuit()
        table = CircuitTable(circuit, (0, 1, 2))

        table[(0, 0, 0)] = circuit.const(1)
        table[(0, 0, 1)] = circuit.const(2)
        table[(0, 1, 0)] = circuit.const(3)
        table[(0, 1, 1)] = circuit.const(4)
        table[(1, 0, 0)] = circuit.const(5)
        table[(1, 0, 1)] = circuit.const(6)
        table[(1, 1, 0)] = circuit.const(7)
        table[(1, 1, 1)] = circuit.const(8)

        new_table: CircuitTable = sum_out(table, [0, 2, 1])

        self.assertIs(new_table.circuit, circuit)
        self.assertEqual(new_table.rv_idxs, ())
        self.assertEqual(len(new_table), 1)
        self.assertEqual(list(new_table.keys()), [()])

        self.assertEqual(_reduce(new_table.top()), 36)
        self.assertEqual(_reduce(new_table[()]), 36)

    def test_product_with_common_rv(self):
        circuit = Circuit()

        table_1 = CircuitTable(circuit, (0, 1))
        table_1[(0, 0)] = circuit.const(2)
        table_1[(0, 1)] = circuit.const(3)
        table_1[(1, 0)] = circuit.const(5)
        table_1[(1, 1)] = circuit.const(7)

        table_2 = CircuitTable(circuit, (1, 2))
        table_2[(0, 0)] = circuit.const(11)
        table_2[(0, 1)] = circuit.const(13)
        table_2[(1, 0)] = circuit.const(17)
        table_2[(1, 1)] = circuit.const(19)

        table_3 = product(table_1, table_2)

        self.assertIs(table_3.circuit, circuit)
        self.assertArraySetEqual(table_3.rv_idxs, (0, 1, 2))
        self.assertEqual(len(table_3), 8)

        def inst(*vals: int) -> Tuple[int, ...]:
            return tuple(
                vals[i]
                for i in table_3.rv_idxs
            )

        self.assertEqual(_reduce(table_3[inst(0, 0, 0)]), 2 * 11)
        self.assertEqual(_reduce(table_3[inst(0, 0, 1)]), 2 * 13)
        self.assertEqual(_reduce(table_3[inst(0, 1, 0)]), 3 * 17)
        self.assertEqual(_reduce(table_3[inst(0, 1, 1)]), 3 * 19)
        self.assertEqual(_reduce(table_3[inst(1, 0, 0)]), 5 * 11)
        self.assertEqual(_reduce(table_3[inst(1, 0, 1)]), 5 * 13)
        self.assertEqual(_reduce(table_3[inst(1, 1, 0)]), 7 * 17)
        self.assertEqual(_reduce(table_3[inst(1, 1, 1)]), 7 * 19)

    def test_product_with_common_rv_swapped(self):
        circuit = Circuit()

        table_1 = CircuitTable(circuit, (0, 1))
        table_1[(0, 0)] = circuit.const(2)
        table_1[(0, 1)] = circuit.const(3)
        table_1[(1, 0)] = circuit.const(5)
        table_1[(1, 1)] = circuit.const(7)

        table_2 = CircuitTable(circuit, (1, 2))
        table_2[(0, 0)] = circuit.const(11)
        table_2[(0, 1)] = circuit.const(13)
        table_2[(1, 0)] = circuit.const(17)
        table_2[(1, 1)] = circuit.const(19)

        table_3 = product(table_2, table_1)

        self.assertIs(table_3.circuit, circuit)
        self.assertArraySetEqual(table_3.rv_idxs, (0, 1, 2))
        self.assertEqual(len(table_3), 8)

        def inst(*vals: int) -> Tuple[int, ...]:
            return tuple(
                vals[i]
                for i in table_3.rv_idxs
            )

        self.assertEqual(_reduce(table_3[inst(0, 0, 0)]), 2 * 11)
        self.assertEqual(_reduce(table_3[inst(0, 0, 1)]), 2 * 13)
        self.assertEqual(_reduce(table_3[inst(0, 1, 0)]), 3 * 17)
        self.assertEqual(_reduce(table_3[inst(0, 1, 1)]), 3 * 19)
        self.assertEqual(_reduce(table_3[inst(1, 0, 0)]), 5 * 11)
        self.assertEqual(_reduce(table_3[inst(1, 0, 1)]), 5 * 13)
        self.assertEqual(_reduce(table_3[inst(1, 1, 0)]), 7 * 17)
        self.assertEqual(_reduce(table_3[inst(1, 1, 1)]), 7 * 19)

    def test_product_without_common_rv(self):
        circuit = Circuit()

        table_1 = CircuitTable(circuit, (0, 1))
        table_1[(0, 0)] = circuit.const(2)
        table_1[(0, 1)] = circuit.const(3)
        table_1[(1, 0)] = circuit.const(5)
        table_1[(1, 1)] = circuit.const(7)

        table_2 = CircuitTable(circuit, (2, 3))
        table_2[(0, 0)] = circuit.const(11)
        table_2[(0, 1)] = circuit.const(13)
        table_2[(1, 0)] = circuit.const(17)
        table_2[(1, 1)] = circuit.const(19)

        table_3 = product(table_1, table_2)

        self.assertIs(table_3.circuit, circuit)
        self.assertEqual(table_3.rv_idxs, (0, 1, 2, 3))
        self.assertEqual(len(table_3), 16)

        self.assertEqual(_reduce(table_3[(0, 0, 0, 0)]), 2 * 11)
        self.assertEqual(_reduce(table_3[(0, 0, 0, 1)]), 2 * 13)
        self.assertEqual(_reduce(table_3[(0, 0, 1, 0)]), 2 * 17)
        self.assertEqual(_reduce(table_3[(0, 0, 1, 1)]), 2 * 19)

        self.assertEqual(_reduce(table_3[(0, 1, 0, 0)]), 3 * 11)
        self.assertEqual(_reduce(table_3[(0, 1, 0, 1)]), 3 * 13)
        self.assertEqual(_reduce(table_3[(0, 1, 1, 0)]), 3 * 17)
        self.assertEqual(_reduce(table_3[(0, 1, 1, 1)]), 3 * 19)

        self.assertEqual(_reduce(table_3[(1, 0, 0, 0)]), 5 * 11)
        self.assertEqual(_reduce(table_3[(1, 0, 0, 1)]), 5 * 13)
        self.assertEqual(_reduce(table_3[(1, 0, 1, 0)]), 5 * 17)
        self.assertEqual(_reduce(table_3[(1, 0, 1, 1)]), 5 * 19)

        self.assertEqual(_reduce(table_3[(1, 1, 0, 0)]), 7 * 11)
        self.assertEqual(_reduce(table_3[(1, 1, 0, 1)]), 7 * 13)
        self.assertEqual(_reduce(table_3[(1, 1, 1, 0)]), 7 * 17)
        self.assertEqual(_reduce(table_3[(1, 1, 1, 1)]), 7 * 19)

    def test_product_subsumed(self):
        circuit = Circuit()

        table_1 = CircuitTable(circuit, (0, 1))
        table_1[(0, 0)] = circuit.const(2)
        table_1[(0, 1)] = circuit.const(3)
        table_1[(1, 0)] = circuit.const(5)
        table_1[(1, 1)] = circuit.const(7)

        table_2 = CircuitTable(circuit, (1,))
        table_2[(0,)] = circuit.const(11)
        table_2[(1,)] = circuit.const(13)

        table_3 = product(table_1, table_2)

        self.assertIs(table_3.circuit, circuit)
        self.assertEqual(table_3.rv_idxs, (0, 1))
        self.assertEqual(len(table_3), 4)

        self.assertEqual(_reduce(table_3[(0, 0)]), 2 * 11)
        self.assertEqual(_reduce(table_3[(0, 1)]), 3 * 13)
        self.assertEqual(_reduce(table_3[(1, 0)]), 5 * 11)
        self.assertEqual(_reduce(table_3[(1, 1)]), 7 * 13)


def _reduce(node: CircuitNode) -> float | int:
    """
    Reduce a constant circuit expression to a single value.
    Uses a CircuitInterpreter.
    """
    program = compile_circuit(node, input_vars=[])
    return program([]).item()


if __name__ == '__main__':
    test_main()
