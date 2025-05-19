import unittest

from ck.circuit_compiler.interpret_compiler import compile_circuit
from ck.in_out.parse_ace_nnf import read_nnf
from ck.in_out.parser_utils import ParseError
from ck.program import Program


class TestNnfParser(unittest.TestCase):

    def test_parse_empty(self):
        input_data = (
            """
            c this is a comment
            c
            c the 'nnf' line defines the sizes
            nnf 0 0 0
            """
        )

        top_node, slot_map = read_nnf(input_data)
        circuit = top_node.circuit

        self.assertEqual(circuit.number_of_vars, 0)
        self.assertTrue(top_node.is_zero())
        self.assertEqual(len(slot_map), 0)

    def test_parse_simple_add(self):
        input_data = (
            """
            c this is a comment
            c
            c the 'nnf' line defines the sizes
            nnf 3 2 0
            l 10
            l 20
            + 2 0 1
            """
        )

        top_node, slot_map = read_nnf(input_data)
        circuit = top_node.circuit

        self.assertEqual(circuit.number_of_vars, 2)

        self.assertEqual(len(slot_map), 2)
        self.assertEqual(slot_map[10], 0)
        self.assertEqual(slot_map[20], 1)

        program = Program(compile_circuit(top_node))

        self.assertEqual(program(123, 456), 579)

    def test_parse_simple_mul(self):
        input_data = (
            """
            c this is a comment
            c
            c the 'nnf' line defines the sizes
            nnf 3 2 0
            l 10
            l 20
            * 2 0 1
            """
        )

        top_node, slot_map = read_nnf(input_data)
        circuit = top_node.circuit

        self.assertEqual(circuit.number_of_vars, 2)

        self.assertEqual(len(slot_map), 2)
        self.assertEqual(slot_map[10], 0)
        self.assertEqual(slot_map[20], 1)

        program = Program(compile_circuit(top_node))
        self.assertEqual(program(123, 456), 56088)

    def test_header_check_underestimate(self):
        input_data = (
            """
            nnf 5 5 5
            + 0
            l 1
            + 2 0 1
            """
        )
        with self.assertRaises(ParseError):
            read_nnf(input, check_header=True)

        read_nnf(input_data, check_header=False)
        self.assertTrue(True)

    def test_header_check_overestimate(self):
        input_data = (
            """
            nnf 5 5 5
            + 0
            l 1
            + 2 0 1
            """
        )
        with self.assertRaises(ParseError):
            read_nnf(input, check_header=True)

        read_nnf(input_data, check_header=False)
        self.assertTrue(True)

    def test_parse_nnf(self):
        # Network created is a single Boolean rv with state probabilities
        # of 0.75 and 0.25
        input_data = """
        nnf 7 6 4
        L 1
        L 3
        A 2 0 1
        L 2
        L 4
        A 2 3 4
        O 1 2 2 5
        """
        top_node, slot_map = read_nnf(input_data)
        circuit = top_node.circuit

        self.assertEqual(circuit.number_of_vars, 4)

        self.assertEqual(len(slot_map), 4)
        self.assertEqual(slot_map[1], 0)
        self.assertEqual(slot_map[3], 1)

        self.assertEqual(slot_map[2], 2)
        self.assertEqual(slot_map[4], 3)

        program = Program(compile_circuit(top_node))
        self.assertEqual(program(1, 0.25, 0, 0.75), 0.25)


if __name__ == '__main__':
    unittest.main()
