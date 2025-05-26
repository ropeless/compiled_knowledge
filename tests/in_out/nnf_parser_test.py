import unittest

from ck.circuit_compiler.interpret_compiler import compile_circuit
from ck.in_out.parse_ace_lmap import LiteralMap
from ck.in_out.parse_ace_nnf import read_nnf, read_nnf_with_literal_map
from ck.in_out.parser_utils import ParseError
from ck.pgm import Indicator
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
        self.assertEqual(program(123, 456), 123 * 456)

    def test_end_of_file(self):
        input_data = (
            """
            nnf 3 2 0
            l 10
            l 20
            * 2 0 1
            %
            + 1 2 3 4 5 6 7 8 9
            """
        )

        top_node, slot_map = read_nnf(input_data)
        circuit = top_node.circuit

        self.assertEqual(circuit.number_of_vars, 2)

        self.assertEqual(len(slot_map), 2)
        self.assertEqual(slot_map[10], 0)
        self.assertEqual(slot_map[20], 1)

        program = Program(compile_circuit(top_node))
        self.assertEqual(program(123, 456), 123 * 456)

    def test_check_no_header(self):
        input_data = (
            """
            l 10
            l 20
            * 2 0 1
            """
        )

        with self.assertRaises(ParseError) as context:
            read_nnf(input_data)

        self.assertEqual(context.exception.line_num, 2)  # counting from 1, including blank line

    def test_check_malformed_header(self):
        input_data = (
            """
            nnf 1 2 3 4 5
            l 10
            l 20
            * 2 0 1
            """
        )

        with self.assertRaises(ParseError) as context:
            read_nnf(input_data)

        self.assertEqual(context.exception.line_num, 2)  # counting from 1, including blank line

    def test_check_malformed_literal(self):
        input_data = (
            """
            nnf 3 2 0
            l 10
            l 2 0 1
            * 2 0 1
            """
        )

        with self.assertRaises(ParseError) as context:
            read_nnf(input_data)

        self.assertEqual(context.exception.line_num, 4)  # counting from 1, including blank line

    def test_check_wrong_number_of_args(self):
        input_data = (
            """
            nnf 3 2 0
            l 10
            l 20
            * 2 0 1 1
            """
        )

        with self.assertRaises(ParseError) as context:
            read_nnf(input_data)

        self.assertEqual(context.exception.line_num, 5)  # counting from 1, including blank line

    def test_check_wrong_code(self):
        input_data = (
            """
            nnf 3 2 0
            l 10
            X 20
            * 2 0 1
            """
        )

        with self.assertRaises(ParseError) as context:
            read_nnf(input_data)
        self.assertEqual(context.exception.line_num, 4)  # counting from 1, including blank line

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
            read_nnf(input_data, check_header=True)

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
            read_nnf(input_data, check_header=True)

        read_nnf(input_data, check_header=False)
        self.assertTrue(True)

    def test_parse_nnf(self):
        # literal_1 * literal_3 + literal_2 * literal_4
        # var[0] <-> literal_1
        # var[1] <-> literal_3
        # var[2] <-> literal_2
        # var[3] <-> literal_4
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
        top_node, var_literals = read_nnf(input_data)
        circuit = top_node.circuit

        self.assertEqual(circuit.number_of_vars, 4)

        self.assertEqual(len(var_literals), 4)
        self.assertEqual(var_literals[1], 0)
        self.assertEqual(var_literals[3], 1)

        self.assertEqual(var_literals[2], 2)
        self.assertEqual(var_literals[4], 3)

        program = Program(compile_circuit(top_node))
        self.assertEqual(program(1, 0.25, 3, 4.75), 1 * 0.25 + 3 * 4.75)

    def test_parse_nnf_with_literal_map_const_params(self):
        # literal_1 * literal_3 + literal_2 * literal_4
        #
        # var[0] Indicator(0, 0)
        # var[1] Indicator(0, 1) <-> literal_1
        # var[2] Indicator(1, 0)
        # var[3] Indicator(1, 1) <-> literal_2
        #
        # constant 123 <-> literal_3
        # constant 456 <-> literal_4
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
        indicators = [
            Indicator(0, 0),
            Indicator(0, 1),
            Indicator(1, 0),
            Indicator(1, 1),
        ]
        literal_map = LiteralMap(
            rvs={},
            indicators={1: Indicator(0, 1), 2: Indicator(1, 1)},
            params={3: 123, 4: 456},
        )

        top_node, slot_map, params = read_nnf_with_literal_map(
            input_data,
            literal_map,
            indicators=indicators,
            const_parameters=True
        )
        circuit = top_node.circuit

        self.assertEqual(circuit.number_of_vars, 4)
        self.assertEqual(len(slot_map), 4)
        self.assertEqual(len(params), 0)

        self.assertEqual(slot_map[Indicator(0, 0)], 0)
        self.assertEqual(slot_map[Indicator(0, 1)], 1)
        self.assertEqual(slot_map[Indicator(1, 0)], 2)
        self.assertEqual(slot_map[Indicator(1, 1)], 3)

        program = Program(compile_circuit(top_node))
        self.assertEqual(program(1, 0.25, 3, 4.75), 123 * 0.25 + 456 * 4.75)

    def test_parse_nnf_with_literal_map_non_const_params(self):
        # literal_1 * literal_3 + literal_2 * literal_4
        #
        # var[0] Indicator(0, 0)
        # var[1] Indicator(0, 1) <-> literal_1
        # var[2] Indicator(1, 0)
        # var[3] Indicator(1, 1) <-> literal_2
        # var[4] param literal_3 = 123
        # var[5] param literal_4 = 123
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
        indicators = [
            Indicator(0, 0),
            Indicator(0, 1),
            Indicator(1, 0),
            Indicator(1, 1),
        ]
        literal_map = LiteralMap(
            rvs={},
            indicators={1: Indicator(0, 1), 2: Indicator(1, 1)},
            params={3: 123, 4: 456},
        )

        top_node, slot_map, params = read_nnf_with_literal_map(
            input_data,
            literal_map,
            indicators=indicators,
            const_parameters=False
        )
        circuit = top_node.circuit

        self.assertEqual(circuit.number_of_vars, 6)
        self.assertEqual(len(slot_map), 4)
        self.assertEqual(len(params), 2)

        self.assertEqual(slot_map[Indicator(0, 0)], 0)
        self.assertEqual(slot_map[Indicator(0, 1)], 1)
        self.assertEqual(slot_map[Indicator(1, 0)], 2)
        self.assertEqual(slot_map[Indicator(1, 1)], 3)

        self.assertEqual(params[0], 123)
        self.assertEqual(params[1], 456)

        program = Program(compile_circuit(top_node))
        self.assertEqual(program(1, 0.25, 3, 4.75, 123, 456), 123 * 0.25 + 456 * 4.75)


if __name__ == '__main__':
    unittest.main()
