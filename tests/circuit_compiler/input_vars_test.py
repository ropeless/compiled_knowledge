from ck.circuit_compiler.support.input_vars import infer_input_vars, InferVars
from ck.circuit import Circuit
from tests.helpers.unittest_fixture import Fixture, test_main


class InputVarsTest(Fixture):

    def test_infer_vars_empty(self):
        inferred_vars = infer_input_vars(
            circuit=None,
            results=[],
            input_vars=[]
        )

        self.assertArrayEqual(inferred_vars, [])

    def test_only_input_vars_single(self):
        circuit = Circuit()
        x0 = circuit.new_var()

        inferred_vars = infer_input_vars(
            circuit=None,
            results=(),
            input_vars=x0
        )

        self.assertArrayEqual(inferred_vars, [x0])

    def test_only_input_vars_multi(self):
        circuit = Circuit()
        x0 = circuit.new_var()
        x1 = circuit.new_var()
        x2 = circuit.new_var()

        inferred_vars = infer_input_vars(
            circuit=None,
            results=(),
            input_vars=[x0, x1, x2]
        )

        self.assertArrayEqual(inferred_vars, [x0, x1, x2])

    def test_infer_vars_all(self):
        circuit = Circuit()
        x = circuit.new_vars(4)
        result = x[0] + x[2]

        inferred_vars = infer_input_vars(
            circuit=None,
            results=[result],
            input_vars=InferVars.ALL
        )

        self.assertArrayEqual(inferred_vars, [x[0], x[1], x[2], x[3]])

    def test_infer_vars_ref(self):
        circuit = Circuit()
        x = circuit.new_vars(4)
        result = x[0] + x[2]

        inferred_vars = infer_input_vars(
            circuit=None,
            results=[result],
            input_vars=InferVars.REF
        )

        self.assertArrayEqual(inferred_vars, [x[0], x[2]])

    def test_infer_vars_low(self):
        circuit = Circuit()
        x = circuit.new_vars(4)
        result = x[0] + x[2]

        inferred_vars = infer_input_vars(
            circuit=None,
            results=[result],
            input_vars=InferVars.LOW
        )

        self.assertArrayEqual(inferred_vars, [x[0], x[1], x[2]])

    def test_infer_vars_low_no_results(self):
        circuit = Circuit()
        x = circuit.new_vars(4)
        _ = x[0] + x[2]

        inferred_vars = infer_input_vars(
            circuit=circuit,
            results=[],
            input_vars=InferVars.LOW
        )

        self.assertArrayEqual(inferred_vars, [])

    def test_compatible_input_output(self):
        circuit = Circuit()
        x = circuit.new_vars(4)
        result = x[0] + x[2]

        inferred_vars = infer_input_vars(
            circuit=None,
            results=[result],
            input_vars=(x[2], x[0])
        )

        self.assertArrayEqual(inferred_vars, [x[2], x[0]])

    def test_incompatible_circuit_input_output(self):
        circuit = Circuit()
        x = circuit.new_vars(4)
        result = x[0] + x[2]

        circuit2 = Circuit()
        y = circuit2.new_vars(4)

        with self.assertRaises(ValueError):
            _ = infer_input_vars(
                circuit=None,
                results=[result],
                input_vars=y
            )

    def test_insufficient_input_vars(self):
        circuit = Circuit()
        x = circuit.new_vars(4)
        result = x[0] + x[2]

        with self.assertRaises(ValueError):
            _ = infer_input_vars(
                circuit=None,
                results=[result],
                input_vars=x[1]
            )

    def test_compatible_input_output_circuit(self):
        circuit = Circuit()
        x = circuit.new_vars(4)
        result = x[0] + x[2]

        inferred_vars = infer_input_vars(
            circuit=circuit,
            results=[result],
            input_vars=(x[2], x[0])
        )

        self.assertArrayEqual(inferred_vars, [x[2], x[0]])

    def test_incompatible_input_output_circuit(self):
        circuit = Circuit()
        x = circuit.new_vars(4)
        result = x[0] + x[2]

        circuit2 = Circuit()

        with self.assertRaises(ValueError):
            _ = infer_input_vars(
                circuit=circuit2,
                results=[result],
                input_vars=x[0]
            )

    def test_duplicate_inputs(self):
        circuit = Circuit()
        x = circuit.new_vars(4)
        result = x[0]

        with self.assertRaises(ValueError):
            _ = infer_input_vars(
                circuit=circuit,
                results=[result],
                input_vars=[x[0], x[0]]
            )

    def test_replicated_inputs(self):
        circuit = Circuit()
        x = circuit.new_vars(4)
        result = x[0] * x[2] + x[0] * x[2]

        inferred_vars = infer_input_vars(
            circuit=None,
            results=[result],
            input_vars=InferVars.REF
        )

        self.assertArrayEqual(inferred_vars, [x[0], x[2]])

    def test_avoid_const_nodes(self):
        circuit = Circuit()
        x = circuit.new_vars(4)
        result = x[0] * circuit.const(123) + circuit.const(456) * x[2]

        inferred_vars = infer_input_vars(
            circuit=None,
            results=[result],
            input_vars=InferVars.REF
        )

        self.assertArrayEqual(inferred_vars, [x[0], x[2]])


if __name__ == '__main__':
    test_main()
