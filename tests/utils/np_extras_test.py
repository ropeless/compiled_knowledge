import numpy as np

from ck.utils.np_extras import dtype_for_number_of_states, DTypeStates, NDArray
from tests.helpers.unittest_fixture import Fixture, test_main


class TestNPExtras(Fixture):

    def test_infer_dtype(self):
        self.assertEqual(dtype_for_number_of_states(0), np.uint8)
        self.assertEqual(dtype_for_number_of_states(1), np.uint8)
        self.assertEqual(dtype_for_number_of_states(2 ** 8), np.uint8)
        self.assertEqual(dtype_for_number_of_states(2 ** 8 + 1), np.uint16)
        self.assertEqual(dtype_for_number_of_states(2 ** 16), np.uint16)
        self.assertEqual(dtype_for_number_of_states(2 ** 16 + 1), np.uint32)
        self.assertEqual(dtype_for_number_of_states(2 ** 32), np.uint32)
        self.assertEqual(dtype_for_number_of_states(2 ** 32 + 1), np.uint64)
        self.assertEqual(dtype_for_number_of_states(2 ** 64), np.uint64)

        with self.assertRaises(ValueError):
            dtype_for_number_of_states(2 ** 64 + 1)

    def _check_inferred_dtype_works(self, number_of_states: int) -> None:
        dtype: DTypeStates = dtype_for_number_of_states(number_of_states)
        array: NDArray = np.array([0, number_of_states - 1], dtype=dtype)
        self.assertEqual(array[0], 0, msg=f'number_of_state = {number_of_states}')
        self.assertEqual(array[1], number_of_states - 1, msg=f'number_of_state = {number_of_states}')

    def test_inferred_dtype_works(self) -> None:
        test_number_of_states = [
            1,
            2 ** 8 - 1,
            2 ** 8,
            2 ** 8 + 1,
            2 ** 16 - 1,
            2 ** 16,
            2 ** 16 + 1,
            2 ** 32 - 1,
            2 ** 32,
            2 ** 32 + 1,
            2 ** 64 - 1,
            2 ** 64,
        ]
        for number_of_states in test_number_of_states:
            self._check_inferred_dtype_works(number_of_states)


if __name__ == '__main__':
    test_main()
