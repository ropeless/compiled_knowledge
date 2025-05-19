import numpy as np

from ck.utils.np_extras import dtype_for_number_of_states
from tests.helpers.unittest_fixture import Fixture, test_main


class TestNPExtras(Fixture):

    def test_infer_dtype(self):
        self.assertEqual(dtype_for_number_of_states(1), np.uint8)
        self.assertEqual(dtype_for_number_of_states(2 ** 8 - 1), np.uint8)
        self.assertEqual(dtype_for_number_of_states(2 ** 8), np.uint16)
        self.assertEqual(dtype_for_number_of_states(2 ** 16 - 1), np.uint16)
        self.assertEqual(dtype_for_number_of_states(2 ** 16), np.uint32)
        self.assertEqual(dtype_for_number_of_states(2 ** 32 - 1), np.uint32)
        self.assertEqual(dtype_for_number_of_states(2 ** 32), np.uint64)
        self.assertEqual(dtype_for_number_of_states(2 ** 64 - 1), np.uint64)

        with self.assertRaises(ValueError):
            dtype_for_number_of_states(2 ** 64)


if __name__ == '__main__':
    test_main()
