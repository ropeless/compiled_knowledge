import numpy as np

from ck.dataset import HardDataset, SoftDataset
from ck.pgm import PGM
from ck.utils.np_extras import NDArray
from tests.helpers.unittest_fixture import Fixture, test_main


class TestHardDataset(Fixture):

    def test_empty(self):
        dataset = HardDataset()
        self.assertEqual(len(dataset), 0)
        self.assertEqual(len(dataset.rvs), 0)
        self.assertEqual(len(dataset.weights), 0)

    def test_with_length(self):
        dataset = HardDataset(length=5)

        self.assertEqual(len(dataset), 5)
        self.assertEqual(len(dataset.rvs), 0)
        self.assertEqual(len(dataset.weights), 5)
        self.assertEqual(dataset.weights[0], 1)
        self.assertEqual(dataset.weights[1], 1)
        self.assertEqual(dataset.weights[2], 1)
        self.assertEqual(dataset.weights[3], 1)
        self.assertEqual(dataset.weights[4], 1)

    def test_with_weights(self):
        dataset = HardDataset(weights=np.array([1, 2, 3, 4, 5]))

        self.assertEqual(len(dataset), 5)
        self.assertEqual(len(dataset.rvs), 0)
        self.assertEqual(len(dataset.weights), 5)
        self.assertEqual(dataset.weights[0], 1)
        self.assertEqual(dataset.weights[1], 2)
        self.assertEqual(dataset.weights[2], 3)
        self.assertEqual(dataset.weights[3], 4)
        self.assertEqual(dataset.weights[4], 5)

    def test_total_weight(self):
        dataset = HardDataset(weights=np.array([1, 2, 3, 4, 5]))
        self.assertEqual(dataset.total_weight(), 15)

    def test_add_rv(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        dataset = HardDataset(length=5)

        self.assertEqual(dataset.rvs, ())

        rv_data = dataset.add_rv(x)

        self.assertEqual(dataset.rvs, (x,))
        self.assertEqual(rv_data.shape, (5,))
        self.assertTrue(np.all(rv_data == 0))

        rv_data = dataset.add_rv(y)

        self.assertEqual(dataset.rvs, (x, y))
        self.assertEqual(rv_data.shape, (5,))
        self.assertTrue(np.all(rv_data == 0))

    def test_remove_rv(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))
        z = pgm.new_rv('z', ('1', '2'))

        dataset = HardDataset([
            (x, [0, 0, 1, 1, 0]),
            (y, [1, 1, 1, 0, 0]),
            (z, [1, 1, 0, 1, 1]),
        ])

        self.assertEqual(dataset.rvs, (x, y, z))

        dataset.remove_rv(y)

        self.assertEqual(dataset.rvs, (x, z))
        self.assertArrayEqual(dataset.state_idxs(x), [0, 0, 1, 1, 0])
        self.assertArrayEqual(dataset.state_idxs(z), [1, 1, 0, 1, 1])

    def test_add_rv_from_state_idxs(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        dataset = HardDataset(length=5)

        dataset.add_rv_from_state_idxs(x, [0, 0, 1, 1, 0])
        self.assertEqual(len(dataset.rvs), 1)
        self.assertIs(dataset.rvs[0], x)

        rv_data = dataset.state_idxs(x)
        self.assertArrayEqual(rv_data, [0, 0, 1, 1, 0])

        dataset.add_rv_from_state_idxs(y, [0, 2, 1, 1, 0])
        self.assertEqual(len(dataset.rvs), 2)
        self.assertIs(dataset.rvs[0], x)
        self.assertIs(dataset.rvs[1], y)

        rv_data = dataset.state_idxs(y)
        self.assertArrayEqual(rv_data, [0, 2, 1, 1, 0])

    def test_add_rv_from_states(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        dataset = HardDataset(length=5)

        dataset.add_rv_from_states(x, [True, True, False, False, True])
        self.assertEqual(len(dataset.rvs), 1)
        self.assertIs(dataset.rvs[0], x)

        rv_data = dataset.state_idxs(x)
        self.assertArrayEqual(rv_data, [0, 0, 1, 1, 0])

        dataset.add_rv_from_states(y, ['yes', 'maybe', 'no', 'no', 'yes'])
        self.assertEqual(len(dataset.rvs), 2)
        self.assertIs(dataset.rvs[0], x)
        self.assertIs(dataset.rvs[1], y)

        rv_data = dataset.state_idxs(y)
        self.assertArrayEqual(rv_data, [0, 2, 1, 1, 0])

    def test_add_rv_from_state_weights(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        dataset = HardDataset(length=5)
        self.assertArrayEqual(dataset.weights, [1, 1, 1, 1, 1])

        dataset.add_rv_from_state_weights(
            x,
            np.array([
                [0.6, 0.4],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.3, 0.7],
                [0.9, 0.1],
            ]),
        )
        self.assertEqual(len(dataset.rvs), 1)
        self.assertIs(dataset.rvs[0], x)

        rv_data = dataset.state_idxs(x)
        self.assertArrayEqual(rv_data, [0, 0, 1, 1, 0])
        self.assertArrayAlmostEqual(dataset.weights, [1, 1, 1, 1, 1])

        dataset.add_rv_from_state_weights(
            y,
            np.array([
                [0.6, 0.3, 0.1],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.3, 0.4, 0.3],
                [0.8, 0.1, 0.1],
            ]),
        )
        self.assertEqual(len(dataset.rvs), 2)
        self.assertIs(dataset.rvs[0], x)
        self.assertIs(dataset.rvs[1], y)
        self.assertArrayAlmostEqual(dataset.weights, [1, 1, 1, 1, 1])

        rv_data = dataset.state_idxs(y)
        self.assertArrayEqual(rv_data, [0, 2, 1, 1, 0])

    def test_add_rv_from_state_weights_unnormalised_adjusted(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        dataset = HardDataset(length=5)
        self.assertArrayAlmostEqual(dataset.weights, [1, 1, 1, 1, 1])

        dataset.add_rv_from_state_weights(
            x,
            np.array([
                [0.6, 0.4],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.3, 0.7],
                [0.4, 0.1],  # weight 0.5
            ]),
        )
        self.assertEqual(len(dataset.rvs), 1)
        self.assertIs(dataset.rvs[0], x)

        rv_data = dataset.state_idxs(x)
        self.assertArrayEqual(rv_data, [0, 0, 1, 1, 0])
        self.assertArrayAlmostEqual(dataset.weights, [1, 1, 1, 1, 0.5])

        dataset.add_rv_from_state_weights(
            y,
            np.array([
                [0.6, 0.3, 0.1],
                [0.0, 0.0, 0.0],  # weight 0
                [0.0, 1.0, 0.0],
                [3.0, 4.0, 3.0],  # weight 10
                [0.8, 0.1, 0.1],
            ]),
        )
        self.assertEqual(len(dataset.rvs), 2)
        self.assertIs(dataset.rvs[0], x)
        self.assertIs(dataset.rvs[1], y)
        self.assertArrayAlmostEqual(dataset.weights, [1, 0, 1, 10, 0.5])

        rv_data = dataset.state_idxs(y)
        self.assertArrayEqual(rv_data, [0, 0, 1, 1, 0])

    def test_add_rv_from_state_weights_unnormalised_unadjusted(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        dataset = HardDataset(length=5)
        self.assertArrayEqual(dataset.weights, [1, 1, 1, 1, 1])

        dataset.add_rv_from_state_weights(
            x,
            np.array([
                [0.6, 0.4],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.3, 0.7],
                [0.4, 0.1],  # weight 0.5
            ]),
            adjust_instance_weights=False,
        )
        self.assertEqual(len(dataset.rvs), 1)
        self.assertIs(dataset.rvs[0], x)

        rv_data = dataset.state_idxs(x)
        self.assertArrayEqual(rv_data, [0, 0, 1, 1, 0])
        self.assertArrayEqual(dataset.weights, [1, 1, 1, 1, 1])

        dataset.add_rv_from_state_weights(
            y,
            np.array([
                [0.6, 0.3, 0.1],
                [0.0, 0.0, 0.0],  # weight 0
                [0.0, 1.0, 0.0],
                [3.0, 4.0, 3.0],  # weight 10
                [0.8, 0.1, 0.1],
            ]),
            adjust_instance_weights=False,
        )
        self.assertEqual(len(dataset.rvs), 2)
        self.assertIs(dataset.rvs[0], x)
        self.assertIs(dataset.rvs[1], y)
        self.assertArrayEqual(dataset.weights, [1, 1, 1, 1, 1])

        rv_data = dataset.state_idxs(y)
        self.assertArrayEqual(rv_data, [0, 0, 1, 1, 0])

    def test_from_soft_dataset(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        x_state_weights = np.array([
            [0.6, 0.4],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.3, 0.7],
            [0.9, 0.1],
        ])
        y_state_weights = np.array([
            [0.6, 0.3, 0.1],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.3, 0.4, 0.3],
            [0.8, 0.1, 0.1],
        ])
        soft_dataset = SoftDataset([(x, x_state_weights), (y, y_state_weights)])

        dataset = HardDataset.from_soft_dataset(soft_dataset)

        self.assertEqual(len(dataset.rvs), 2)
        self.assertIs(dataset.rvs[0], x)
        self.assertIs(dataset.rvs[1], y)

        self.assertArrayAlmostEqual(dataset.weights, [1, 1, 1, 1, 1])

        rv_data = dataset.state_idxs(x)
        self.assertArrayEqual(rv_data, [0, 0, 1, 1, 0])

        rv_data = dataset.state_idxs(y)
        self.assertArrayEqual(rv_data, [0, 2, 1, 1, 0])

    def test_from_soft_dataset_unnormalised(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        x_state_weights = np.array([
            [0.6, 0.4],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.3, 0.7],
            [0.4, 0.1],  # weight 0.5
        ])
        y_state_weights = np.array([
            [0.6, 0.3, 0.1],
            [0.0, 0.0, 0.0],  # weight 0
            [0.0, 1.0, 0.0],
            [3.0, 4.0, 3.0],  # weight 10
            [0.8, 0.1, 0.1],
        ])
        soft_dataset = SoftDataset([(x, x_state_weights), (y, y_state_weights)])

        dataset = HardDataset.from_soft_dataset(soft_dataset)

        self.assertEqual(len(dataset.rvs), 2)
        self.assertIs(dataset.rvs[0], x)
        self.assertIs(dataset.rvs[1], y)

        self.assertArrayAlmostEqual(dataset.weights, [1, 0, 1, 10, 0.5])

        rv_data = dataset.state_idxs(x)
        self.assertArrayEqual(rv_data, [0, 0, 1, 1, 0])

        rv_data = dataset.state_idxs(y)
        self.assertArrayEqual(rv_data, [0, 0, 1, 1, 0])

    def test_bad_weights(self):
        with self.assertRaises(ValueError):
            # Not enough weights
            HardDataset(weights=np.array([1, 2, 3, 4, 5]), length=10)

    def test_duplicated_rv(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        _ = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        dataset = HardDataset([(x, [0, 0, 1, 1, 0])])

        with self.assertRaises(ValueError):
            dataset.add_rv_from_state_idxs(x, [0, 0, 1, 1, 0])  # x already added

    def test_bad_state_idxs_shape(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        dataset = HardDataset([(x, [0, 0, 1, 1, 0])])

        with self.assertRaises(ValueError):
            dataset.add_rv_from_state_idxs(y, np.array([0, 2, 1, 1, 0, 0]))  # Extra data point

    def test_bad_state_weights_shape(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        dataset = HardDataset([(x, [0, 0, 1, 1, 0])])

        with self.assertRaises(ValueError):
            dataset.add_rv_from_state_weights(y, np.array([0, 2, 1, 1, 0]))  # uses states instead of weights

    def test_bad_data_length(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        dataset = HardDataset([(x, [0, 0, 1, 1, 0])])

        with self.assertRaises(ValueError):
            dataset.add_rv_from_state_idxs(y, [0, 2, 1, 1, 0, 0])  # Extra data point


class TestSoftDataset(Fixture):

    def test_empty(self):
        dataset = SoftDataset()
        self.assertEqual(len(dataset), 0)
        self.assertEqual(len(dataset.rvs), 0)
        self.assertEqual(len(dataset.weights), 0)

    def test_with_length(self):
        dataset = SoftDataset(length=5)

        self.assertEqual(len(dataset), 5)
        self.assertEqual(len(dataset.rvs), 0)
        self.assertEqual(len(dataset.weights), 5)
        self.assertEqual(dataset.weights[0], 1)
        self.assertEqual(dataset.weights[1], 1)
        self.assertEqual(dataset.weights[2], 1)
        self.assertEqual(dataset.weights[3], 1)
        self.assertEqual(dataset.weights[4], 1)

    def test_with_weights(self):
        dataset = SoftDataset(weights=np.array([1, 2, 3, 4, 5]))

        self.assertEqual(len(dataset), 5)
        self.assertEqual(len(dataset.rvs), 0)
        self.assertEqual(len(dataset.weights), 5)
        self.assertEqual(dataset.weights[0], 1)
        self.assertEqual(dataset.weights[1], 2)
        self.assertEqual(dataset.weights[2], 3)
        self.assertEqual(dataset.weights[3], 4)
        self.assertEqual(dataset.weights[4], 5)

    def test_total_weight(self):
        dataset = SoftDataset(weights=np.array([1, 2, 3, 4, 5]))
        self.assertEqual(dataset.total_weight(), 15)

    def test_add_rv(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        dataset = SoftDataset(length=5)

        self.assertEqual(dataset.rvs, ())

        rv_data = dataset.add_rv(x)

        self.assertEqual(dataset.rvs, (x,))
        self.assertEqual(rv_data.shape, (5, len(x)))
        self.assertTrue(np.all(rv_data == 0))

        rv_data = dataset.add_rv(y)

        self.assertEqual(dataset.rvs, (x, y))
        self.assertEqual(rv_data.shape, (5, len(y)))
        self.assertTrue(np.all(rv_data == 0))

    def test_construct_from_ndarray(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))
        z = pgm.new_rv('z', ('1', '2'))

        x_data: NDArray = np.array([[0.6, 0.4], [1.0, 0.0]])
        y_data: NDArray = np.array([[0.6, 0.3, 0.1], [0.0, 0.0, 1.0]])
        z_data: NDArray = np.array([[0.6, 0.4], [1.0, 0.0]])
        dataset = SoftDataset([
            (x, x_data),
            (y, y_data),
            (z, z_data),
        ])

        self.assertEqual(dataset.rvs, (x, y, z))

        self.assertIs(dataset.state_weights(x), x_data)
        self.assertIs(dataset.state_weights(y), y_data)
        self.assertIs(dataset.state_weights(z), z_data)

    def test_construct_from_list(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))
        z = pgm.new_rv('z', ('1', '2'))

        dataset = SoftDataset([
            (x, [[0.6, 0.4], [1.0, 0.0]]),
            (y, [[0.6, 0.3, 0.1], [0.0, 0.0, 1.0]]),
            (z, [[0.6, 0.4], [1.0, 0.0]]),
        ])

        self.assertEqual(dataset.rvs, (x, y, z))

        self.assertNDArrayEqual(dataset.state_weights(x), np.array([[0.6, 0.4], [1.0, 0.0]]))
        self.assertNDArrayEqual(dataset.state_weights(y), np.array([[0.6, 0.3, 0.1], [0.0, 0.0, 1.0]]))
        self.assertNDArrayEqual(dataset.state_weights(z), np.array([[0.6, 0.4], [1.0, 0.0]]))

    def test_remove_rv(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))
        z = pgm.new_rv('z', ('1', '2'))

        dataset = SoftDataset([
            (x, np.array([[0.6, 0.4], [1.0, 0.0]])),
            (y, np.array([[0.6, 0.3, 0.1], [0.0, 0.0, 1.0]])),
            (z, np.array([[0.6, 0.4], [1.0, 0.0]])),
        ])

        self.assertEqual(dataset.rvs, (x, y, z))

        dataset.remove_rv(y)

        self.assertEqual(dataset.rvs, (x, z))
        self.assertNDArrayEqual(dataset.state_weights(x), np.array([[0.6, 0.4], [1.0, 0.0]]))
        self.assertNDArrayEqual(dataset.state_weights(z), np.array([[0.6, 0.4], [1.0, 0.0]]))

    def test_add_rv_from_state_idxs(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        dataset = SoftDataset(length=5)

        dataset.add_rv_from_state_idxs(x, [0, 0, 1, 1, 0])
        self.assertEqual(len(dataset.rvs), 1)
        self.assertIs(dataset.rvs[0], x)

        rv_data = dataset.state_weights(x)
        self.assertArrayEqual(
            rv_data.tolist(),
            [
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
                [1, 0],
            ]
        )

        dataset.add_rv_from_state_idxs(y, [0, 2, 1, 1, 0])
        self.assertEqual(len(dataset.rvs), 2)
        self.assertIs(dataset.rvs[0], x)
        self.assertIs(dataset.rvs[1], y)

        rv_data = dataset.state_weights(y)
        self.assertArrayEqual(
            rv_data.tolist(),
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 0],
                [1, 0, 0],
            ]
        )

    def test_add_rv_from_states(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        dataset = SoftDataset(length=5)

        dataset.add_rv_from_states(x, [True, True, False, False, True])
        self.assertEqual(len(dataset.rvs), 1)
        self.assertIs(dataset.rvs[0], x)

        rv_data = dataset.state_weights(x)
        self.assertArrayEqual(
            rv_data.tolist(),
            [
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
                [1, 0],
            ]
        )

        dataset.add_rv_from_states(y, ['yes', 'maybe', 'no', 'no', 'yes'])
        self.assertEqual(len(dataset.rvs), 2)
        self.assertIs(dataset.rvs[0], x)
        self.assertIs(dataset.rvs[1], y)

        rv_data = dataset.state_weights(y)
        self.assertArrayEqual(
            rv_data.tolist(),
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 0],
                [1, 0, 0],
            ]
        )

    def test_add_rv_from_state_weights(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        dataset = SoftDataset(length=5)

        dataset.add_rv_from_state_weights(
            x,
            np.array([
                [0.6, 0.4],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.3, 0.7],
                [0.9, 0.1],
            ]),
        )
        self.assertEqual(len(dataset.rvs), 1)
        self.assertIs(dataset.rvs[0], x)

        rv_data = dataset.state_weights(x)
        self.assertArrayEqual(
            rv_data.tolist(),
            [
                [0.6, 0.4],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.3, 0.7],
                [0.9, 0.1],
            ]
        )

        dataset.add_rv_from_state_weights(
            y,
            np.array([
                [0.6, 0.3, 0.1],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.3, 0.4, 0.3],
                [0.8, 0.1, 0.1],
            ]),
        )
        self.assertEqual(len(dataset.rvs), 2)
        self.assertIs(dataset.rvs[0], x)
        self.assertIs(dataset.rvs[1], y)

        rv_data = dataset.state_weights(y)
        self.assertArrayEqual(
            rv_data.tolist(),
            [
                [0.6, 0.3, 0.1],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.3, 0.4, 0.3],
                [0.8, 0.1, 0.1],
            ]
        )

    def test_normalise(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        x_weights = np.array([1, 2, 1, 0.1, 1.0])
        y_weights = np.array([1, 1, 2, 1.0, 0.1])

        x_raw_data = np.array([
            [0.6, 0.4],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.3, 0.7],
            [0.9, 0.1],
        ])
        y_raw_data = np.array([
            [0.6, 0.3, 0.1],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.3, 0.4, 0.3],
            [0.8, 0.1, 0.1],
        ])

        x_data = x_raw_data.copy()
        for row, weight in zip(x_data, x_weights):
            row *= weight
        y_data = y_raw_data.copy()
        for row, weight in zip(y_data, y_weights):
            row *= weight

        dataset = SoftDataset([
            (x, x_data),
            (y, y_data),
        ])

        expected_original_weights = np.ones(len(dataset))
        self.assertNDArrayAlmostEqual(dataset.weights, expected_original_weights)
        self.assertNDArrayAlmostEqual(dataset.state_weights(x), x_data)
        self.assertNDArrayAlmostEqual(dataset.state_weights(y), y_data)

        dataset.normalise()

        expected_normalised_weights = x_weights * y_weights
        self.assertNDArrayAlmostEqual(dataset.weights, expected_normalised_weights)
        self.assertNDArrayAlmostEqual(dataset.state_weights(x), x_raw_data)
        self.assertNDArrayAlmostEqual(dataset.state_weights(y), y_raw_data)

    def test_normalise_with_zeros(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        x_weights = np.array([1, 2, 1, 0.0, 1.0])
        y_weights = np.array([1, 1, 2, 1.0, 0.0])

        x_raw_data = np.array([
            [0.6, 0.4],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.3, 0.7],
            [0.9, 0.1],
        ])
        y_raw_data = np.array([
            [0.6, 0.3, 0.1],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.3, 0.4, 0.3],
            [0.8, 0.1, 0.1],
        ])

        x_data = x_raw_data.copy()
        for row, weight in zip(x_data, x_weights):
            row *= weight
        y_data = y_raw_data.copy()
        for row, weight in zip(y_data, y_weights):
            row *= weight

        dataset = SoftDataset([
            (x, x_data),
            (y, y_data),
        ])

        expected_original_weights = np.ones(len(dataset))

        self.assertNDArrayAlmostEqual(dataset.weights, expected_original_weights)
        self.assertNDArrayAlmostEqual(dataset.state_weights(x), x_data)
        self.assertNDArrayAlmostEqual(dataset.state_weights(y), y_data)

        dataset.normalise()

        x_normalised_data = np.array([
            [0.6, 0.4],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ])
        y_normalised_data = np.array([
            [0.6, 0.3, 0.1],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        normalised_weights = x_weights * y_weights

        self.assertNDArrayAlmostEqual(dataset.weights, normalised_weights)
        self.assertNDArrayAlmostEqual(dataset.state_weights(x), x_normalised_data)
        self.assertNDArrayAlmostEqual(dataset.state_weights(y), y_normalised_data)

    def test_from_hard_dataset(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        hard_dataset = HardDataset([(x, np.array([0, 0, 1, 1, 0]))])

        dataset = SoftDataset.from_hard_dataset(hard_dataset)

        self.assertEqual(len(dataset.rvs), 1)
        self.assertIs(dataset.rvs[0], x)

        rv_data = dataset.state_weights(x)
        self.assertArrayEqual(
            rv_data.tolist(),
            [
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
                [1, 0],
            ]
        )

        dataset.add_rv_from_states(y, ['yes', 'maybe', 'no', 'no', 'yes'])
        self.assertEqual(len(dataset.rvs), 2)
        self.assertIs(dataset.rvs[0], x)
        self.assertIs(dataset.rvs[1], y)

        rv_data = dataset.state_weights(y)
        self.assertArrayEqual(
            rv_data.tolist(),
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 0],
                [1, 0, 0],
            ]
        )

    def test_bad_weights(self):
        with self.assertRaises(ValueError):
            # Not enough weights
            SoftDataset(weights=np.array([1, 2, 3, 4, 5]), length=10)

    def test_duplicated_rv(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))

        x_weights: NDArray = np.array([
            [0.6, 0.4],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.3, 0.7],
            [0.9, 0.1],
        ])

        dataset = SoftDataset([(x, x_weights)])

        with self.assertRaises(ValueError):
            dataset.add_rv_from_state_weights(x, x_weights)

    def test_bad_data_shape(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        x_data: NDArray = np.array([
            [0.6, 0.4],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.3, 0.7],
            [0.9, 0.1],
        ])
        y_data: NDArray = np.array([
            [0.6, 0.3, 0.1],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.3, 0.4, 0.3],
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],  # Extra instance
        ])

        dataset = SoftDataset([(x, x_data)])

        with self.assertRaises(ValueError):
            dataset.add_rv_from_state_weights(y, y_data)  # Extra instance


if __name__ == '__main__':
    test_main()
