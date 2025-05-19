from tests.helpers.unittest_fixture import Fixture, test_main
from ck.utils.map_set import MapSet


class TestMapSet(Fixture):

    def test_empty(self):
        map_set = MapSet()
        self.assertEqual(0, len(map_set))
        self.assertArrayEqual([], map_set.keys())
        self.assertArrayEqual([], map_set.values())
        self.assertArrayEqual([], map_set.items())
        self.assertIterFinished(iter(map_set))
        self.assertIsNone(map_set.get('some_random_key'))

        # note that this actually creates an entry
        self.assertSetEqual(set(), map_set.get_set('some_random_key'))

    def test_get_set(self):
        map_set = MapSet()
        key = 'some_random_key'

        keyed_set = map_set.get_set(key)
        self.assertEmpty(keyed_set)

        keyed_set.add('some-value')

        second_set = map_set.get_set(key)
        self.assertSetEqual({'some-value'}, second_set)

    def test_add(self):
        map_set = MapSet()
        key = 'some_random_key'

        map_set.add(key, 'some-value')
        self.assertSetEqual({'some-value'}, map_set[key])


if __name__ == '__main__':
    test_main()
