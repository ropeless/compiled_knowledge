from tests.helpers.unittest_fixture import Fixture, test_main
from ck.utils.map_list import MapList


class TestMapList(Fixture):

    def test_empty(self):
        map_list = MapList()
        self.assertEqual(0, len(map_list))
        self.assertEmpty(map_list.keys())
        self.assertEmpty(map_list.values())
        self.assertEmpty(map_list.items())
        self.assertIterFinished(iter(map_list))
        self.assertIsNone(map_list.get('some_random_key'))

        # note that this actually creates an entry
        self.assertEqual([], map_list.get_list('some_random_key'))

    def test_get_list(self):
        map_list = MapList()
        key = 'some_random_key'

        keyed_list = map_list.get_list(key)
        self.assertEqual([], keyed_list)

        keyed_list.append('some-value')

        second_list = map_list.get_list(key)
        self.assertArrayEqual(['some-value'], second_list)

    def test_append(self):
        map_list = MapList()
        key = 'some_random_key'

        map_list.append(key, 'some-value')
        self.assertArrayEqual(['some-value'], map_list[key])


if __name__ == '__main__':
    test_main()
