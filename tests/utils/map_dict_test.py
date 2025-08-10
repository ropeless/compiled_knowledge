
from tests.helpers.unittest_fixture import Fixture, test_main
from ck.utils.map_dict import MapDict


class TestMapDict(Fixture):

    def test_empty(self):
        map_dict = MapDict()
        self.assertEqual(0, len(map_dict))
        self.assertArrayEqual([], map_dict.keys())
        self.assertArrayEqual([], map_dict.values())
        self.assertArrayEqual([], map_dict.items())
        self.assertIterFinished(iter(map_dict))
        self.assertIsNone(map_dict.get('some_random_key'))

        # note that this actually creates an entry
        self.assertDictEqual(dict(), map_dict.get_dict('some_random_key'))

    def test_get_dict(self):
        map_dict = MapDict()
        key = 'some_random_key'

        self.assertEmpty(map_dict)

        keyed_dict = map_dict.get_dict(key)

        self.assertNotEmpty(map_dict)
        self.assertEmpty(keyed_dict)

        keyed_dict['sub-key'] = 'some-value'

        second_dict = map_dict.get_dict(key)
        self.assertDictEqual({'sub-key': 'some-value'}, second_dict)


if __name__ == '__main__':
    test_main()
