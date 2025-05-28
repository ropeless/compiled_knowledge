from ck.in_out.pgm_pickle import write_pickle, read_pickle
from ck.pgm import PGM
from tests.helpers.pgm_test_cases import PGMTestCases
from ck.utils.tmp_dir import tmp_dir
from tests.helpers.unittest_fixture import test_main
from tests.pgm.pgm_test import PGMFixture


class TestPGMPickle(PGMTestCases, PGMFixture):

    def check_pgm(self, pgm: PGM) -> None:
        with tmp_dir():
            filename = 'pgm_round_trip_test.pkl'
            write_pickle(pgm, filename)
            loaded = read_pickle(filename)

        self.assertTrue(isinstance(loaded, PGM))
        self.assertEqual(pgm.name, loaded.name)  # Name does not change
        self.assertPGMsStructureEqual(pgm, loaded)
        self.assertPGMsFunctionallySame(pgm, loaded, places=6)


if __name__ == '__main__':
    test_main()
