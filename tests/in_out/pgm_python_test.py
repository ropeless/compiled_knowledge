import io

from ck.in_out.pgm_python import write_python, read_python
from ck.pgm import PGM
from tests.helpers.pgm_test_cases import PGMTestCases
from ck.utils.tmp_dir import tmp_dir
from tests.helpers.unittest_fixture import test_main
from tests.pgm.pgm_test import PGMFixture


class TestPGMPython(PGMTestCases, PGMFixture):

    def check_pgm(self, pgm: PGM) -> None:
        with tmp_dir():
            filename = 'pgm_round_trip_test.py'
            with open(filename, 'w', encoding='utf8') as out:
                write_python(pgm, file=out)
            loaded: PGM = read_python(filename)

        self.assertTrue(isinstance(loaded, PGM))
        self.assertEquivalentName(pgm, loaded)  # Name may change
        self.assertPGMsStructureEqual(pgm, loaded)
        self.assertPGMsFunctionallySame(pgm, loaded, places=6)


class TestPGMPythonWithArguments(PGMTestCases, PGMFixture):

    def check_pgm(self, pgm: PGM) -> None:
        with tmp_dir():
            filename = 'pgm_round_trip_test.py'
            with open(filename, 'w', encoding='utf8') as out:
                write_python(pgm, file=out, pgm_name='my_bad_pgm', package_name='some_pgm')
            loaded: PGM = read_python(filename)

        self.assertTrue(isinstance(loaded, PGM))
        self.assertEquivalentName(pgm, loaded)
        self.assertPGMsStructureEqual(pgm, loaded)
        self.assertPGMsFunctionallySame(pgm, loaded, places=6)


class TestPGMPythonStream(PGMTestCases, PGMFixture):

    def check_pgm(self, pgm: PGM) -> None:
        out = io.StringIO()
        write_python(pgm, file=out)

        # read it back in (via a temp file)
        with tmp_dir():
            filename = 'tmp.py'
            with open(filename, 'w', encoding='utf8') as f:
                f.write(out.getvalue())
            loaded: PGM = read_python(filename)

        self.assertTrue(isinstance(loaded, PGM))
        self.assertEquivalentName(pgm, loaded)
        self.assertPGMsStructureEqual(pgm, loaded)
        self.assertPGMsFunctionallySame(pgm, loaded, places=6)


if __name__ == '__main__':
    test_main()
