"""
Unit test Fixture
"""
from importlib import import_module as _import
from typing import Iterable, Optional
from unittest import \
    TestCase, \
    main as _unittest_main, \
    defaultTestLoader as _testLoader, \
    TestSuite as _TestSuite


def test_main():
    """
    Execute unittest.main().
    """
    return _unittest_main()


def make_suit(test_modules: Iterable[str], package: Optional[str] = None):
    # noinspection GrazieInspection
    """
        Construct a `unittest.TestSuite` object containing all the unit test in the named test modules.

        Args:
            test_modules: a collection of strings, each naming a test module.
            package: optional argument for prefixing each test module.

        Returns:
            a `unittest.TestSuite` object.
        """
    suite = _TestSuite()
    for t in test_modules:
        test_module = t if package is None else f'{package}.{t}'
        try:
            # if the module defines a suite() function, call it to get the suite.
            module_t = _import(test_module)
            suite_t = getattr(module_t, 'suite')
            suite.addTest(suite_t())
        except (ImportError, AttributeError):
            # load all the test cases from the module
            suite.addTest(_testLoader.loadTestsFromName(t))
    return suite


class Fixture(TestCase):

    def assertEmpty(self, got, msg=None):
        self.assertEqual(len(got), 0, msg=msg)

    def assertArrayEqual(self, got, expect, msg=None):
        self.assertEqual(len(expect), len(got))
        for idx in range(len(expect)):
            expect_i = expect[idx]
            got_i = got[idx]
            self.assertEqual(
                expect_i,
                got_i,
                msg=_make_msg(msg, "at index ", idx, ": expected ", expect, ", got: ", got)
            )

    def assertArrayAlmostEqual(self, got, expect, places=None, delta=None, msg=None):
        self.assertEqual(len(expect), len(got))
        for idx in range(len(expect)):
            expect_i = expect[idx]
            got_i = got[idx]
            self.assertAlmostEqual(
                got_i,
                expect_i,
                places=places,
                delta=delta,
                msg=_make_msg(msg, "at index ", idx, ": expected ", expect, ", got: ", got)
            )

    def assertArraySetEqual(self, got, expect, msg=None):
        len_expect = len(expect)
        self.assertEqual(len(expect), len(got))
        expect = set(expect)
        got = set(got)
        self.assertEqual(len(expect), len_expect)
        self.assertEqual(len(expect), len(got))
        for elem in expect:
            self.assertIn(
                elem,
                got,
                msg=_make_msg(msg, "expected ", expect, ", got: ", got)
            )

    def assertIterFinished(self, it, msg=None):
        """
        Args:
            it: an iterator which is expected to throw StopIteration
                when next(it) is called.
            msg: an optional message to pass to `assertRaises`.
        """
        with self.assertRaises(StopIteration, msg=msg):
            next(it)


def _make_msg(orig_msg, *parts):
    msg = ''.join(str(part) for part in parts)
    if orig_msg is not None:
        msg += ", "
        msg += str(orig_msg)
    return msg
