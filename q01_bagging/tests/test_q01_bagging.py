from unittest import TestCase
from inspect import getfullargspec
from ..build import bagging


class TestBagging(TestCase):
    def test_bagging(self):  # Input parameters tests
        args = getfullargspec(bagging)
        self.assertEqual(len(args[0]), 5, "Expected arguments %d, Given %d" % (5, len(args[0])))

    def test_bagging_default(self):  # Input parameter default
        args = getfullargspec(bagging)
        self.assertEqual(args[3], None, "Expected default values do not match given default values")

    # Return data types
    # Nothing to check here

    # Return value tests
    # Nothing to check here
