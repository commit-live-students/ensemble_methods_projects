from unittest import TestCase
from inspect import getfullargspec
from ..build import bagging


class TestBagging(TestCase):
    def test_bagging(self):

        # Input parameters tests
        args = getfullargspec(bagging).args
        args_default = getfullargspec(bagging).defaults
        self.assertEqual(len(args), 5, "Expected arguments %d, Given %d" % (5, len(args)))
        self.assertEqual(args_default, None, "Expected default values do not match given default values")

        # Return data types
        # Nothing to check here

        # Return value tests
        # Nothing to check here
