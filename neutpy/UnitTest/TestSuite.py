#!/usr/bin/python

import unittest
import neutpy

class NeutpyUnitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.npi = neutpy.neutrals()
        cls.npi.set_cpu_cores(8)

class CommonMethods(object):

    def test_neutpy_has_nn(cls):
        cls.assertTrue(hasattr(cls.npi.nn, "s"))
        cls.assertTrue(hasattr(cls.npi.nn, "t"))
        cls.assertTrue(hasattr(cls.npi.nn, "tot"))

    def test_neutpy_has_izn(cls):
        cls.assertTrue(hasattr(cls.npi.izn_rate, "s"))
        cls.assertTrue(hasattr(cls.npi.izn_rate, "t"))
        cls.assertTrue(hasattr(cls.npi.izn_rate, "tot"))



class NeutpySingleLowerNullTest(NeutpyUnitTest, CommonMethods):
    """
    Tests that neutpy can run on a single lower null shot.
    """
    @classmethod
    def setUpClass(cls):
        super(NeutpySingleLowerNullTest, cls).setUpClass()
        cls.npi.from_file("144977_3000/toneutpy.conf")


    def test_cpu_override(self):
        self.assertIs(self.npi.cpu_cores, 6)

class NeutpyDoubleNullTest(NeutpyUnitTest, CommonMethods):
    """
    Tests that neutpy can run on a double null shot.
    """

    @classmethod
    def setUpClass(cls):
        super(NeutpyDoubleNullTest, cls).setUpClass()
        cls.npi.from_file("175826_2010/toneutpy.conf")
    def setUp(self):
        pass


class NeutpyNegativeTriangularityDoubleNullTest(NeutpyUnitTest, CommonMethods):
    """
    Tests that neutpy can run on a double null, negative triangularity shot.
    """
    @classmethod
    def setUpClass(cls):
        super(NeutpyNegativeTriangularityDoubleNullTest, cls).setUpClass()
        cls.npi.from_file("170672_1900/toneutpy.conf")

class NeutpyFromGT3Test(NeutpyUnitTest, CommonMethods):
    """
    Tests that NeutPy can run from a GT3 instance
    """
    npi = neutpy.neutrals

    @classmethod
    def setUpClass(cls):
        super(NeutpyFromGT3Test, cls).setUpClass()
        from GT3.TestBase import getGT3Test
        plasma = getGT3Test()
        cls.npi.from_gt3(plasma.core, plasma.inp)


if __name__ == '__main__':
    unittest.main()
