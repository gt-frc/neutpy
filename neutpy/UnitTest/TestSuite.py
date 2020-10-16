#!/usr/bin/python

import unittest
import neutpy

class NeutpyUnitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.npi = neutpy.neutrals()
        cls.npi.set_cpu_cores(6)


class NeutpySingleLowerNullTest(NeutpyUnitTest):
    """
    Tests that neutpy can run on a single lower null shot.
    """
    @classmethod
    def setUpClass(cls):
        super(NeutpySingleLowerNullTest, cls).setUpClass()
        cls.npi.from_file("144977_3000/toneutpy.conf")


    def test_cpu_override(self):
        self.assertIs(self.npi.cpu_cores, 6)

    def test_neutpy_has_nn(self):
        self.assertTrue(hasattr(self.npi.nn, "s"))
        self.assertTrue(hasattr(self.npi.nn, "t"))
        self.assertTrue(hasattr(self.npi.nn, "tot"))

    def test_neutpy_has_izn(self):
        self.assertTrue(hasattr(self.npi.izn_rate, "s"))
        self.assertTrue(hasattr(self.npi.izn_rate, "t"))
        self.assertTrue(hasattr(self.npi.izn_rate, "tot"))



class NeutpyDoubleNullTest(NeutpyUnitTest):
    """
    Tests that neutpy can run on a double null shot.
    """

    @classmethod
    def setUpClass(cls):
        super(NeutpyDoubleNullTest, cls).setUpClass()
        cls.npi.from_file("175826_2010/toneutpy.conf")
    def setUp(self):
        pass

    def test_neutpy_has_nn(self):
        self.assertTrue(hasattr(self.npi.nn, "s"))
        self.assertTrue(hasattr(self.npi.nn, "t"))
        self.assertTrue(hasattr(self.npi.nn, "tot"))

    def test_neutpy_has_izn(self):
        self.assertTrue(hasattr(self.npi.izn_rate, "s"))
        self.assertTrue(hasattr(self.npi.izn_rate, "t"))
        self.assertTrue(hasattr(self.npi.izn_rate, "tot"))

class NeutpyNegativeTriangularityDoubleNullTest(NeutpyUnitTest):
    """
    Tests that neutpy can run on a double null, negative triangularity shot.
    """
    @classmethod
    def setUpClass(cls):
        super(NeutpyNegativeTriangularityDoubleNullTest, cls).setUpClass()
        cls.npi.from_file("170672_1900/toneutpy.conf")

    def test_neutpy_has_nn(self):
        self.assertTrue(hasattr(self.npi.nn, "s"))
        self.assertTrue(hasattr(self.npi.nn, "t"))
        self.assertTrue(hasattr(self.npi.nn, "tot"))

    def test_neutpy_has_izn(self):
        self.assertTrue(hasattr(self.npi.izn_rate, "s"))
        self.assertTrue(hasattr(self.npi.izn_rate, "t"))
        self.assertTrue(hasattr(self.npi.izn_rate, "tot"))

class NeutpyFromGT3Test(NeutpyUnitTest):
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

    def test_neutpy_has_nn(self):
        self.assertTrue(hasattr(self.npi.nn, "s"))
        self.assertTrue(hasattr(self.npi.nn, "t"))
        self.assertTrue(hasattr(self.npi.nn, "tot"))

    def test_neutpy_has_izn(self):
        self.assertTrue(hasattr(self.npi.izn_rate, "s"))
        self.assertTrue(hasattr(self.npi.izn_rate, "t"))
        self.assertTrue(hasattr(self.npi.izn_rate, "tot"))


if __name__ == '__main__':
    unittest.main()
