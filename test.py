#!/usr/bin/python

from pathos.multiprocessing import ProcessPool as Pool
from math import sqrt
import dill

def fart():

    cords = [1., 2., 3., 4., 5., 6.]
    result = Pool(4).amap(sqrt, cords)
    return result