# !/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Neutpy calculates neutral densities, ionization rates, and related quantities in tokamaks.

The neutpy module contains three classes: neutpy, read_infile, and neutpyplot.

"""

from __future__ import division
import os
        
class read_infile():
    def __init__(self, infile):
        print ('READING NEUTPY INPUT FILE')
        
        # some regex commands we'll use when reading stuff in from the input file
        r0di = "r'.*%s *= *([ , \d]*) *'%(v)"
        r0df = "r'.*%s *= *([+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?) *'%(v)"
        r0ds = "r'.*%s *= *(\w+) *'%(v)"
        r1di = "r'.*%s\( *(\d*) *\) *= *(\d*) *'%(v)"
        r1df = "r'.*%s\( *(\d*)\) *= *([+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?) *'%(v)"
        # r2df = "r'.*%s\( *(\d*)\) *= *([+\-]*[ , \.\d]*) *'%(v)"
        # r2df = "r'.*%s\( *(\d*)\) *= *((?:[+\-]?\d*\.?\d*(?:[eE]?[+\-]?\d+)?, ?)+) *'%(v)"
        r2df = "r'.*%s\( *(\d*)\) *= *((?:[+\-]?\d*\.?\d*(?:[eE]?[+\-]?\d+), ? ?)+) *'%(v)"
        
        v0d = {}
        v0d["nCells"] = ["int", r0di]
        v0d["nPlasmReg"] = ["int", r0di]
        v0d["nWallSegm"] = ["int", r0di]
        v0d["aneut"] = ["int", r0di]
        v0d["zion"] = ["int", r0di]
        v0d["aion"] = ["int", r0di]
        v0d["tslow"] = ["float", r0df]
        v0d["int_method"] = ["str", r0ds]
        v0d["phi_int_pts"] = ["int", r0di]
        v0d["xi_int_pts"] = ["int", r0di]
        v0d["xsec_ioni"] = ["str", r0ds]
        v0d["xsec_ione"] = ["str", r0ds]
        v0d["xsec_cx"] = ["str", r0ds]
        v0d["xsec_rec"] = ["str", r0ds]
        v0d["xsec_el"] = ["str", r0ds]
        v0d["xsec_eln"] = ["str", r0ds]
        v0d["refmod_e"] = ["str", r0ds]
        v0d["refmod_n"] = ["str", r0ds]
        v0d["cell1_ctr_x"] = ["float", r0df]
        v0d["cell1_ctr_y"] = ["float", r0df]
        v0d["cell1_theta0"] = ["float", r0df]

        # initialize 0d varialbes
        for v in v0d:
            exec('self.%s = 0' %(v))

        # populate 0d variables. Need to do this first to calculate the sizes of the other arrays
        with open(os.getcwd() + '/' + infile, 'r') as toneut:
            for count, line in enumerate(toneut):
                if not line.startswith("# "):
                    # read in 0d variables
                    for v in v0d:
                        exec("result = re.match(%s, line)"%(v0d[v][1]))
                        if result:
                            exec("self.%s = %s(result.group(1))"%(v, v0d[v][0]))

        self.nCells_tot = self.nCells + self.nPlasmReg + self.nWallSegm

        # now we can do the same thing for the 1d and 2d arrays
        v1d = {}
        v1d["iType"] = [self.nCells_tot, "int", r1di]
        v1d["nSides"] = [self.nCells_tot, "int", r1di]
        v1d["zwall"] = [self.nCells_tot, "int", r1di]
        v1d["awall"] = [self.nCells_tot, "int", r1di]
        v1d["elecTemp"] = [self.nCells_tot, "float", r1df]
        v1d["ionTemp"] = [self.nCells_tot, "float", r1df]
        v1d["elecDens"] = [self.nCells_tot, "float", r1df]
        v1d["ionDens"] = [self.nCells_tot, "float", r1df]
        v1d["twall"] = [self.nCells_tot, "float", r1df]
        v1d["f_abs"] = [self.nCells_tot, "float", r1df]
        v1d["alb_s"] = [self.nCells_tot, "float", r1df]
        v1d["alb_t"] = [self.nCells_tot, "float", r1df]
        v1d["s_ext"] = [self.nCells_tot, "float", r1df]
        
        v2d = {}        
        v2d["adjCell"] = [self.nCells_tot, 4, "int", r2df]
        v2d["lsides"] = [self.nCells_tot, 4, "float", r2df]
        v2d["angles"] = [self.nCells_tot, 4, "float", r2df]
        
        # initialize elements and arrays
        for v in v1d:
            exec('self.%s = np.zeros(%s, dtype=%s)' %(v, v1d[v][0], v1d[v][1]))
        for v in v2d:
            exec('self.%s = np.zeros((%s, %s), dtype=%s)' %(v, v2d[v][0], v2d[v][1], v2d[v][2]))
            # fill with -1. elements that aren't replaced with a non-negative number correspond to
            # a side that doesn't exist. Several other more elegant approaches were tried, including 
            # masked arrays, and they all resulted in significantly hurt performance. 
            exec('self.%s[:] = -1' %(v))

        # populate arrays
        with open(os.getcwd() + '/' + infile, 'r') as toneut:
            for count, line in enumerate(toneut):
                if not line.startswith("# "):                   

                    # read in 1d arrays
                    for v in v1d:
                        exec("result = re.match(%s, line)"%(v1d[v][2]))
                        if result:
                            exec("self.%s[int(result.group(1))] = result.group(2)"%(v))

                    # read in 2d arrays
                    for v in v2d:
                        exec("result = re.match(%s, line)"%(v2d[v][3]))
                        if result:
                            # read new vals into an array
                            exec("newvals = np.asarray(result.group(2).split(', '), dtype=%s)"%(v2d[v][2]))
                            # pad it to make it the correct size to include in the array
                            exec("self.%s[int(result.group(1)), :] = np.pad(newvals, (0, %s), mode='constant', constant_values=0)"%(v, 4-newvals.shape[0]))
                            # finally, mask the array elements that don't correspond to anything to prevent 
                            # problems later (and there will be problems later if we don't.)
                            # fill with -1. elements that aren't replaced with a non-negative number correspond to
                            # a side that doesn't exist. Several other more elegant approaches were tried, including 
                            # masked arrays, and they all resulted in significantly hurt performance. 
                            exec("self.%s[int(result.group(1)), %s:] = -1"%(v, newvals.shape[0]))
        return        


if __name__ == "__main__":
    # CREATE NEUTPY INSTANCE (READS INPUT FILE, INSTANTIATES SOME FUNCTIONS)
    # (OPTIONALLY PASS THE NAME OF AN INPUT FILE. IF NO ARGUMENT PASSED, IT LOOKS
    # FOR A FILE CALLED 'neutpy_in'.)

    # neut = neutpy('toneut_converted_mod')
    neut = neutpy('neutpy_in_generated2')
