#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Neutpy calculates neutral densities, ionization rates, and related quantities in tokamaks.

The neutpy module contains three classes: neutpy, read_infile, and neutpyplot.

"""
from __future__ import division
import sys
import os
import re
from math import sqrt, pi, sin, tan, exp
import numpy as np
from scipy.interpolate import griddata, interp1d
from scipy import integrate
from scipy.constants import m_p
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from neutpy_xsec import calc_xsec

# instantiate cross sections
sv = calc_xsec()

class neutpy:
    
    def __init__(self, infile=None, inarrs=None):
        print 'BEGINNING NEUTPY'
        
        sys.dont_write_bytecode = True 
        if not os.path.exists(os.getcwd()+'/outputs'):
            os.makedirs(os.getcwd()+'/outputs')
        if not os.path.exists(os.getcwd()+'/figures'):
            os.makedirs(os.getcwd()+'/figures')
            
        # EITHER READ INPUT FILE OR ACCEPT ARRAYS FROM ANOTHER PROGRAM
        if infile is None and inarrs is None:
            # should look for default input file like neutpy_in.txt or something
            pass
        elif infile is not None and inarrs is None:
            # read input file
            inp = read_infile(infile)
            self.__dict__ = inp.__dict__.copy()
        elif infile is None and inarrs is not None:
            # accept arrays
            self.__dict__ = inarrs  # .__dict__.copy()
        elif infile is not None and inarrs is not None:
            print 'You\'ve specified both an input file and passed arrays directly.'
            print 'Please remove one of the inputs and try again.'
            sys.exit()
        
        print 'CALCULATING CELL QUANTITIES'
        self.calc_cell_quantities()
        print 'CALCULATING FACE QUANTITIES'
        self.calc_face_quantities()
        print 'CALCULATING TCOEFS'
        self.calc_tcoefs()
        print 'CREATING AND SOLVING MATRIX'
        self.solve_matrix()
        print 'CALCULATING NEUTRAL DENSITY'
        self.calc_neutral_dens()
        # print 'WRITING NEUTPY OUTPUTS'
        # self.write_outputs()
    
    # isclose is included in python3.5+, so you can delete this if the code
    # ever gets ported into python3.5+
    @staticmethod
    def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
        return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
    
    @staticmethod
    def midpoint2D(f, f_limx, f_limy, nx, ny, **kwargs):
        """calculates a double integral using the midpoint rule"""
        I = 0
        # start with outside (y) limits of integration
        c, d = f_limy(**kwargs)
        hy = (d - c)/float(ny)
        for j in range(ny):
            yj = c + hy/2 + j*hy
            # for each j, calculate inside limits of integration
            a, b = f_limx(yj, **kwargs)
            hx = (b - a)/float(nx)
            for i in range(nx):
                xi = a + hx/2 + i*hx
                I += hx*hy*f(xi, yj, **kwargs)
        return I
    
    @staticmethod
    def print_progress(iteration, total, prefix='', suffix='', decimals=0, bar_length=50):
        """creates a progress bar
        
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            bar_length  - Optional  : character length of bar (Int)
        """
        str_format = "{0:." + str(decimals) + "f}"
        percents = str_format.format(100 * ((iteration+1) / float(total)))
        filled_length = int(round(bar_length * (iteration+1) / float(total)))
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
        sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    
    @staticmethod
    def calc_e_reflect(e0, am1, am2, z1, z2):
        """Calculates the energy reflection coefficient
    
        Args:
            e0 (float): 
            am1 (str): 
            am2 (str): 
            z1 (str): 
            z2 (str):
    
        Returns:
            r_e (float): The energy reflection coefficient
        """
        e = 2.71828
        
        ae = np.array([[0.001445,  0.2058, 0.4222, 0.4484, 0.6831],
                       [404.7,     3.848,  3.092,  27.16,  27.16],
                       [73.73,     19.07,  13.17,  15.66,  15.66],
                       [0.6519,    0.4872, 0.5393, 0.6598, 0.6598],
                       [4.66,     15.13,  4.464,  7.967,  7.967],
                       [1.971,     1.638,  1.877,  1.822,  1.822]])
        
        mu = am2 / am1
        zfactr = 1.0 / (z1 * z2 * sqrt(z1**0.67 + z2**0.67))
        epsln = 32.55 * mu * zfactr * e0 / (1.+mu)
        
        if mu == 1:
            col = 0
        elif mu == 3:
            col = 1
        elif mu >= 6.0 and mu <= 7.0:
            col = 2
        elif mu >= 12.0 and mu <= 15.0:
            col = 3
        elif mu >= 20.0:
            col = 4
            
        r_e = ae[0, col] * np.log(ae[1, col]*epsln + e) / \
            (1 + ae[2, col]*epsln**ae[3, col] + ae[4, col]*epsln**ae[5, col])
        return r_e
        
    @staticmethod
    def calc_n_reflect(e0, am1, am2, z1, z2):
        """Calculates the particle reflection coefficient
    
        Args:
            e0 (float): 
            am1 (str): 
            am2 (str): 
            z1 (str): 
            z2 (str):
    
        Returns:
            r_n (float): The particle reflection coefficient
        """
        e = 2.71828        
        
        an = np.array([[ 0.02129,  0.36800,  0.51730,  0.61920,  0.82500],
                       [16.39000,  2.98500,  2.54900, 20.01000, 21.41000],
                       [26.39000,  7.12200,  5.32500,  8.92200,  8.60600],
                       [ 0.91310,  0.58020,  0.57190,  0.66690,  0.64250],
                       [ 6.24900,  4.21100,  1.09400,  1.86400,  1.90700],
                       [ 2.55000,  1.59700,  1.93300,  1.89900,  1.92700]])
        
        mu = am2 / am1
        zfactr = 1.0 / (z1 * z2 * sqrt(z1**0.67 + z2**0.67))
        epsln = 32.55 * mu * zfactr * e0 / (1.+mu)
    
        if mu == 1:
            col = 0
        elif mu == 3:
            col = 1
        elif mu >= 6.0 and mu <= 7.0:
            col=2
        elif mu >= 12.0 and mu <= 15.0:
            col=3
        elif mu>=20.0:
            col=4

        r_n = an[0, col]*np.log(an[1, col]*epsln + e) / \
            (1 + an[2, col]*epsln**an[3, col] + an[4, col]*epsln**an[5, col])
        return r_n

    def calc_cell_quantities(self):

        self.cell_area = np.zeros((self.nCells))
        self.cell_perim = np.zeros((self.nCells))

        for i in range(0, self.nCells):
            L_sides = self.lsides[i, :self.nSides[i]]
        
            angles = self.angles[i, :self.nSides[i]]*2*pi/360  # in radians
            theta = np.zeros(angles.shape)
            for j in range(0, int(self.nSides[i])):
                if j == 0:
                    theta[j] = 0.0
                elif j == 1:
                    theta[j] = angles[0]
                else:
                    theta[j] = theta[j-1]+angles[j-1] - pi
        
            x_comp = L_sides*np.cos(theta)
            y_comp = L_sides*np.sin(theta)
            
            x_comp[0] = 0
            y_comp[0] = 0
            
            xs = np.cumsum(x_comp)
            ys = np.cumsum(y_comp)
            
            # calculate cell area and perimeter
            self.cell_area[i] = 1.0/2.0 * abs(np.sum(xs*np.roll(ys,-1) - ys*np.roll(xs,-1)))
            self.cell_perim[i] = np.sum(L_sides)
            
        self.cell_ni = self.ionDens[:self.nCells]
        self.cell_ne = self.elecDens[:self.nCells]
        self.cell_Ti = self.ionTemp[:self.nCells]
        self.cell_Te = self.elecTemp[:self.nCells]
        self.cell_nn = np.zeros(self.nCells)
            
        # initial cell neutral temperatures. Used for calculation of escape
        # probabilities
        self.cell_Tn_t = self.ionTemp[:self.nCells]
        self.cell_Tn_s = np.zeros(self.nCells) + 0.002
            
        # electron ionization cross sections
        if self.xsec_ione == 'janev':
            self.cell_sv_ion = sv.ione_janev(self.cell_Te)
        elif self.xsec_ione == 'stacey_thomas':
            self.cell_sv_ion = sv.ione_st(self.cell_ne, self.cell_Te)
        elif self.xsec_ione == 'degas':
            self.cell_sv_ion = sv.ione_degas(self.cell_ne, self.cell_Te)

        # recombination cross sections
        if self.xsec_rec == 'stacey_thomas':
            self.cell_sv_rec = sv.rec_st(self.cell_ne, self.cell_Te)
        elif self.xsec_rec == 'degas':
            self.cell_sv_rec = sv.rec_degas(self.cell_ne, self.cell_Te)

        # charge exchange cross sections
        if self.xsec_cx == 'janev':
            self.cell_sv_cx_s = sv.cx_janev(self.cell_Ti, self.cell_Tn_s)
            self.cell_sv_cx_t = sv.cx_janev(self.cell_Ti, self.cell_Tn_t)
        elif self.xsec_cx == 'stacey_thomas':
            self.cell_sv_cx_s = sv.cx_st(self.cell_Ti, self.cell_Tn_s)
            self.cell_sv_cx_t = sv.cx_st(self.cell_Ti, self.cell_Tn_t)
        elif self.xsec_cx == 'degas':
            self.cell_sv_cx_s = sv.cx_degas(self.cell_Ti, self.cell_Tn_s)
            self.cell_sv_cx_t = sv.cx_degas(self.cell_Ti, self.cell_Tn_t)

        # elastic scattering with ions cross sections
        if self.xsec_el == 'janev':
            print 'janev elastic scattering cross sections not available. Stopping.'
            sys.exit()
        elif self.xsec_el == 'stacey_thomas':
            self.cell_sv_el_s = sv.el_st(self.cell_Ti, self.cell_Tn_s)
            self.cell_sv_el_t = sv.el_st(self.cell_Ti, self.cell_Tn_t)
        elif self.xsec_el == 'degas':
            print 'degas elastic scattering cross sections not available. Stopping.'
            sys.exit()

        # elastic scattering with neutrals cross sections
        if self.xsec_eln == 'janev':
            print 'janev elastic scattering cross sections not available. Stopping.'
            sys.exit()
        elif self.xsec_eln == 'stacey_thomas':
            self.cell_sv_eln_s = sv.eln_st(self.cell_Tn_s)
            self.cell_sv_eln_t = sv.eln_st(self.cell_Tn_t)
        elif self.xsec_eln == 'degas':
            print 'degas elastic scattering cross sections not available. Stopping.'
            sys.exit()
            
        #calculate "cell mfp" for cold and thermal energy group using "cell" cross sections 
        #calculated above. This is used for the calculation of excape probabilities.

        self.cell_mfp_s = np.zeros(self.nCells)
        self.cell_mfp_t = np.zeros(self.nCells)
        self.cell_vn_s = np.sqrt(2 * self.cell_Tn_s * 1E3 * 1.6021E-19 / (m_p*self.aneut))
        self.cell_vn_t = np.sqrt(2 * self.cell_Tn_t * 1E3 * 1.6021E-19 / (m_p*self.aneut))
        for i, v in enumerate(self.cell_vn_t):
            self.cell_mfp_s[i] = self.cell_vn_s[i] / \
                            (self.cell_ne[i]*self.cell_sv_ion[i] + self.cell_ni[i]*self.cell_sv_cx_s[i] + self.cell_ni[i]*self.cell_sv_el_s[i])
            self.cell_mfp_t[i] = self.cell_vn_t[i] / \
                            (self.cell_ne[i]*self.cell_sv_ion[i] + self.cell_ni[i]*self.cell_sv_cx_t[i] + self.cell_ni[i]*self.cell_sv_el_t[i])
        
        self.c_i_s = (self.cell_sv_cx_s + self.cell_sv_el_s) / \
                        (self.cell_ne/self.cell_ni*self.cell_sv_ion + self.cell_sv_cx_s + self.cell_sv_el_s)
        self.c_i_t = (self.cell_sv_cx_t + self.cell_sv_el_t) / \
                        (self.cell_ne/self.cell_ni*self.cell_sv_ion + self.cell_sv_cx_t + self.cell_sv_el_t)
        
        self.X_i_s = 4.0*self.cell_area / (self.cell_mfp_s * self.cell_perim)
        self.X_i_t = 4.0*self.cell_area / (self.cell_mfp_t * self.cell_perim)
        
        n_sauer = 2.0931773
        self.P_0i_s = 1.0 / self.X_i_s * (1.0-(1.0+self.X_i_s/n_sauer)**-n_sauer)
        self.P_0i_t = 1.0 / self.X_i_t * (1.0-(1.0+self.X_i_t/n_sauer)**-n_sauer)
        self.P_i_s = self.P_0i_s / (1.0-self.c_i_s*(1.0-self.P_0i_s)) 
        self.P_i_t = self.P_0i_t / (1.0-self.c_i_t*(1.0-self.P_0i_t))

        return

    def calc_face_quantities(self):
        self.face_arrays = {}
        self.face_arrays["face_cell_ne"]         = ["float"]
        self.face_arrays["face_cell_ni"]         = ["float"]
        self.face_arrays["face_cell_Te"]         = ["float"]
        self.face_arrays["face_cell_Ti"]         = ["float"]
        self.face_arrays["face_cell_perim"]      = ["float"]
        self.face_arrays["face_lside"]           = ["float"]
        self.face_arrays["face_adjcell"]         = ["int"]
        self.face_arrays["face_int_type"]        = ["int"]
        self.face_arrays["face_alb_s"]           = ["float"]
        self.face_arrays["face_alb_t"]           = ["float"]
        self.face_arrays["face_awall"]           = ["int"]
        self.face_arrays["face_zwall"]           = ["int"]
        self.face_arrays["face_twall"]           = ["float"]
        self.face_arrays["face_f_abs"]           = ["float"]
        self.face_arrays["face_s_ext"]           = ["float"]
        self.face_arrays["face_Tn_fromcell_s"]   = ["float"]
        self.face_arrays["face_Tn_fromcell_t"]   = ["float"]
        self.face_arrays["face_len_frac"]        = ["float"]
        self.face_arrays["face_vns"]             = ["float"]
        self.face_arrays["face_vnt"]             = ["float"]
        self.face_arrays["face_Tn_intocell_s"]   = ["float"]
        self.face_arrays["face_Tn_intocell_t"]   = ["float"]
        self.face_arrays["face_refl_e_s"]        = ["float"]
        self.face_arrays["face_refl_e_t"]        = ["float"]
        self.face_arrays["face_refl_n_s"]        = ["float"]
        self.face_arrays["face_refl_n_t"]        = ["float"]
        self.face_arrays["face_reem_n_s"]        = ["float"]
        self.face_arrays["face_reem_n_t"]        = ["float"]
        
        self.face_arrays["face_sv_cx_s"]         = ["float"]
        self.face_arrays["face_sv_el_s"]         = ["float"]
        self.face_arrays["face_sv_eln_s"]        = ["float"]
        self.face_arrays["face_sv_ion_s"]        = ["float"]
        self.face_arrays["face_sv_cx_t"]         = ["float"]
        self.face_arrays["face_sv_el_t"]         = ["float"]
        self.face_arrays["face_sv_eln_t"]        = ["float"]
        self.face_arrays["face_sv_ion_t"]        = ["float"]
        self.face_arrays["c_ik_s"]               = ["float"]
        self.face_arrays["c_ik_t"]               = ["float"]
        self.face_arrays["face_mfp_s"]           = ["float"]
        self.face_arrays["face_mfp_t"]           = ["float"]

        print '  initializing face arrays'
        # INITIALIZE FACE ARRAYS
        # (#INITIALIZE FACE ARRAYS AS -1, THEN FILL IN WITH ACTUAL VALUES WHERE APPROPRIATE)
        for v in self.face_arrays:
            exec('self.%s = np.zeros((self.nCells,4), dtype=%s)-1' %(v, self.face_arrays[v][0]))
        
        # some face version of cell quantities
        self.face_cell_ne = np.tile(self.cell_ne.reshape(-1,1),4)
        self.face_cell_ni = np.tile(self.cell_ni.reshape(-1,1),4)
        self.face_cell_Te = np.tile(self.cell_Te.reshape(-1,1),4)
        self.face_cell_Ti = np.tile(self.cell_Ti.reshape(-1,1),4)
        self.face_cell_perim = np.tile(self.cell_perim.reshape(-1,1),4)
        
        self.face_lside = self.lsides[:self.nCells]
        self.face_adjcell = self.adjCell[:self.nCells]
        
        # TEMPERATURES FROM CELL MUST BE INITIALIZED BEFORE TEMPERATURES INTO CELL CAN BE CALCULATED
        for (i, j), val in np.ndenumerate(self.face_adjcell):
            if val != -1:
                self.face_Tn_fromcell_s[i, j] = 0.002
                self.face_Tn_fromcell_t[i, j] = self.cell_Ti[i]

        print '  NOW CALCULATE INCOMING TEMPERATURES AND RELATED QUANTITIES'
        # NOW CALCULATE INCOMING TEMPERATURES AND RELATED QUANTITIES
        for (i, j), val in np.ndenumerate(self.face_adjcell):
            if val != -1:
                self.face_int_type[i, j] = self.iType[val]
                # QUANTITIES PERTINENT FOR SIDES ADJACENT TO A NORMAL CELL
                if self.face_int_type[i, j] == 0:
                    self.face_Tn_intocell_s[i, j] = 0.002
                    self.face_Tn_intocell_t[i, j] = self.cell_Ti[val]
        
                    self.face_len_frac[i, j] = self.face_lside[i, j] / self.face_cell_perim[i, j]
                    self.face_vns[i, j] = sqrt(2*self.face_Tn_intocell_s[i, j]*1E3*1.6022E-19/(m_p*self.aneut))
                    self.face_vnt[i, j] = sqrt(2*self.face_Tn_intocell_t[i, j]*1E3*1.6022E-19/(m_p*self.aneut))
        
                # QUANTITIES PERTINENT FOR SIDES ADJACENT TO A PLASMA CORE CELL
                elif self.face_int_type[i, j] == 1:
                    self.face_Tn_intocell_s[i, j] = 0.002
                    self.face_Tn_intocell_t[i, j] = 5.0  # will be set in the input file
                    
                    self.face_alb_s[i, j] = self.alb_s[val]
                    self.face_alb_t[i, j] = self.alb_t[val]
                    self.face_s_ext[i, j] = self.s_ext[val]
                    self.face_len_frac[i, j] = self.face_lside[i,j] / self.face_cell_perim[i, j]
                    self.face_vns[i, j] = sqrt(2*self.face_Tn_intocell_s[i, j]*1E3*1.6022E-19/(m_p*self.aneut))
                    self.face_vnt[i, j] = sqrt(2*self.face_Tn_intocell_t[i, j]*1E3*1.6022E-19/(m_p*self.aneut))
                    
                # QUANTITIES PERTINENT FOR SIDES ADJACENT TO A WALL CELL
                elif self.face_int_type[i, j] == 2:
                    self.face_Tn_intocell_s[i, j] = 0.002
                    self.face_Tn_intocell_t[i, j] = self.face_Tn_fromcell_t[i, j] * self.face_refl_e_t[i, j] / self.face_refl_n_t[i, j]
                    
                    self.face_awall[i, j] = self.awall[val]
                    self.face_zwall[i, j] = self.zwall[val]
                    self.face_twall[i, j] = self.twall[val]
                    self.face_f_abs[i, j] = self.f_abs[val]
                    self.face_s_ext[i, j] = self.s_ext[val]
                    
                    self.face_refl_e_s[i, j] = self.calc_e_reflect(self.face_Tn_fromcell_s[i, j],
                                                                   self.aneut,
                                                                   self.face_awall[i, j],
                                                                   self.zion,
                                                                   self.face_zwall[i, j])
                    self.face_refl_e_t[i, j] = self.calc_e_reflect(self.face_Tn_fromcell_t[i, j],
                                                                   self.aneut,
                                                                   self.face_awall[i, j],
                                                                   self.zion,
                                                                   self.face_zwall[i, j])
                    self.face_refl_n_s[i, j] = self.calc_n_reflect(self.face_Tn_fromcell_s[i, j],
                                                                   self.aneut,
                                                                   self.face_awall[i, j],
                                                                   self.zion,
                                                                   self.face_zwall[i, j])
                    self.face_refl_n_t[i, j] = self.calc_n_reflect(self.face_Tn_fromcell_t[i, j],
                                                                   self.aneut,
                                                                   self.face_awall[i, j],
                                                                   self.zion,
                                                                   self.face_zwall[i, j])
                    self.face_reem_n_s[i, j] = (1-self.face_refl_n_s[i, j])*(1-self.face_f_abs[i, j])
                    self.face_reem_n_t[i, j] = (1-self.face_refl_n_t[i, j])*(1-self.face_f_abs[i, j])
                    self.face_len_frac[i, j] = self.face_lside[i, j] / self.face_cell_perim[i, j]
                    self.face_vns[i, j] = sqrt(2*self.face_Tn_intocell_s[i, j]*1E3*1.6022E-19/(m_p*self.aneut))
                    self.face_vnt[i, j] = sqrt(2*self.face_Tn_intocell_t[i, j]*1E3*1.6022E-19/(m_p*self.aneut))
        
                # IONIZATION CROSS SECTIONS
                if self.xsec_ione == 'janev':
                    self.face_sv_ion_s[i, j] = sv.ione_janev(self.cell_Te[i])
                    self.face_sv_ion_t[i, j] = sv.ione_janev(self.cell_Te[i])
                elif self.xsec_ione == 'stacey_thomas':
                    self.face_sv_ion_s[i, j] = sv.ione_st(self.cell_ne[i], self.cell_Te[i])
                    self.face_sv_ion_t[i, j] = sv.ione_st(self.cell_ne[i], self.cell_Te[i])
                elif self.xsec_ione == 'degas':
                    self.face_sv_ion_s[i, j] = sv.ione_degas(self.cell_ne[i], self.cell_Te[i])
                    self.face_sv_ion_t[i, j] = sv.ione_degas(self.cell_ne[i], self.cell_Te[i])

                # CHARGE EXCHANGE
                if self.xsec_cx == 'janev':
                    self.face_sv_cx_s[i, j] = sv.cx_janev(self.cell_Ti[i], self.face_Tn_intocell_s[i, j])
                    self.face_sv_cx_t[i, j] = sv.cx_janev(self.cell_Ti[i], self.face_Tn_intocell_t[i, j])
                elif self.xsec_cx == 'stacey_thomas':
                    self.face_sv_cx_s[i, j] = sv.cx_st(self.cell_Ti[i], self.face_Tn_intocell_s[i, j])
                    self.face_sv_cx_t[i, j] = sv.cx_st(self.cell_Ti[i], self.face_Tn_intocell_t[i, j])
                elif self.xsec_cx == 'degas':
                    self.face_sv_cx_s[i, j] = sv.cx_degas(self.cell_Ti[i], self.face_Tn_intocell_s[i, j])
                    self.face_sv_cx_t[i, j] = sv.cx_degas(self.cell_Ti[i], self.face_Tn_intocell_t[i, j])

                # ELASTIC SCATTERING WITH IONS CROSS SECTIONS
                if self.xsec_el == 'janev':
                    print 'janev elastic scattering cross sections not available. Stopping.'
                    sys.exit()
                elif self.xsec_el == 'stacey_thomas':
                    self.face_sv_el_s[i, j] = sv.el_st(self.cell_Ti[i], self.face_Tn_intocell_s[i, j])
                    self.face_sv_el_t[i, j] = sv.el_st(self.cell_Ti[i], self.face_Tn_intocell_t[i, j])
                elif self.xsec_el == 'degas':
                    print 'degas elastic scattering cross sections not available. Stopping.'
                    sys.exit()

                # ELASTIC SCATTERING WITH NEUTRALS CROSS SECTIONS
                if self.xsec_eln == 'janev':
                    print 'janev elastic scattering cross sections not available. Stopping.'
                    sys.exit()
                elif self.xsec_eln == 'stacey_thomas':
                    self.face_sv_eln_s[i, j] = sv.eln_st(self.face_Tn_intocell_s[i, j])
                    self.face_sv_eln_t[i, j] = sv.eln_st(self.face_Tn_intocell_t[i, j])
                elif self.xsec_eln =='degas':
                    print 'degas elastic scattering cross sections not available. Stopping.'
                    sys.exit()

                # calculate charge exchange and elastic scattering fraction for each face
                # this is used for once-colided neutrals that retain some "memory of the
                # cell (and therefore temperature) that they came from
                self.c_ik_s[i, j] = (self.face_sv_cx_s[i, j] + self.face_sv_el_s[i, j]) / \
                                (self.face_cell_ne[i, j]/self.face_cell_ni[i, j]*self.face_sv_ion_s[i, j] + self.face_sv_cx_s[i, j] + self.face_sv_el_s[i, j])
                self.c_ik_t[i, j] = (self.face_sv_cx_t[i, j] + self.face_sv_el_t[i, j]) / \
                                (self.face_cell_ne[i, j]/self.face_cell_ni[i, j]*self.face_sv_ion_t[i, j] + self.face_sv_cx_t[i, j] + self.face_sv_el_t[i, j])
                
                self.face_mfp_s[i, j] = self.face_vns[i, j] / (self.cell_ne[i]*self.cell_sv_ion[i] + self.cell_ni[i]*self.face_sv_cx_s[i, j] + self.cell_ni[i]*self.face_sv_el_s[i, j])
                self.face_mfp_t[i, j] = self.face_vnt[i, j] / (self.cell_ne[i]*self.cell_sv_ion[i] + self.cell_ni[i]*self.face_sv_cx_t[i, j] + self.cell_ni[i]*self.face_sv_el_t[i, j])

    def calc_tcoefs(self):
        def f(phi, xi, x_comp, y_comp, x_coords, y_coords, reg, mfp, fromcell, tocell, throughcell):
            try: 
                result = (2.0/(pi*-1*x_comp[-1])) * sin(phi) * self.Ki3_fit(li(phi, xi, x_coords, y_coords, reg) / mfp)
                return result
            except:
                print
                print 'something went wrong when evaluating A transmission coefficient:'
                print 'li = ', li(phi, xi, x_coords, y_coords, reg)
                print 'mfp = ',mfp
                print 'li/mfp = ', li(phi, xi, x_coords, y_coords, reg)/mfp
                #print 'Ki3_fit( li / mfp) = ',self.Ki3_fit( li(phi,xi,x_comp,y_comp,reg) / mfp)
                #print 'x_comp[-1] = ',x_comp[-1]
                print 'fromcell = ', fromcell
                print 'tocell = ', tocell
                print 'throughcell = ', throughcell
                print

                #sys.exit()
                if li(phi, xi, x_coords, y_coords, reg) / mfp > 100:
                    result = (2.0/(pi*-1*x_comp[-1])) * sin(phi) * self.Ki3_fit(100.0)
                    return result

        def li(phi, xi, x_coords, y_coords, reg):

            x_coords = x_coords - xi

            vert_phis = np.arctan2(y_coords, x_coords)
            vert_phis[0] = 0
            vert_phis[-1] = pi
                  
            if phi < pi:
                reg = np.searchsorted(vert_phis, phi, side='right')-1
            else:
                reg = np.searchsorted(vert_phis, phi, side='right')-2
        
            # points defining the side of the cell we're going to intersect with
            # eq of line is y = ((y2-y2)/(x2-x1))(x-x1)+y1
            x1,y1 = x_coords[reg], y_coords[reg]
            x2,y2 = x_coords[reg+1], y_coords[reg+1]
        
            # calculate intersection point
            if self.isclose(x2,x1): #then line is vertical
                x_int = x1
                y_int = tan(phi)*x_int
            else:
                # eq of the intersecting line is y= tan(phi)x ( + 0 because of coordinate system choice)
                # set two equations equal and solve for x, then solve for y
                x_int = ((y2-y1)/(x2-x1)*x1 - y1 ) / ((y2-y1)/(x2-x1) - tan(phi))
                y_int = tan(phi) * x_int
            
            return sqrt(x_int**2 + y_int**2)
        
        def phi_limits(xi, x_comp, y_comp, x_coords, y_coords, reg, mfp, fromcell, tocell, throughcell):
            x_coords = x_coords - xi
            vert_phis = np.arctan2(y_coords, x_coords)
            vert_phis[0] = 0
            vert_phis[-1] = pi
            return [vert_phis[reg], vert_phis[reg+1]]
                 
        def xi_limits(x_comp, y_comp, x_coords, y_coords, reg, mfp, fromcell, tocell, throughcell):
            return [0, -1*x_comp[-1]]
        
        self.T_coef_s = np.zeros((self.nCells, 4, 4), dtype='float')
        self.T_coef_t = np.zeros((self.nCells, 4, 4), dtype='float')
        self.T_from = np.zeros((self.nCells, 4, 4), dtype='int')
        self.T_to = np.zeros((self.nCells, 4, 4), dtype='int')
        self.T_via = np.zeros((self.nCells, 4, 4), dtype='int')
        
        # create bickley-naylor fit (much faster than evaluating Ki3 over and over)
        def Ki3(x):
            return integrate.quad(lambda theta: (sin(theta))**2 * exp(-x/sin(theta)), 0, pi/2)[0]

        Ki3_x = np.linspace(0, 100, 200)
        Ki3_v = np.zeros(Ki3_x.shape)
        for i, x in enumerate(Ki3_x):
            Ki3_v[i] = Ki3(x)
        self.Ki3_fit = interp1d(Ki3_x, Ki3_v)
        

        trans_coef_file = open(os.getcwd()+'/outputs/T_coef.txt','w')
        trans_coef_file.write(('{:^6s}'*3+'{:^12s}'*4+'\n').format("from", "to", "via", "T_slow", "T_thermal", "mfp_s", "mfp_t"))
        outof = np.sum(self.nSides[:self.nCells]**2)
        self.timetrack = 0.0
        print
        for (i, j, k), val in np.ndenumerate(self.T_coef_s):

            progress = self.nSides[i]**2 * i  # + self.nSides[i]*j + k

            L_sides = np.roll(self.lsides[i, :self.nSides[i]], -(j+1))  # begins with length of the current "from" side
            adj_cells = np.roll(self.adjCell[i, :self.nSides[i]], -j)
            angles = np.roll(self.angles[i, :self.nSides[i]], -j)*2*pi/360  # converted to radians
            angles[1:] = 2*pi-(pi-angles[1:])

            if k < adj_cells.size and j < adj_cells.size:
                
                self.T_from[i, j, k] = adj_cells[0]
                self.T_to[i, j, k] = adj_cells[k-j]
                self.T_via[i, j, k] = i
                if j == k:
                    # All flux from a side back through itself must have at least one collision
                    self.T_coef_s[i, j, k] = 0.0
                    self.T_coef_t[i, j, k] = 0.0
                    trans_coef_file.write(('{:>6d}'*3+'{:>12.3E}'*4+'\n').format(int(self.T_from[i,j,k]),int(self.T_to[i,j,k]),int(self.T_via[i,j,k]),self.T_coef_s[i,j,k],self.T_coef_t[i,j,k],self.face_mfp_s[i,k],self.face_mfp_t[i,k]))
                else:
                    side_thetas = np.cumsum(angles)
        
                    x_comp = np.cos(side_thetas) * L_sides
                    y_comp = np.sin(side_thetas) * L_sides
                    
                    y_coords = np.roll(np.flipud(np.cumsum(y_comp)), -1)
                    x_coords = np.roll(np.flipud(np.cumsum(x_comp)), -1)  # this gets adjusted for xi later, as part of the integration process
                    
                    reg = np.where(np.flipud(adj_cells[1:]) == self.T_to[i, j, k])[0][0]
                    
                    if self.int_method == 'midpoint':

                        kwargs_s = {"x_comp": x_comp,
                                    "y_comp": y_comp,
                                    "x_coords": x_coords,
                                    "y_coords": y_coords,
                                    "reg": reg,
                                    "mfp": self.face_mfp_s[i, j],  # not sure if this is j or k
                                    "fromcell": adj_cells[0],
                                    "tocell": adj_cells[k-j],
                                    "throughcell": i}

                        kwargs_t = {"x_comp": x_comp,
                                    "y_comp": y_comp,
                                    "x_coords": x_coords,
                                    "y_coords": y_coords,
                                    "reg": reg,
                                    "mfp": self.face_mfp_t[i, j],
                                    "fromcell": adj_cells[0],
                                    "tocell": adj_cells[k-j],
                                    "throughcell": i}
                        nx = 30
                        ny = 30

                        self.T_coef_t[i, j, k] = self.midpoint2D(f, phi_limits, xi_limits, nx, ny, **kwargs_t)
                        self.T_coef_s[i, j, k] = self.midpoint2D(f, phi_limits, xi_limits, nx, ny, **kwargs_s)

                        print 'angles = ', angles
                        print 'L_sides = ', L_sides
                        for key, value in kwargs_t.iteritems():
                            print key, value
                        print 'self.T_coef_t[i, j, k] = ',self.T_coef_t[i, j, k]
                        sys.exit()

                    elif self.int_method == 'quad':

                        self.T_coef_t[i, j, k] = integrate.nquad(f, [phi_limits, xi_limits], args=(x_comp, y_comp, reg, self.face_mfp_t[i, j]))[0]
                        self.T_coef_s[i, j, k] = integrate.nquad(f, [phi_limits, xi_limits], args=(x_comp, y_comp, reg, self.face_mfp_s[i, j]))[0]

                    trans_coef_file.write(('{:>6d}'*3+'{:>12.3E}'*4+'\n').format(int(self.T_from[i, j, k]),
                                                                                 int(self.T_to[i, j, k]),
                                                                                 int(self.T_via[i, j, k]),
                                                                                 self.T_coef_s[i, j, k],
                                                                                 self.T_coef_t[i, j, k],
                                                                                 self.face_mfp_s[i, j],
                                                                                 self.face_mfp_t[i, j]))
        
            self.print_progress(progress, outof)

            
        print '\n'
        trans_coef_file.close()

        # create t_coef_sum arrays for use later
        self.tcoef_sum_s = np.zeros((self.nCells, 4))
        self.tcoef_sum_t = np.zeros((self.nCells, 4))
        for i in range(0, self.nCells):
            for k in range(0, int(self.nSides[i])):
                self.tcoef_sum_s[i, k] = np.sum(self.T_coef_s[np.where((self.T_via == i) &
                                                                       (self.T_from == self.adjCell[i, k]))])
                self.tcoef_sum_t[i, k] = np.sum(self.T_coef_t[np.where((self.T_via == i) &
                                                                       (self.T_from == self.adjCell[i, k]))])

    def solve_matrix(self):
        # calculate size of matrix and initialize
        
        num_fluxes = int(np.sum(self.nSides[:self.nCells]))
        num_en_groups = 2
        M_size = int(num_fluxes * num_en_groups)
        M_matrix = np.zeros((M_size, M_size))
        
        # set the order of the fluxes in the solution vector
        flux_pos = np.zeros((M_size, 3), dtype=int)
        pos_count = 0
        for g in range(0, num_en_groups):
            for i in range(0, self.nCells):
                for j in range(0, (self.nSides[i])):
                    flux_pos[pos_count, 0] = g
                    flux_pos[pos_count, 1] = i
                    flux_pos[pos_count, 2] = self.adjCell[i, j]
                    pos_count += 1

        m_sparse = 1
        
        if m_sparse == 1 or m_sparse == 2:
            # construct sparse matrix using coordinate list (COO) method
            m_row = []
            m_col = []
            m_data = []

        #M_row = 0
        for i in range(0, self.nCells):
            for j in range(0, int(self.nSides[i])):
                for k in range(0, int(self.nSides[i])):
                    
                    curcell = int(i)
                    tocell = int(self.adjCell[i,j])
                    fromcell = int(self.adjCell[i,k])
        
                    # curcell_type = int(inp.iType[curcell])
                    # tocell_type = int(inp.iType[tocell])
                    fromcell_type = int(self.iType[fromcell])
        
                    T_coef_loc = np.where((self.T_via == curcell) &
                                          (self.T_from == fromcell) &
                                          (self.T_to == tocell))

                    face_loc = T_coef_loc[:2]
                    
                    ###############################################################
                    # FROM NORMAL CELL
                    ###############################################################
                    if fromcell_type == 0:  # if fromcell is normal cell
                        # determine row and column of matrix

                        M_row_s = np.where((flux_pos[:, 1] == curcell) & (flux_pos[:, 2] == tocell))[0][0]
                        M_col_s = np.where((flux_pos[:, 1] == fromcell) & (flux_pos[:, 2] == curcell))[0][0]
                        M_row_t = M_row_s + num_fluxes
                        M_col_t = M_col_s + num_fluxes
                        
                        if fromcell == tocell:
                            uncoll_ss = 0
                            uncoll_tt = 0
                        else:
                            uncoll_ss = self.T_coef_s[T_coef_loc]
                            uncoll_tt = self.T_coef_t[T_coef_loc]
        
                        coll_ss = 0 
                        coll_tt = (1-self.tcoef_sum_t[i, k]) * self.c_ik_t[i, k] * \
                                        (self.P_0i_t[i]*self.face_len_frac[i, j] + (1-self.P_0i_t[i])*self.c_i_t[i]*self.P_i_t[i]*self.face_len_frac[i, j])
                        coll_ts = 0 
                        coll_st = (1-self.tcoef_sum_s[i, k]) * self.c_ik_s[i, k] * \
                                        (self.P_0i_t[i]*self.face_len_frac[i, j] + (1-self.P_0i_t[i])*self.c_i_t[i]*self.P_i_t[i]*self.face_len_frac[i, j])
                        # SLOW GROUP EQUATIONS
                        if m_sparse == 0 or m_sparse == 2:
                            # matrix: from slow group into slow group
                            M_matrix[M_row_s, M_col_s] = uncoll_ss + coll_ss
                            # matrix: from thermal group into slow group
                            M_matrix[M_row_s, M_col_t] = coll_ts
                            # matrix: from slow group into thermal group
                            M_matrix[M_row_t, M_col_s] = coll_st
                            # matrix: from thermal group into thermal group
                            M_matrix[M_row_t, M_col_t] = uncoll_tt + coll_tt
                            
                        if m_sparse == 1 or m_sparse == 2:
                            # matrix: from slow group into slow group
                            m_row.append(M_row_s)
                            m_col.append(M_col_s)
                            m_data.append(uncoll_ss + coll_ss)
                            
                            # matrix: from thermal group into slow group
                            m_row.append(M_row_s)
                            m_col.append(M_col_t)
                            m_data.append(coll_ts)
                            
                            # matrix: from slow group into thermal group
                            m_row.append(M_row_t)
                            m_col.append(M_col_s)
                            m_data.append(coll_st)
                            
                            # matrix: from thermal group into thermal group
                            m_row.append(M_row_t)
                            m_col.append(M_col_t)
                            m_data.append(uncoll_tt + coll_tt)

                    ###############################################################
                    # FROM PLASMA CORE
                    ###############################################################
                    elif fromcell_type == 1:  # if fromcell is plasma core cell
                        # column from,to swapped because incoming is defined as a function of outgoing via albedo condition
                        M_row_s = np.where((flux_pos[:, 1] == curcell) & (flux_pos[:, 2] == tocell))[0][0]
                        M_col_s = np.where((flux_pos[:, 1] == curcell) & (flux_pos[:, 2] == fromcell))[0][0]
                        M_row_t = M_row_s + num_fluxes
                        M_col_t = M_col_s + num_fluxes
        
                        # uncollided flux from slow group to slow group
                        if fromcell == tocell:
                            uncoll_ss = 0.0
                            uncoll_tt = 0.0
                        else:
                            uncoll_ss = self.face_alb_s[face_loc] * self.T_coef_s[T_coef_loc]
                            uncoll_tt = self.face_alb_t[face_loc] * self.T_coef_t[T_coef_loc]
                        
                        coll_ss = 0
                        coll_tt = 0
                        coll_ts = 0
                        coll_st = 0
                        
                        if m_sparse == 0 or m_sparse == 2:
                            # matrix: from slow group into slow group
                            M_matrix[M_row_s, M_col_s] = uncoll_ss + coll_ss
                            # matrix: from thermal group into slow group
                            M_matrix[M_row_s, M_col_t] = coll_ts
                            # matrix: from slow group into thermal group
                            M_matrix[M_row_t, M_col_s] = coll_st
                            # matrix: from thermal group into thermal group
                            M_matrix[M_row_t, M_col_t] = uncoll_tt + coll_tt

                        if m_sparse == 1 or m_sparse == 2:
                            # matrix: from slow group into slow group
                            m_row.append(M_row_s)
                            m_col.append(M_col_s)
                            m_data.append(uncoll_ss + coll_ss)
                            
                            # matrix: from thermal group into slow group
                            m_row.append(M_row_s)
                            m_col.append(M_col_t)
                            m_data.append(coll_ts)
                            
                            # matrix: from slow group into thermal group
                            m_row.append(M_row_t)
                            m_col.append(M_col_s)
                            m_data.append(coll_st)
                            
                            # matrix: from thermal group into thermal group
                            m_row.append(M_row_t)
                            m_col.append(M_col_t)
                            m_data.append(uncoll_tt + coll_tt)
        
                    ###############################################################
                    # FROM WALL
                    ###############################################################
                    elif fromcell_type == 2:  # if fromcell is wall cell
                        # column from,to swapped because incoming is defined as a function of outgoing via reflection coefficient
                        M_row_s = np.where((flux_pos[:, 1]==curcell) & (flux_pos[:, 2] == tocell))[0][0]
                        M_col_s = np.where((flux_pos[:, 1]==curcell) & (flux_pos[:, 2] == fromcell))[0][0]
                        M_row_t = M_row_s + num_fluxes
                        M_col_t = M_col_s + num_fluxes
                        
                        # UNCOLLIDED FLUX
                        if fromcell == tocell:
                            # a particle going back to the cell it came from implies a collision
                            uncoll_refl_ss = 0 
                            uncoll_refl_tt = 0
                            uncoll_refl_ts = 0
                            uncoll_refl_st = 0
        
                            uncoll_reem_ss = 0
                            uncoll_reem_tt = 0
                            uncoll_reem_ts = 0
                            uncoll_reem_st = 0
                        else:
                            # slow neutrals can hit the wall, reflect as slow, and stream uncollided into adjacent cells
                            uncoll_refl_ss = self.face_refl_n_s[face_loc] * self.T_coef_s[T_coef_loc]
                            # thermal neutrals can hit the wall, reflect as thermals, and stream uncollided into adjacent cells
                            uncoll_refl_tt = self.face_refl_n_t[face_loc] * self.T_coef_t[T_coef_loc]
                            # thermal neutrals can reflect back as slow and then stay slow , but we'll treat this later
                            uncoll_refl_ts = 0
                            # slow neutrals can reflect back as slow neutrals, and end up as thermal, but that implies a collision
                            uncoll_refl_st = 0 
        
                            # slow neutrals can hit the wall, be reemitted as slow, and stream uncollided into adjacent cells
                            uncoll_reem_ss = (1.0 - self.face_refl_n_s[face_loc])*(1-self.face_f_abs[face_loc]) * self.T_coef_s[T_coef_loc] 
                            # thermal neutrals can hit the wall, be reemitted as slow, and then end up thermal again, but that implies a collision.
                            uncoll_reem_tt = 0
                            # thermal neutrals can hit the wall, be reemitted as slow, and then stream uncollided into adjacent cells
                            uncoll_reem_ts = (1.0 - self.face_refl_n_t[face_loc])*(1-self.face_f_abs[face_loc]) * self.T_coef_s[T_coef_loc]
                            # slow neutrals can hit the wall, be reemitted as slow, and then end up thermal again, but that implies a collision.
                            uncoll_reem_st = 0 
        
                        # COLLIDED FLUX
                        # slow neutrals can hit the wall, reflect as slow neutrals, but a collision removes them from the slow group
                        coll_refl_ss = 0
                        # thermal neutrals can hit the wall, reflect as thermal neutrals, and then have a collision and stay thermal afterward
                        coll_refl_tt = self.face_refl_n_t[face_loc] * (1-self.tcoef_sum_t[i,k]) * \
                                            self.c_ik_t[i,k]*(self.P_0i_t[i]*self.face_len_frac[i,j] + (1-self.P_0i_t[i])*self.c_i_t[i]*self.P_i_t[i]*self.face_len_frac[i,j])
                        # thermal neutrals can hit the wall, reflect as thermal neutrals, but they won't reenter the slow group
                        coll_refl_ts = 0
                        # slow neutrals can hit the wall, reflect as slow neutrals, and have a collision to enter and stay in the thermal group
        
                        coll_refl_st = self.face_refl_n_s[face_loc] * (1-self.tcoef_sum_s[i,k]) * \
                                            self.c_ik_s[i,k]*(self.P_0i_t[i]*self.face_len_frac[i,j] + (1-self.P_0i_t[i])*self.c_i_t[i]*self.P_i_t[i]*self.face_len_frac[i,j])
        
                        
                        # slow neutrals can hit the wall, be reemitted as slow, but a collision removes them from the slow group
                        coll_reem_ss = 0
                        # thermal neutrals can hit the wall, be reemitted as slow, and then collide to enter and stay in the thermal group
                        coll_reem_tt = (1-self.face_refl_n_t[face_loc])*(1-self.face_f_abs[face_loc])*(1-self.tcoef_sum_s[i,k]) * \
                                            self.c_ik_s[i,k]*(self.P_0i_t[i]*self.face_len_frac[i,j] + (1-self.P_0i_t[i])*self.c_i_t[i]*self.P_i_t[i]*self.face_len_frac[i,j])
                        # thermal neutrals can hit the wall, be reemitted as slow, but a collision removes them from the slow group
                        coll_reem_ts = 0
                        # slow neutrals can hit the wall, be reemitted as slow, and then collide to enter and stay in the thermal group
                        coll_reem_st = (1-self.face_refl_n_s[face_loc])*(1-self.face_f_abs[face_loc])*(1-self.tcoef_sum_s[i,k]) * \
                                            self.c_ik_s[i,k]*(self.P_0i_t[i]*self.face_len_frac[i,j] + (1-self.P_0i_t[i])*self.c_i_t[i]*self.P_i_t[i]*self.face_len_frac[i,j])
                        
                        if m_sparse == 0 or m_sparse == 2:
                            # matrix: from slow group into slow group
                            M_matrix[M_row_s, M_col_s] = uncoll_refl_ss + uncoll_reem_ss + coll_refl_ss + coll_reem_ss
                            # matrix: from thermal group into slow group
                            M_matrix[M_row_s, M_col_t] = uncoll_refl_ts + uncoll_reem_ts + coll_refl_ts + coll_reem_ts
                            # matrix: from slow group into thermal group
                            M_matrix[M_row_t, M_col_s] = uncoll_refl_st + uncoll_reem_st + coll_refl_st + coll_reem_st
                            # matrix: from thermal group into thermal group
                            M_matrix[M_row_t, M_col_t] = uncoll_refl_tt + uncoll_reem_tt + coll_refl_tt + coll_reem_tt          
                        if m_sparse == 1 or m_sparse == 2:
                            # matrix: from slow group into slow group
                            m_row.append(M_row_s)
                            m_col.append(M_col_s)
                            m_data.append(uncoll_refl_ss + uncoll_reem_ss + coll_refl_ss + coll_reem_ss)
                            
                            # matrix: from thermal group into slow group
                            m_row.append(M_row_s)
                            m_col.append(M_col_t)
                            m_data.append(uncoll_refl_ts + uncoll_reem_ts + coll_refl_ts + coll_reem_ts)
                            
                            # matrix: from slow group into thermal group
                            m_row.append(M_row_t)
                            m_col.append(M_col_s)
                            m_data.append(uncoll_refl_st + uncoll_reem_st + coll_refl_st + coll_reem_st)
                            
                            # matrix: from thermal group into thermal group
                            m_row.append(M_row_t)
                            m_col.append(M_col_t)
                            m_data.append(uncoll_refl_tt + uncoll_reem_tt + coll_refl_tt + coll_reem_tt)
        
        # create source vector

        # account for external sources
        flux_cells = self.adjCell[np.where(self.face_s_ext > 0)]
        flux_vals = self.face_s_ext[np.where(self.face_s_ext > 0)]
        
        source = np.zeros(M_size)
        for i, v in enumerate(flux_pos):
            group = v[0]
            cell_io = v[1]
            cell_to = v[2]
            if group == 0:
                # ADD CONTRIBUTION FROM VOLUMETRIC SOURCE (I.E. RECOMBINATION)
                source[i] = source[i] + 0  # assumed that all neutrals from recombination are thermal
                
                # ADD CONTRIBUTION FROM EXTERNAL SOURCES (all external sources are assumed to be slow for now)
                # if our cell has a wall that has an external source
                # TODO: I'm not confident that that this calculation is quite right. In the event that there are
                # multiple sides with external fluxes, it should loop over them and treat each of them individually.
                # this essentially requires an improved treatment of "cell_from"
                if np.any(np.in1d(flux_cells, self.adjCell[cell_io])):
                    cell_from = list(set(flux_cells) & set(self.adjCell[cell_io]))[0]
                    incoming_flux = flux_vals[np.where(flux_cells == cell_from)][0]
                    
                    T_coef_loc = np.where((self.T_via == cell_io) &
                                          (self.T_from == cell_from) &
                                          (self.T_to == cell_to))
                    face_from_loc = T_coef_loc[:2]
                    face_to_loc = [T_coef_loc[0], T_coef_loc[-1]]
        
                    # add uncollided slow flux from slow external source
                    source[i] = source[i] + incoming_flux * self.T_coef_s[T_coef_loc][0]
                    # add collided slow flux from slow external source
                    source[i] = source[i] + incoming_flux * 0
                    # add collided thermal flux from slow external source
                    source[i+num_fluxes] = source[i+num_fluxes] + incoming_flux * (1-self.tcoef_sum_s[face_from_loc]) * \
                                            self.c_ik_s[face_from_loc]*(self.P_0i_t[cell_io]*self.face_len_frac[face_to_loc] + (1-self.P_0i_t[cell_io])*self.c_i_t[cell_io]*self.P_i_t[cell_io]*self.face_len_frac[face_to_loc])
        
            if group == 1:
                # ADD CONTRIBUTION FROM VOLUMETRIC SOURCE (I.E. RECOMBINATION)
                source[i] = source[i] + 0  # cell_area[cell_io]*cell_ni[cell_io]*cell_sv_rec[cell_io]*P_i_t[cell_io]*0.25 #assumed that all neutrals from recombination are thermal

                # ADD CONTRIBUTION FROM IMPINGING IONS REFLECTING AS THERMAL NEUTRALS
                # loop over "from cells and determine if any are wall cells
                for count, cell_from in enumerate(self.adjCell[cell_io, :self.nSides[cell_io]]):
                    # if it's a wall cell
                    if self.iType[cell_from] == 2:
                        
                        # calculate flux to wall segment
                        ni = self.cell_ni[cell_io]
                        Ti = self.cell_Ti[cell_io] * 1.0E3 * 1.6021E-19
                        vel = sqrt(2.0*Ti/3.343583719E-27)
                        wall_length = self.face_lside[cell_io,count]
                        R0_wall = 1.4 #TODO: Read in R values for each wall segment
                        flux2wall = 2.0*pi*ni*vel*wall_length*R0_wall
                        
                        # calculate returning neutral flux
                        refl_coef = self.face_refl_n_t[cell_io, count]
                        reem_coef = self.face_reem_n_t[cell_io, count]
                        incoming_flux = flux2wall * refl_coef
                        
                        # calculate uncollided source to cell_to
                        T_coef_loc = np.where((self.T_via==cell_io)&(self.T_from==cell_from)&(self.T_to==cell_to))
                        face_from_loc = T_coef_loc[:2]
                        face_to_loc = [T_coef_loc[0], T_coef_loc[-1]]  
                        source[i] = source[i] + incoming_flux * self.T_coef_t[T_coef_loc][0]
                        
                        # calculate collided source to cell_to
                        source[i] = source[i] + incoming_flux * (1.0-self.tcoef_sum_t[face_from_loc]) * \
                                    self.c_ik_t[face_from_loc]*(self.P_0i_t[cell_io]*self.face_len_frac[face_to_loc] + (1.0-self.P_0i_t[cell_io])*self.c_i_t[cell_io]*self.P_i_t[cell_io]*self.face_len_frac[face_to_loc])
                        if incoming_flux<0:
                            print 'incoming flux less than zero'
                            print 'stopping'
                            sys.exit()

        # CREATE FINAL MATRIX AND SOLVE
        if m_sparse==0 or m_sparse==2:
            M_matrix = np.identity(M_size) - M_matrix
            flux_out = spsolve(M_matrix, source)
        if m_sparse==1 or m_sparse==2:
            # multiply m_data by -1 and append "identify matrix"
            # note: we're taking advantage of the way coo_matrix handles duplicate
            # entries to achieve the same thing as I - M
            m_row  = np.concatenate(( np.asarray(m_row),  np.arange(0,M_size)))
            m_col  = np.concatenate(( np.asarray(m_col),  np.arange(0,M_size)))
            m_data = np.concatenate((-np.asarray(m_data), np.ones(M_size)))
            m_sp_final = coo_matrix((m_data, (m_row, m_col))).tocsc()
            flux_out = spsolve(m_sp_final,source)

        # np.savetxt(os.getcwd()+'/outputs/matrix.txt',M_matrix,fmt='%1.5f')
        self.flux_out_s = np.zeros((self.nCells,4))
        self.flux_out_t = np.zeros((self.nCells,4))
        flux_counter = 0
        for g,g1 in enumerate(range(num_en_groups)):
            for i,v1 in enumerate(range(self.nCells)):
                for j,v2 in enumerate(range(self.nSides[i])):
                    if g==0:
                        self.flux_out_s[i,j] = flux_out[flux_counter]
                    if g==1:
                        self.flux_out_t[i,j] = flux_out[flux_counter]
                    flux_counter+=1
        
        # flux_out_s = np.reshape(np.split(flux_out,num_en_groups)[0],(inp.nCells,-1))
        # flux_out_t = np.reshape(np.split(flux_out,num_en_groups)[1],(inp.nCells,-1))
        self.flux_out_tot = self.flux_out_s + self.flux_out_t 

    def calc_neutral_dens(self):
        # first calculate the fluxes into cells from the fluxes leaving the cells
        self.flux_in_s = self.flux_out_s*0
        self.flux_in_t = self.flux_out_t*0
        
        flux_out_dest = self.adjCell[:self.nCells,:]
        flux_out_src  = flux_out_dest*0
        for i in range(self.nCells):
            flux_out_src[i] = i
        
        for i in range(0,self.nCells):
            for k in range(0,self.nSides[i]):
                if self.face_int_type[i,k]==0:
                    self.flux_in_s[i,k] = self.flux_out_s[np.where((flux_out_dest==i)&(flux_out_src==self.adjCell[i,k]))]
                    self.flux_in_t[i,k] = self.flux_out_t[np.where((flux_out_dest==i)&(flux_out_src==self.adjCell[i,k]))]
                if self.face_int_type[i,k]==1:
                    self.flux_in_s[i,k] = self.flux_out_s[i,k]*self.face_alb_s[i,k]
                    self.flux_in_t[i,k] = self.flux_out_t[i,k]*self.face_alb_t[i,k]
                    self.flux_in_s[i,k] = self.flux_in_s[i,k] + self.face_s_ext[i,k]
                if self.face_int_type[i,k]==2:
                    self.flux_in_s[i,k] = self.flux_out_s[i,k]*(self.face_refl_n_s[i,k]+(1.0-self.face_refl_n_s[i,k])*(1-self.face_f_abs[i,k])) + \
                                     self.flux_out_t[i,k]*(1.0-self.face_refl_n_t[i,k])*(1-self.face_f_abs[i,k])
                    self.flux_in_t[i,k] = self.flux_out_t[i,k]*self.face_refl_n_t[i,k]
                    self.flux_in_s[i,k] = self.flux_in_s[i,k] + self.face_s_ext[i,k]
        
        
        self.flux_in_tot = self.flux_in_s + self.flux_in_t
        
        #calculate ionization rate
        self.cell_izn_rate_s = np.zeros(self.nCells)
        self.cell_izn_rate_t = np.zeros(self.nCells)
        for i in range(0,self.nCells):
            #add contribution to ionization rate from fluxes streaming into cell
            for k in range(0,self.nSides[i]):
                    self.cell_izn_rate_s[i] = self.cell_izn_rate_s[i] + self.flux_in_s[i,k]*(1.0-self.tcoef_sum_s[i,k])* \
                                            ((1.0-self.c_ik_s[i,k]) + self.c_ik_s[i,k]*(1.0-self.c_i_t[i])*(1.0-self.P_0i_t[i])/(1-self.c_i_t[i]*(1.0-self.P_0i_t[i])))
                                            
                    self.cell_izn_rate_t[i] = self.cell_izn_rate_t[i] + self.flux_in_t[i,k]*(1.0-self.tcoef_sum_t[i,k])* \
                                            ((1.0-self.c_ik_t[i,k]) + self.c_ik_t[i,k] *(1.0-self.c_i_t[i])*(1.0-self.P_0i_t[i])/(1-self.c_i_t[i]*(1.0-self.P_0i_t[i])))
                                          
            #add contribution to ionization rate from volumetric recombination within the cell
            self.cell_izn_rate_s[i] = self.cell_izn_rate_s[i] + 0 #all recombination neutrals are assumed to be thermal
            self.cell_izn_rate_t[i] = self.cell_izn_rate_t[i] + 0*(1-self.P_0i_t[i])*self.cell_area[i]*self.cell_ni[i]*self.cell_sv_rec[i]* \
                                            (1.0 - self.c_i_t[i] + self.c_i_t[i]*(1.0-self.c_i_t[i])*(1-self.P_0i_t[i])/(1.0-self.c_i_t[i]*(self.P_0i_t[i])))
        
        self.cell_izn_rate = self.cell_izn_rate_s + self.cell_izn_rate_t
        
        #calculate neutral densities from ionization rates
        self.cell_nn_s = self.cell_izn_rate_s / (self.cell_ni*self.cell_sv_ion*self.cell_area)
        self.cell_nn_t = self.cell_izn_rate_t / (self.cell_ni*self.cell_sv_ion*self.cell_area)
        self.cell_nn = self.cell_izn_rate / (self.cell_ni*self.cell_sv_ion*self.cell_area)
        
        ## fix negative values (usually won't be necessary)
        for i,v1 in enumerate(self.cell_nn):
            if v1<0:
                #get neutral densities of adjacent cells
                #average the positive values
                print i, 'Found a negative density. Fixing by using average of surrounding cells.'
                # print 'You probably need to adjust the way you are calculating the transmission coefficients.'
                nn_sum = 0
                nn_count = 0
                for j,v2 in enumerate(self.adjCell[i]):
                    if self.iType[v2]==0:
                        if self.cell_nn[j]>0:
                            nn_sum = nn_sum + self.cell_nn[j]
                            nn_count +=1
                self.cell_nn[i] = nn_sum / nn_count
        
        #print 'Checking particle balance...'
        #relerr = abs(np.sum(self.flux_out_tot)+np.sum(self.cell_izn_rate)-np.sum(self.flux_in_tot))/ \
        #            ((np.sum(self.flux_out_tot)+np.sum(self.cell_izn_rate)+np.sum(self.flux_in_tot))/2)
        #if relerr<0.001: #this is arbitrary. It's just here to alert if there's a problem in the particle balance.
        #    print 'Particle balance passes. {5.3f}% relative error.'.format(relerr*100)
        #else:
        #    print 'Particle balance failed. {}% relative error.'.format(relerr*100)
        #    for i,(v1,v2) in enumerate(zip(self.flux_in_tot,self.flux_out_tot)):
        #        print i,np.sum(v1),np.sum(v2)+self.cell_izn_rate[i]
                
    def write_outputs(self):
        #WRITE FACE QUANTITIES
        face_data = open("./outputs/face_data.txt","w")
        for v in self.face_arrays:
            face_data.write(v+'\n')
            array = self.face_int_type
            exec('array = %s'%(v))
            for row,vals in enumerate(array):
                face_data.write(('{:<3}'+'{:>12.3E}'*4+'\n').format(row,vals[0],vals[1],vals[2],vals[3]))
            face_data.write('\n')
        face_data.close()
            
        #WRITE CELL QUANTITIES
        cell_vals_file = open(os.getcwd()+'/outputs/cell_values.txt','w')
        cell_vals_file.write(('{:^{width}s}'*20+'\n').format("cell_area","cell_perimeter","cell_ni","cell_ne","cell_nn","cell_mfp_s", "cell_mfp_t", "cell_sv_ion", "cell_sv_rec", "cell_sv_cx_s", "cell_sv_el_s", "cell_sv_eln_s", "cell_sv_cx_t", "cell_sv_el_t", "cell_sv_eln_t", "cell_izn_rate", "c_i_t","X_i_t","P_0i_t","P_i_t",width=15))
        for i,(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20) in enumerate(zip(self.cell_area,self.cell_perim,self.cell_ni, self.cell_ne, self.cell_nn, self.cell_mfp_s, self.cell_mfp_t, self.cell_sv_ion, self.cell_sv_rec, self.cell_sv_cx_s, self.cell_sv_el_s, self.cell_sv_eln_s, self.cell_sv_cx_t, self.cell_sv_el_t, self.cell_sv_eln_t, self.cell_izn_rate, self.c_i_t,self.X_i_t,self.P_0i_t,self.P_i_t)):
            cell_vals_file.write(('{:>{width}.3E}'*20+'\n').format(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,width=15))
        cell_vals_file.close()
        
class read_infile():
    def __init__(self,infile):
        print ('READING NEUTPY INPUT FILE')
        
        #some regex commands we'll use when reading stuff in from the input file
        r0di = "r'.*%s *= *([ ,\d]*) *'%(v)"
        r0df = "r'.*%s *= *([+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?) *'%(v)"
        r0ds = "r'.*%s *= *(\w+) *'%(v)"
        r1di = "r'.*%s\( *(\d*) *\) *= *(\d*) *'%(v)"
        r1df = "r'.*%s\( *(\d*)\) *= *([+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?) *'%(v)"
        #r2df = "r'.*%s\( *(\d*)\) *= *([+\-]*[ ,\.\d]*) *'%(v)"
        #r2df = "r'.*%s\( *(\d*)\) *= *((?:[+\-]?\d*\.?\d*(?:[eE]?[+\-]?\d+)?,?)+) *'%(v)"
        r2df = "r'.*%s\( *(\d*)\) *= *((?:[+\-]?\d*\.?\d*(?:[eE]?[+\-]?\d+),? ?)+) *'%(v)"
        
        v0d = {}
        v0d["nCells"]       = ["int",r0di]
        v0d["nPlasmReg"]    = ["int",r0di]
        v0d["nWallSegm"]    = ["int",r0di]
        v0d["aneut"]        = ["int",r0di]
        v0d["zion"]         = ["int",r0di]
        v0d["aion"]         = ["int",r0di]
        v0d["tslow"]        = ["float",r0df]
        v0d["int_method"]   = ["str",r0ds]
        v0d["phi_int_pts"]  = ["int",r0di]
        v0d["xi_int_pts"]   = ["int",r0di]
        v0d["xsec_ioni"]    = ["str",r0ds]
        v0d["xsec_ione"]    = ["str",r0ds]
        v0d["xsec_cx"]      = ["str",r0ds]
        v0d["xsec_rec"]     = ["str",r0ds]
        v0d["xsec_el"]      = ["str",r0ds]
        v0d["xsec_eln"]     = ["str",r0ds]
        v0d["refmod_e"]     = ["str",r0ds]
        v0d["refmod_n"]     = ["str",r0ds]
        v0d["cell1_ctr_x"]  = ["float",r0df]
        v0d["cell1_ctr_y"]  = ["float",r0df]
        v0d["cell1_theta0"] = ["float",r0df]

        #initialize 0d varialbes
        for v in v0d:
            exec('self.%s = 0' %(v))

        #populate 0d variables. Need to do this first to calculate the sizes of the other arrays
        with open(os.getcwd() + '/' + infile, 'r') as toneut:
            for count, line in enumerate(toneut):
                if not line.startswith("#"):
                    #read in 0d variables
                    for v in v0d:
                        exec("result = re.match(%s,line)"%(v0d[v][1]))
                        if result:
                            exec("self.%s = %s(result.group(1))"%(v,v0d[v][0]))

        self.nCells_tot  = self.nCells + self.nPlasmReg + self.nWallSegm

        #now we can do the same thing for the 1d and 2d arrays
        v1d = {}
        v1d["iType"]     = [self.nCells_tot,"int",r1di]
        v1d["nSides"]    = [self.nCells_tot,"int",r1di]
        v1d["zwall"]     = [self.nCells_tot,"int",r1di]
        v1d["awall"]     = [self.nCells_tot,"int",r1di]
        v1d["elecTemp"]  = [self.nCells_tot,"float",r1df]
        v1d["ionTemp"]   = [self.nCells_tot,"float",r1df]
        v1d["elecDens"]  = [self.nCells_tot,"float",r1df]
        v1d["ionDens"]   = [self.nCells_tot,"float",r1df]
        v1d["twall"]     = [self.nCells_tot,"float",r1df]
        v1d["f_abs"]     = [self.nCells_tot,"float",r1df]
        v1d["alb_s"]     = [self.nCells_tot,"float",r1df]
        v1d["alb_t"]     = [self.nCells_tot,"float",r1df]
        v1d["s_ext"]     = [self.nCells_tot,"float",r1df]
        
        v2d = {}        
        v2d["adjCell"]    = [self.nCells_tot,4,"int",r2df]
        v2d["lsides"]     = [self.nCells_tot,4,"float",r2df]
        v2d["angles"]     = [self.nCells_tot,4,"float",r2df]
        
        #initialize elements and arrays
        for v in v1d:
            exec('self.%s = np.zeros(%s, dtype=%s)' %(v, v1d[v][0], v1d[v][1]))
        for v in v2d:
            exec('self.%s = np.zeros((%s,%s), dtype=%s)' %(v, v2d[v][0], v2d[v][1], v2d[v][2]))
            #fill with -1. elements that aren't replaced with a non-negative number correspond to
            #a side that doesn't exist. Several other more elegant approaches were tried, including 
            #masked arrays, and they all resulted in significantly hurt performance. 
            exec('self.%s[:] = -1' %(v))

        #populate arrays
        with open(os.getcwd() + '/' + infile, 'r') as toneut:
            for count, line in enumerate(toneut):
                if not line.startswith("#"):                   

                    #read in 1d arrays
                    for v in v1d:
                        exec("result = re.match(%s,line)"%(v1d[v][2]))
                        if result:
                            exec("self.%s[int(result.group(1))] = result.group(2)"%(v))

                    #read in 2d arrays
                    for v in v2d:
                        exec("result = re.match(%s,line)"%(v2d[v][3]))
                        if result:
                            #read new vals into an array
                            exec("newvals = np.asarray(result.group(2).split(','),dtype=%s)"%(v2d[v][2]))
                            #pad it to make it the correct size to include in the array
                            exec("self.%s[int(result.group(1)),:] = np.pad(newvals,(0,%s),mode='constant',constant_values=0)"%(v,4-newvals.shape[0]))
                            #finally, mask the array elements that don't correspond to anything to prevent 
                            #problems later (and there will be problems later if we don't.)
                            #fill with -1. elements that aren't replaced with a non-negative number correspond to
                            #a side that doesn't exist. Several other more elegant approaches were tried, including 
                            #masked arrays, and they all resulted in significantly hurt performance. 
                            exec("self.%s[int(result.group(1)),%s:] = -1"%(v,newvals.shape[0]))
        return        

class neutpyplot():
    
    def __init__(self,neut=None):
        print 'GENERATING NEUTPY PLOTS'
        sys.setrecursionlimit(10000)
        if neut==None: #then look for input file 'neutpy_in_generated'
            neut = infile('neutpy_in_generated2')
            self.cellmap(neut)
        else:
            self.cellmap(neut)
        self.cellprop(neut)

    def cellmap(self,neut):
        
        xcoords = np.zeros(neut.adjCell.shape)
        ycoords = np.zeros(neut.adjCell.shape)
        beta = np.zeros(neut.adjCell.shape) #beta is the angle of each side with respect to the +x axis. 

        def loop(neut,oldcell,curcell,cellscomplete,xcoords,ycoords):
            beta[curcell,:neut.nSides[curcell]] = np.cumsum(np.roll(neut.angles[curcell,:neut.nSides[curcell]],1)-180)+180
            
            #if first cell:
            if oldcell==0 and curcell==0:
                #rotate cell by theta0 value (specified)
                beta[curcell,:neut.nSides[curcell]] = beta[curcell,:neut.nSides[curcell]] + neut.cell1_theta0
                x_comp = np.cos(np.radians(beta[curcell,:neut.nSides[curcell]])) * neut.lsides[curcell,:neut.nSides[curcell]]
                y_comp = np.sin(np.radians(beta[curcell,:neut.nSides[curcell]])) * neut.lsides[curcell,:neut.nSides[curcell]]        
                xcoords[curcell,:neut.nSides[curcell]] = np.roll(np.cumsum(x_comp),1) + neut.cell1_ctr_x
                ycoords[curcell,:neut.nSides[curcell]] = np.roll(np.cumsum(y_comp),1) + neut.cell1_ctr_x
                
            #for all other cells:
            else:

                #adjust all values in beta for current cell such that the side shared
                #with oldcell has the same beta as the oldcell side
                oldcell_beta = beta[oldcell,:][np.where(neut.adjCell[oldcell,:]==curcell)][0]
                delta_beta = beta[curcell,np.where(neut.adjCell[curcell,:]==oldcell)]+180 - oldcell_beta
                beta[curcell,:neut.nSides[curcell]] = beta[curcell,:neut.nSides[curcell]]-delta_beta

                #calculate non-shifted x- and y- coordinates
                x_comp = np.cos(np.radians(beta[curcell,:neut.nSides[curcell]])) * neut.lsides[curcell,:neut.nSides[curcell]]
                y_comp = np.sin(np.radians(beta[curcell,:neut.nSides[curcell]])) * neut.lsides[curcell,:neut.nSides[curcell]]      
                xcoords[curcell,:neut.nSides[curcell]] = np.roll(np.cumsum(x_comp),1)  #xcoords[oldcell,np.where(neut.adjCell[oldcell,:]==curcell)[0][0]]
                ycoords[curcell,:neut.nSides[curcell]] = np.roll(np.cumsum(y_comp),1)  #ycoords[oldcell,np.where(neut.adjCell[oldcell,:]==curcell)[0][0]]

                cur_in_old = np.where(neut.adjCell[oldcell,:]==curcell)[0][0]
                old_in_cur = np.where(neut.adjCell[curcell,:]==oldcell)[0][0]
                mdpt_old_x = (xcoords[oldcell,cur_in_old] + np.roll(xcoords[oldcell,:],-1)[cur_in_old])/2
                mdpt_old_y = (ycoords[oldcell,cur_in_old] + np.roll(ycoords[oldcell,:],-1)[cur_in_old])/2
                mdpt_cur_x = (xcoords[curcell,old_in_cur] + np.roll(xcoords[curcell,:],-1)[old_in_cur])/2
                mdpt_cur_y = (ycoords[curcell,old_in_cur] + np.roll(ycoords[curcell,:],-1)[old_in_cur])/2

                xshift = mdpt_old_x - mdpt_cur_x
                yshift = mdpt_old_y - mdpt_cur_y
  
                xcoords[curcell,:] = xcoords[curcell,:] + xshift #xcoords[oldcell,np.where(neut.adjCell[oldcell,:]==curcell)[0][0]]
                ycoords[curcell,:] = ycoords[curcell,:] + yshift #ycoords[oldcell,np.where(neut.adjCell[oldcell,:]==curcell)[0][0]]

            #continue looping through adjacent cells
            for j,newcell in enumerate(neut.adjCell[curcell,:neut.nSides[curcell]]):
                #if the cell under consideration is a normal cell (>3 sides) and not complete, then move into that cell and continue
                if neut.nSides[newcell]>=3 and cellscomplete[newcell]==0:
                    cellscomplete[newcell]=1
                    loop(neut,curcell,newcell,cellscomplete,xcoords,ycoords)
            
            return xcoords,ycoords
        
        ## Add initial cell to the list of cells that are complete
        cellscomplete = np.zeros(neut.nCells)
        cellscomplete[0] = 1
        self.xs,self.ys = loop(neut,0,0,cellscomplete,xcoords,ycoords)
        
        #plot cell diagram
        #grid = plt.figure(figsize=(160,240))
        grid = plt.figure(figsize=(8,12))
        ax1 = grid.add_subplot(111)
        ax1.set_title('Neutrals Mesh',fontsize=30)
        ax1.set_ylabel(r'Z ($m$)',fontsize=30)
        ax1.set_xlabel(r'R ($m$)',fontsize=30)
        ax1.tick_params(labelsize=15)
        ax1.axis('equal')
        for i,(v1,v2) in enumerate(zip(self.xs,self.ys)):
            if neut.nSides[i]==len(v1):
                v1 = np.append(v1,v1[0])
                v2 = np.append(v2,v2[0])
            elif neut.nSides[i]==3 and len(v1)==4:
                v1[3] = v1[0]
            #elif neut.nSides[i]==3 and len(v1)==3:
            #    v1 = np.append(v1,v1[0])
            #    v2 = np.append(v2,v2[0])
            ax1.plot(v1,v2,color='black',lw=1)  
        #fig.tight_layout()
        grid.savefig("./figures/neutpy_mesh.png", dpi=300,transparent=True,bbox_inches="tight")

    def cellprop(self,neut):
        colors = np.log10(neut.cell_nn)

        patches = []
        for i in range(0,neut.nCells):
            verts = np.column_stack((self.xs[i,:neut.nSides[i]],self.ys[i,:neut.nSides[i]]))
            polygon = Polygon(verts,closed=True)
            patches.append(polygon)

        collection1 = PatchCollection(patches,cmap='viridis')
        collection1.set_array(np.array(colors))

        fig, ax1 = plt.subplots(figsize=(8, 12))
        cax = ax1.add_collection(collection1)
        ax1.axis('equal')
        ax1.set_title('Calculated Neutral Densities',fontsize=30)
        ax1.set_ylabel(r'Z ($m$)',fontsize=30)
        ax1.set_xlabel(r'R ($m$)',fontsize=30)
        ax1.tick_params(labelsize=30)
        cb = fig.colorbar(cax)
        cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=30)
        #fig.tight_layout()
        fig.savefig("./figures/neutpy_nn.png", dpi=300,transparent=True,bbox_inches="tight")
        
if __name__ == "__main__":
    ###############################################################################
    # CREATE NEUTPY INSTANCE (READS INPUT FILE, INSTANTIATES SOME FUNCTIONS)
    # (OPTIONALLY PASS THE NAME OF AN INPUT FILE. IF NO ARGUMENT PASSED, IT LOOKS
    # FOR A FILE CALLED 'neutpy_in'.)
    ###############################################################################
    #neut = neutpy('toneut_converted_mod')
    neut = neutpy('neutpy_in_generated2')
    plot = neutpyplot(neut)
