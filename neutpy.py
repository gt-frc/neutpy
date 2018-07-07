# !/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Neutpy calculates neutral densities, ionization rates, and related quantities in tokamaks.

The neutpy module contains three classes: neutpy, read_infile, and neutpyplot.

"""
from __future__ import division
import sys
import os
from math import sqrt, pi, sin, tan, exp
import numpy as np
from scipy.interpolate import interp1d
from scipy import integrate
from scipy.constants import m_p
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix
from neutpy_xsec import calc_xsec
from collections import namedtuple
import pandas as pd

# instantiate cross sections class
sv = calc_xsec()


def isnamedtupleinstance(x):
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple:
        return False
    f = getattr(t, '_fields', None)
    if not isinstance(f, tuple):
        return False
    return all(type(n) == str for n in f)


def iterate_namedtuple(object, df):
    if isnamedtupleinstance(object):
        for key, item in object._asdict().iteritems():
            if isnamedtupleinstance(item):
                iterate_namedtuple(item, df)
            else:
                df[key] = pd.Series(item.flatten(), name=key)
    else:
        pass
    return df


# isclose is included in python3.5+, so you can delete this if the code ever gets ported into python3.5+
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


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
    percents = str_format.format(100 * ((iteration + 1) / float(total)))
    filled_length = int(round(bar_length * (iteration + 1) / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix))


def midpoint2D(f, f_limx, f_limy, nx, ny, **kwargs):
    """calculates a double integral using the midpoint rule"""
    I = 0
    # start with outside (y) limits of integration
    c, d = f_limy(**kwargs)
    hy = (d - c) / float(ny)
    for j in range(ny):
        yj = c + hy / 2 + j * hy
        # for each j, calculate inside limits of integration
        a, b = f_limx(yj, **kwargs)
        hx = (b - a) / float(nx)
        for i in range(nx):
            xi = a + hx / 2 + i * hx
            I += hx * hy * f(xi, yj, **kwargs)
    return I


def calc_Ki3(x):
    return integrate.quad(lambda theta: (sin(theta)) ** 2 * exp(-x / sin(theta)), 0, pi / 2)[0]


def calc_e_reflect(e0, am1, am2, z1, z2):
    """
    Calculates the energy reflection coefficient
    :param e0:
    :param am1:
    :param am2:
    :param z1:
    :param z2:
    :return:
    """

    e = 2.71828

    ae = np.array([[0.001445, 0.2058, 0.4222, 0.4484, 0.6831],
                   [404.7, 3.848, 3.092, 27.16, 27.16],
                   [73.73, 19.07, 13.17, 15.66, 15.66],
                   [0.6519, 0.4872, 0.5393, 0.6598, 0.6598],
                   [4.66, 15.13, 4.464, 7.967, 7.967],
                   [1.971, 1.638, 1.877, 1.822, 1.822]])

    mu = am2 / am1

    zfactr = 1.0 / (z1 * z2 * np.sqrt(z1 ** 0.67 + z2 ** 0.67))
    epsln = 32.55 * mu * zfactr * e0 / (1. + mu)
    if mu == 1:
        col = 0
    elif mu == 3:
        col = 1
    elif 6.0 <= mu <= 7.0:
        col = 2
    elif 12.0 <= mu <= 15.0:
        col = 3
    elif mu >= 20.0:
        col = 4

    r_e = ae[0, col] * np.log(ae[1, col] * epsln + e) / \
          (1 + ae[2, col] * epsln ** ae[3, col] + ae[4, col] * epsln ** ae[5, col])

    return r_e


def calc_n_reflect(e0, am1, am2, z1, z2):
    """

    :param e0:
    :param am1:
    :param am2:
    :param z1:
    :param z2:
    :return:
    """
    e = 2.71828

    an = np.array([[0.02129, 0.36800, 0.51730, 0.61920, 0.82500],
                   [16.39000, 2.98500, 2.54900, 20.01000, 21.41000],
                   [26.39000, 7.12200, 5.32500, 8.92200, 8.60600],
                   [0.91310, 0.58020, 0.57190, 0.66690, 0.64250],
                   [6.24900, 4.21100, 1.09400, 1.86400, 1.90700],
                   [2.55000, 1.59700, 1.93300, 1.89900, 1.92700]])

    mu = am2 / am1
    zfactr = 1.0 / (z1 * z2 * sqrt(z1 ** 0.67 + z2 ** 0.67))
    epsln = 32.55 * mu * zfactr * e0 / (1. + mu)

    if mu == 1:
        col = 0
    elif mu == 3:
        col = 1
    elif 6.0 <= mu <= 7.0:
        col = 2
    elif 12.0 <= mu <= 15.0:
        col = 3
    elif mu >= 20.0:
        col = 4

    r_n = an[0, col] * np.log(an[1, col] * epsln + e) / \
          (1 + an[2, col] * epsln ** an[3, col] + an[4, col] * epsln ** an[5, col])
    return r_n


def calc_svione(n, T, xsec_ione='degas'):
    """

    :param n:
    :param T:
    :param xsec_ione:
    :return:
    """

    Te = T.e

    # reshape ne if necessary, i.e. when calculating face values
    # note, this shouldn't ever be necessary for this sv, but I could be wrong.
    if Te.ndim == 2:
        ne = np.repeat(n.e.reshape(-1, 1), Te.shape[1], axis=1)
    else:
        ne = n.e

    if xsec_ione == 'janev':
        sv_ione = sv.ione_janev(Te)
    elif xsec_ione == 'stacey_thomas':
        sv_ione = sv.ione_st(ne, Te)
    elif xsec_ione == 'degas':
        sv_ione = sv.ione_degas(ne, Te)
    return sv_ione


def calc_svrec(n, T, xsec_rec='stacey_thomas'):
    """
        Calculate recombination cross section.

    :param n:
    :param T:
    :param xsec_rec:
    :return:
    """

    Te = T.e

    # reshape ne if necessary, i.e. when calculating face values
    # note, this shouldn't ever be necessary for this sv, but I could be wrong.
    if Te.ndim == 2:
        ne = np.repeat(n.e.reshape(-1, 1), Te.shape[1], axis=1)
    else:
        ne = n.e

    if xsec_rec == 'stacey_thomas':
        sv_rec = sv.rec_st(ne, Te)
    elif xsec_rec == 'degas':
        sv_rec = sv.rec_degas(ne, Te)
    return sv_rec


def calc_svcx(T, Tn, en_grp, xsec_cx='degas'):
    """
        Calculate charge exchange cross sections

    :param T:
    :param Tn:
    :param en_grp:
    :param xsec_cx:
    :return:
    """
    # determine neutral temperature array to use based on specified group
    Tn = Tn.s if en_grp == 'slow' else Tn.t

    # reshape ne if necessary, i.e. when calculating face values
    if Tn.ndim == 2:
        Ti = np.repeat(T.i.reshape(-1, 1), Tn.shape[1], axis=1)
    else:
        Ti = T.i

    if xsec_cx == 'janev':
        sv_cx = sv.cx_janev(Ti, Tn)
    elif xsec_cx == 'stacey_thomas':
        sv_cx = sv.cx_st(Ti, Tn)
    elif xsec_cx == 'degas':
        sv_cx = sv.cx_degas(Ti, Tn)

    return sv_cx


def calc_svel(T, Tn, en_grp, xsec_el='stacey_thomas'):
    """
        Calculates elastic scattering with ions cross sections

    :param T:
    :param Tn:
    :param en_grp:
    :param xsec_el:
    :return:
    """
    # determine neutral temperature array to use based on specified group
    Tn = Tn.s if en_grp == 'slow' else Tn.t

    # reshape ne if necessary, i.e. when calculating face values
    if Tn.ndim == 2:
        Ti = np.repeat(T.i.reshape(-1, 1), Tn.shape[1], axis=1)
    else:
        Ti = T.i

    if xsec_el == 'janev':
        print 'janev elastic scattering cross sections not available. Using Stacey-Thomas instead.'
        sv_el = sv.el_st(Ti, Tn)
    elif xsec_el == 'stacey_thomas':
        sv_el = sv.el_st(Ti, Tn)
    elif xsec_el == 'degas':
        print 'degas elastic scattering cross sections not available. Using Stacey-Thomas instead.'
        sv_el = sv.el_st(Ti, Tn)
    return sv_el


def calc_sveln(Tn, en_grp, xsec_eln='stacey_thomas'):
    """
        Calculates elastic scattering with neutrals cross sections

    :param Tn:
    :param en_grp:
    :param xsec_eln:
    :return:
    """
    # determine neutral temperature array to use based on specified group
    Tn = Tn.s if en_grp == 'slow' else Tn.t

    if xsec_eln == 'janev':
        print 'janev elastic scattering cross sections not available. Using Stacey-Thomas instead..'
        sv_eln = sv.eln_st(Tn)
    elif xsec_eln == 'stacey_thomas':
        sv_eln = sv.eln_st(Tn)
    elif xsec_eln == 'degas':
        print 'degas elastic scattering cross sections not available. Using Stacey-Thomas instead..'
        sv_eln = sv.eln_st(Tn)
    return sv_eln


def calc_mfp(Tn, n, sv, en_grp):
    """
        Calculates the mean free path of a neutral particle through a background plasma

    :param Tn:
    :param n:
    :param sv:
    :param en_grp:
    :return:
    """
    # TODO: get this information from input data
    mn = 2*m_p

    Tn = Tn.s if en_grp == 'slow' else Tn.t
    svcx = sv.cx_s if en_grp == 'slow' else sv.cx_t
    svel = sv.el_s if en_grp == 'slow' else sv.el_t

    # reshape ne and ni if necessary, i.e. when calculating face values
    if Tn.ndim == 2:
        ne = np.repeat(n.e.reshape(-1, 1), Tn.shape[1], axis=1)
        ni = np.repeat(n.i.reshape(-1, 1), Tn.shape[1], axis=1)
        svion = np.repeat(sv.ion.reshape(-1, 1), Tn.shape[1], axis=1)
    else:
        ne = n.e
        ni = n.i
        svion = sv.ion

    vn = np.sqrt(2 * Tn * 1E3 * 1.6021E-19 / mn)
    mfp = vn / (ne * svion + ni * svcx + ni * svel)

    # test if there are any NaN's in the array before returning
    if np.any(np.isnan(mfp)):
        array_type = 'cell' if Tn.ndim == 2 else 'face'
        nan_locs = np.argwhere(np.isnan(mfp))
        print 'an NAN was found in the '+array_type+' '+en_grp+' mfp array'
        print 'indices:'
        print nan_locs
        print
        print 'vn at those indices'
        print vn[nan_locs]
        print
        print 'ne at those indices'
        print ne[nan_locs]
        print
        print 'ni at those indices'
        print svion[nan_locs]
        print
        print 'svion at those indices'
        print vn[nan_locs]
        print
        print 'svcx at those indices'
        print svcx[nan_locs]
        print
        print 'svel at those indices'
        print svel[nan_locs]
        print
        print 'mfp array'
        print mfp
        print 'stopping.'
        sys.exit()


    return mfp


def calc_c_i(n, sv, en_grp):
    """

    :param n:
    :param sv:
    :param en_grp:
    :return:
    """

    svcx = sv.cx_s if en_grp == 'slow' else sv.cx_t
    svel = sv.el_s if en_grp == 'slow' else sv.el_t

    # reshape ne and ni if necessary, i.e. when calculating face values
    if svcx.ndim == 2:
        ne = np.repeat(n.e.reshape(-1, 1), svcx.shape[1], axis=1)
        ni = np.repeat(n.i.reshape(-1, 1), svcx.shape[1], axis=1)
        svion = np.repeat(sv.ion.reshape(-1, 1), svcx.shape[1], axis=1)
    else:
        ne = n.e
        ni = n.i
        svion = sv.ion

    c_i = (svcx + svel) / (ne / ni * svion + svcx + svel)
    return c_i


def calc_X_i(geom, mfp, en_grp):
    """

    :param geom:
    :param mfp:
    :param en_grp:
    :return:
    """

    mfp_vals = mfp.s if en_grp == 'slow' else mfp.t

    X_i = 4.0 * geom.area / (mfp_vals * geom.perim)
    return X_i


def calc_P_0i(X_i, en_grp):
    """

    :param X_i:
    :param en_grp:
    :return:
    """
    X_i = X_i.s if en_grp == 'slow' else X_i.t

    n_sauer = 2.0931773
    P_0i = 1 / X_i * (1 - (1 + X_i / n_sauer) ** -n_sauer)
    return P_0i


def calc_P_i(n, sv, P_0i, en_grp):
    """

    :param n:
    :param sv:
    :param P_0i:
    :param en_grp:
    :return:
    """

    P_0i = P_0i.s if en_grp == 'slow' else P_0i.t

    c_i = calc_c_i(n, sv, en_grp)
    P_i = P_0i / (1 - c_i * (1 - P_0i))
    return P_i


def calc_refl_alb(cell_T, face_adj):
    # TODO: get am1 and z1 from input data
    am1 = 2
    z1 = 1

    refle_s = np.zeros(face_adj.int_type.shape)
    refle_t = np.zeros(face_adj.int_type.shape)
    refln_s = np.zeros(face_adj.int_type.shape)
    refln_t = np.zeros(face_adj.int_type.shape)
    alb_s = np.zeros(face_adj.int_type.shape)
    alb_t = np.zeros(face_adj.int_type.shape)
    f_abs = np.zeros(face_adj.int_type.shape)

    for (cell, side), itype in np.ndenumerate(face_adj.int_type):
        if itype == 0:  # regular cell
            refle_s[cell, side] = 0
            refle_t[cell, side] = 0
            refln_s[cell, side] = 0
            refln_t[cell, side] = 0
            alb_s[cell, side] = 0
            alb_t[cell, side] = 0
            f_abs[cell, side] = 0
        elif itype == 1:  # plasma core cell
            refle_s[cell, side] = 0
            refle_t[cell, side] = 0
            refln_s[cell, side] = 0
            refln_t[cell, side] = 0
            # TODO: get albedo information from input data
            alb_s[cell, side] = 0.1
            alb_t[cell, side] = 0
            f_abs[cell, side] = 0
        elif itype == 2:  # wall cell
            # TODO: get Tn_s from input data
            refle_s[cell, side] = calc_e_reflect(0.002, am1, face_adj.awall[cell, side], z1, face_adj.zwall[cell, side])
            refle_t[cell, side] = calc_e_reflect(cell_T.i[cell], am1, face_adj.awall[cell, side], z1, face_adj.zwall[cell, side])
            refln_s[cell, side] = calc_n_reflect(0.002, am1, face_adj.awall[cell, side], z1, face_adj.zwall[cell, side])
            refln_t[cell, side] = calc_n_reflect(cell_T.i[cell], am1, face_adj.awall[cell, side], z1, face_adj.zwall[cell, side])
            alb_s[cell, side] = 0
            alb_t[cell, side] = 0
            f_abs[cell, side] = 0

    refle_dict = {}
    refle_dict['s'] = refle_s
    refle_dict['t'] = refle_t
    refle = namedtuple('refle', refle_dict.keys())(*refle_dict.values())

    refln_dict = {}
    refln_dict['s'] = refle_s
    refln_dict['t'] = refle_t
    refln = namedtuple('refln', refln_dict.keys())(*refln_dict.values())

    refl_dict = {}
    refl_dict['e'] = refle
    refl_dict['n'] = refln
    refl = namedtuple('refl', refl_dict.keys())(*refl_dict.values())

    alb_dict = {}
    alb_dict['s'] = alb_s
    alb_dict['t'] = alb_t
    alb = namedtuple('alb', alb_dict.keys())(*alb_dict.values())

    return alb, refl, f_abs


def calc_Tn_intocell_t(face_adj, cell_T, refl):

    # this function is only concerned with the temperature of incoming THERMAL neutrals
    refle = refl.e.t
    refln = refl.n.t

    Tn_intocell_t = np.zeros(face_adj.int_type.shape)
    for (cell, side), itype in np.ndenumerate(face_adj.int_type):
        adjCell = face_adj.cellnum[cell, side]
        if itype == 0:
            # incoming neutral temperate equal to ion temperature in cell it's coming from
            Tn_intocell_t[cell, side] = cell_T.i[adjCell]
        elif itype == 1:
            # incoming neutral temperature equal to the temperature of the current cell. It's close enough
            # and doesn't make much of a difference.
            Tn_intocell_t[cell, side] = cell_T.i[cell]
        elif itype == 2:
            Tn_intocell_t[cell, side] = cell_T.i[cell] * refle[cell, side] / refln[cell, side]

    return Tn_intocell_t


def calc_ext_src(face_adj, src):

    face_ext_src = np.zeros(face_adj.int_type.shape)
    for (cell, side), itype in np.ndenumerate(face_adj.int_type):
        adjCell = face_adj.cellnum[cell, side]
        if itype == 0:
            face_ext_src[cell, side] = 0
        elif itype == 1:
            face_ext_src[cell, side] = 0
        elif itype == 2:
            face_ext_src[cell, side] = src[adjCell]
    return face_ext_src


class neutpy:
    
    def __init__(self, infile=None, inarrs=None):
        print 'BEGINNING NEUTPY'
        
        sys.dont_write_bytecode = True 
        if not os.path.exists(os.getcwd()+'/outputs'):
            os.makedirs(os.getcwd()+'/outputs')
        if not os.path.exists(os.getcwd()+'/figures'):
            os.makedirs(os.getcwd()+'/figures')
            
        # EITHER READ INPUT FILE OR ACCEPT ARRAYS FROM ANOTHER PROGRAM
        if infile == None and inarrs == None:
            # should look for default input file like neutpy_in.txt or something
            pass
        elif infile != None and inarrs == None:
            # read input file
            inp = read_infile(infile)
            self.__dict__ = inp.__dict__.copy()
        elif infile == None and inarrs != None:
            # accept arrays
            self.__dict__ = inarrs  # .__dict__.copy()
        elif infile != None and inarrs != None:
            print 'You\'ve specified both an input file and passed arrays directly. ' \
                  'Please remove one of the inputs and try again.'
            sys.exit()

        # initialize cell densities
        cell_n_dict = {}
        cell_n_dict['i'] = self.ionDens[:self.nCells]
        cell_n_dict['e'] = self.elecDens[:self.nCells]
        cell_n_dict['n'] = np.zeros(self.nCells)
        cell_n = namedtuple('cell_n', cell_n_dict.keys())(*cell_n_dict.values())

        # initialize cell ion and electron temperatures
        cell_T_dict = {}
        cell_T_dict['i'] = self.ionTemp[:self.nCells]
        cell_T_dict['e'] = self.elecTemp[:self.nCells]
        cell_T = namedtuple('cell_T', cell_T_dict.keys())(*cell_T_dict.values())

        # initialize cell neutral temperatures
        cell_Tn_dict = {}
        cell_Tn_dict['t'] = self.ionTemp[:self.nCells]
        cell_Tn_dict['s'] = np.full(cell_Tn_dict['t'].shape, 0.002)
        cell_Tn = namedtuple('cell_Tn', cell_Tn_dict.keys())(*cell_Tn_dict.values())

        # initialize cell areas and perimeters
        cell_geom_dict = {}
        cell_geom_dict['area'], cell_geom_dict['perim'] = self.calc_cell_geom()
        cell_geom = namedtuple('cell_geom', cell_geom_dict.keys())(*cell_geom_dict.values())

        # initialize cell electron ionization cross sections
        cell_sv_dict = {}
        cell_sv_dict['ion'] = calc_svione(cell_n, cell_T)
        cell_sv_dict['rec'] = calc_svrec(cell_n, cell_T)
        cell_sv_dict['cx_s'] = calc_svcx(cell_T, cell_Tn, 'slow')
        cell_sv_dict['cx_t'] = calc_svcx(cell_T, cell_Tn, 'thermal')

        cell_sv_dict['el_s'] = calc_svel(cell_T, cell_Tn, 'slow')
        cell_sv_dict['el_t'] = calc_svel(cell_T, cell_Tn, 'thermal')
        cell_sv_dict['eln_s'] = calc_sveln(cell_Tn, 'slow')
        cell_sv_dict['eln_t'] = calc_sveln(cell_Tn, 'thermal')
        cell_sv = namedtuple('cell_sv', cell_sv_dict.keys())(*cell_sv_dict.values())

        # initialize cell mfp values
        cell_mfp_dict = {}
        cell_mfp_dict['s'] = calc_mfp(cell_Tn, cell_n, cell_sv, 'slow')
        cell_mfp_dict['t'] = calc_mfp(cell_Tn, cell_n, cell_sv, 'thermal')
        cell_mfp = namedtuple('cell_mfp', cell_mfp_dict.keys())(*cell_mfp_dict.values())

        # initialize cell c_i values
        cell_ci_dict = {}
        cell_ci_dict['s'] = calc_c_i(cell_n, cell_sv, 'slow')
        cell_ci_dict['t'] = calc_c_i(cell_n, cell_sv, 'thermal')
        cell_ci = namedtuple('cell_ci', cell_ci_dict.keys())(*cell_ci_dict.values())

        # initialize cell X_i values
        cell_Xi_dict = {}
        cell_Xi_dict['s'] = calc_X_i(cell_geom, cell_mfp, 'slow')
        cell_Xi_dict['t'] = calc_X_i(cell_geom, cell_mfp, 'thermal')
        cell_Xi = namedtuple('cell_Xi', cell_Xi_dict.keys())(*cell_Xi_dict.values())

        # initialize cell P_0i values
        cell_P0i_dict = {}
        cell_P0i_dict['s'] = calc_P_0i(cell_Xi, 'slow')
        cell_P0i_dict['t'] = calc_P_0i(cell_Xi, 'thermal')
        cell_P0i = namedtuple('cell_P0i', cell_P0i_dict.keys())(*cell_P0i_dict.values())

        # initialize cell P_i values
        cell_Pi_dict = {}
        cell_Pi_dict['s'] = calc_P_i(cell_n, cell_sv, cell_P0i, 'slow')
        cell_Pi_dict['t'] = calc_P_i(cell_n, cell_sv, cell_P0i, 'thermal')
        cell_Pi = namedtuple('cell_Pi', cell_Pi_dict.keys())(*cell_Pi_dict.values())

        # combine into 'cell' dictionary and namedtuple
        cell_dict = {}
        cell_dict['n'] = cell_n
        cell_dict['T'] = cell_T
        cell_dict['Tn'] = cell_Tn
        cell_dict['area'] = cell_geom.area
        cell_dict['perim'] = cell_geom.perim
        cell_dict['sv'] = cell_sv
        cell_dict['mfp'] = cell_mfp
        cell_dict['ci'] = cell_ci
        cell_dict['Xi'] = cell_Xi
        cell_dict['P0i'] = cell_P0i
        cell_dict['Pi'] = cell_Pi
        cell = namedtuple('cell', cell_dict.keys())(*cell_dict.values())

        #########################################################################

        # initialize face geometry and adjacent cell parameters
        face_geom_dict = {}
        face_geom_dict['lside'] = self.lsides[:self.nCells]
        face_geom_dict['lfrac'] = face_geom_dict['lside'] / \
                                  np.sum(face_geom_dict['lside'], axis=1).reshape((-1, 1))
        face_geom = namedtuple('face_geom', face_geom_dict.keys())(*face_geom_dict.values())

        face_adj = self.calc_adj_cell_prop()  # instance method that already has everything it needs

        # initialize neutral temperatures for neutrals entering the cell
        face_alb, face_refl, face_f_abs = calc_refl_alb(cell_T, face_adj)

        face_Tn_in_dict = {}
        face_Tn_in_dict['s'] = np.full(face_adj.int_type.shape, 0.002)
        face_Tn_in_dict['t'] = calc_Tn_intocell_t(face_adj, cell_T, face_refl)
        face_Tn_in = namedtuple('face_Tn_in', face_Tn_in_dict.keys())(*face_Tn_in_dict.values())
        face_s_ext = calc_ext_src(face_adj, self.s_ext)

        face_sv_dict = {}
        face_sv_dict['ion'] = cell_sv.ion  # included so face_sv has a complete set of cross sections
        face_sv_dict['rec'] = cell_sv.rec  # included so face_sv has a complete set of cross sections
        face_sv_dict['cx_s'] = calc_svcx(cell_T, face_Tn_in, 'slow')
        face_sv_dict['cx_t'] = calc_svcx(cell_T, face_Tn_in, 'thermal')
        face_sv_dict['el_s'] = calc_svel(cell_T, face_Tn_in, 'slow')
        face_sv_dict['el_t'] = calc_svel(cell_T, face_Tn_in, 'thermal')
        face_sv_dict['eln_s'] = calc_sveln(face_Tn_in, 'slow')
        face_sv_dict['eln_t'] = calc_sveln(face_Tn_in, 'thermal')
        face_sv = namedtuple('face_sv', face_sv_dict.keys())(*face_sv_dict.values())

        face_ci_dict = {}
        face_ci_dict['s'] = calc_c_i(cell_n, face_sv, 'slow')
        face_ci_dict['t'] = calc_c_i(cell_n, face_sv, 'thermal')
        face_ci = namedtuple('face_ci', face_ci_dict.keys())(*face_ci_dict.values())

        face_mfp_dict = {}
        face_mfp_dict['s'] = calc_mfp(face_Tn_in, cell_n, face_sv, 'slow')
        face_mfp_dict['t'] = calc_mfp(face_Tn_in, cell_n, face_sv, 'thermal')
        face_mfp = namedtuple('face_mfp', face_mfp_dict.keys())(*face_mfp_dict.values())

        # combine into a face dictionary and namedtuple
        face_dict = {}
        face_dict['lside'] = face_geom.lside
        face_dict['lfrac'] = face_geom.lfrac
        face_dict['adj'] = face_adj
        face_dict['alb'] = face_alb
        face_dict['f_abs'] = face_f_abs
        face_dict['refl'] = face_refl
        face_dict['Tn_in'] = face_Tn_in
        face_dict['s_ext'] = face_s_ext
        face_dict['sv'] = face_sv
        face_dict['ci'] = face_ci
        face_dict['mfp'] = face_mfp
        face = namedtuple('face', face_dict.keys())(*face_dict.values())

        # compute transmission coefficients
        self.T_coef = self.calc_tcoefs(face, int_method='quad')

        # construct and solve the matrix to obtain the fluxes
        self.flux = self.solve_matrix(face, cell, self.T_coef)

        # compute ionization rates and densities
        self.izn_rate, self.nn = self.calc_neutral_dens(cell, face, self.T_coef, self.flux)

        # write neutpy output files
        #self.write_outputs(cell)

    def calc_cell_geom(self):
        cell_area = np.zeros(self.nCells)
        cell_perim = np.zeros(self.nCells)

        for i in range(0, self.nCells):
            L_sides = self.lsides[i, :self.nSides[i]]

            angles2 = self.angles[i, :self.nSides[i]] * 2 * pi / 360  # in radians
            theta = np.zeros(angles2.shape)
            for j in range(0, int(self.nSides[i])):
                if j == 0:
                    theta[j] = 0.0
                elif j == 1:
                    theta[j] = angles2[0]
                else:
                    theta[j] = theta[j - 1] + angles2[j - 1] - pi

            x_comp = L_sides * np.cos(theta)
            y_comp = L_sides * np.sin(theta)

            x_comp[0] = 0
            y_comp[0] = 0

            xs = np.cumsum(x_comp)
            ys = np.cumsum(y_comp)

            # calculate cell area and perimeter
            cell_area[i] = 1.0 / 2.0 * abs(np.sum(xs * np.roll(ys, -1) - ys * np.roll(xs, -1)))
            cell_perim[i] = np.sum(L_sides)
        return cell_area, cell_perim

    def calc_adj_cell_prop(self):
        """
        Determines the type of interface for each face of a cell
        :param adjCells:
        :param iType:
        :return:
        """

        face_adjcell = self.adjCell[:self.nCells]

        face_int_type = np.zeros(face_adjcell.shape)
        face_awall = np.zeros(face_adjcell.shape)
        face_zwall = np.zeros(face_adjcell.shape)
        face_twall = np.zeros(face_adjcell.shape)
        face_f_abs = np.zeros(face_adjcell.shape)
        face_s_ext = np.zeros(face_adjcell.shape)
        for (i, j), val in np.ndenumerate(face_adjcell):
            if val != -1:
                face_int_type[i, j] = self.iType[val]
                face_awall[i, j] = self.awall[val]
                face_zwall[i, j] = self.zwall[val]
                face_twall[i, j] = self.twall[val]
                face_f_abs[i, j] = self.f_abs[val]
                face_s_ext[i, j] = self.s_ext[val]

        face_adj_dict = {}
        face_adj_dict['cellnum'] = face_adjcell
        face_adj_dict['int_type'] = face_int_type
        face_adj_dict['awall'] = face_awall
        face_adj_dict['zwall'] = face_zwall
        face_adj_dict['twall'] = face_twall
        face_adj_dict['f_abs'] = face_f_abs
        face_adj_dict['s_ext'] = face_s_ext
        face_adj = namedtuple('face_adj', face_adj_dict.keys())(*face_adj_dict.values())

        return face_adj

    def calc_tcoefs(self, face, int_method='quad'):
        # create bickley-naylor fit (much faster than evaluating Ki3 over and over)
        Ki3_x = np.linspace(0, 100, 200)
        Ki3 = np.zeros(Ki3_x.shape)
        for i, x in enumerate(Ki3_x):
            Ki3[i] = calc_Ki3(x)
        Ki3_fit = interp1d(Ki3_x, Ki3)

        def f(phi, xi, x_comp, y_comp, x_coords, y_coords, reg, mfp, fromcell, tocell, throughcell):
            try:
                result = (2.0/(pi*-1*x_comp[-1])) * sin(phi) * Ki3_fit(li(phi, xi, x_coords, y_coords, reg) / mfp)
                return result
            except:
                print
                print 'something went wrong when evaluating A transmission coefficient:'
                print 'li = ', li(phi, xi, x_coords, y_coords, reg)
                print 'mfp = ', mfp
                print 'li/mfp = ', li(phi, xi, x_coords, y_coords, reg)/mfp
                print 'fromcell = ', fromcell
                print 'tocell = ', tocell
                print 'throughcell = ', throughcell
                print
                if li(phi, xi, x_coords, y_coords, reg) / mfp > 100:
                    result = (2.0/(pi*-1*x_comp[-1])) * sin(phi) * Ki3_fit(100.0)
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
            x1, y1 = x_coords[reg], y_coords[reg]
            x2, y2 = x_coords[reg+1], y_coords[reg+1]
        
            # calculate intersection point
            if isclose(x2, x1):  # then line is vertical
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

        # arrays to be filled in and returned
        T_coef_s = np.zeros((self.nCells, 4, 4), dtype='float')
        T_coef_t = np.zeros((self.nCells, 4, 4), dtype='float')
        T_from = np.zeros((self.nCells, 4, 4), dtype='int')
        T_to = np.zeros((self.nCells, 4, 4), dtype='int')
        T_via = np.zeros((self.nCells, 4, 4), dtype='int')

        trans_coef_file = open(os.getcwd()+'/outputs/T_coef.txt', 'w')
        trans_coef_file.write(('{:^6s}'*3+'{:^12s}'*4+'\n').format("from", "to", "via", "T_slow", "T_thermal", "mfp_s", "mfp_t"))
        outof = np.sum(self.nSides[:self.nCells]**2)

        for (i, j, k), val in np.ndenumerate(T_coef_s):

            progress = self.nSides[i]**2 * i  # + self.nSides[i]*j + k

            L_sides = np.roll(self.lsides[i, :self.nSides[i]], -(j+1))  # begins with length of the current "from" side
            adj_cells = np.roll(self.adjCell[i, :self.nSides[i]], -j)
            angles = np.roll(self.angles[i, :self.nSides[i]], -j)*2*pi/360  # converted to radians
            angles[1:] = 2*pi-(pi-angles[1:])

            if k < adj_cells.size and j < adj_cells.size:

                T_from[i, j, k] = adj_cells[0]
                T_to[i, j, k] = adj_cells[k-j]
                T_via[i, j, k] = i
                if j == k:
                    # All flux from a side back through itself must have at least one collision
                    T_coef_s[i, j, k] = 0.0
                    T_coef_t[i, j, k] = 0.0
                    trans_coef_file.write(('{:>6d}'*3+'{:>12.3E}'*4+'\n').format(int(T_from[i, j, k]), int(T_to[i, j, k]), int(T_via[i, j, k]), T_coef_s[i, j, k], T_coef_t[i, j, k], face.mfp.s[i, k], face.mfp.t[i, k]))
                else:
                    side_thetas = np.cumsum(angles)

                    x_comp = np.cos(side_thetas) * L_sides
                    y_comp = np.sin(side_thetas) * L_sides

                    y_coords = np.roll(np.flipud(np.cumsum(y_comp)), -1)
                    x_coords = np.roll(np.flipud(np.cumsum(x_comp)), -1)  # this gets adjusted for xi later, as part of the integration process

                    reg = np.where(np.flipud(adj_cells[1:]) == T_to[i, j, k])[0][0]

                    if int_method == 'midpoint':

                        kwargs_s = {"x_comp": x_comp,
                                    "y_comp": y_comp,
                                    "x_coords": x_coords,
                                    "y_coords": y_coords,
                                    "reg": reg,
                                    "mfp": face.mfp.s[i, j], # not sure if this is j or k
                                    "fromcell": adj_cells[0],
                                    "tocell": adj_cells[k-j],
                                    "throughcell": i}

                        kwargs_t = {"x_comp": x_comp,
                                    "y_comp": y_comp,
                                    "x_coords": x_coords,
                                    "y_coords": y_coords,
                                    "reg": reg,
                                    "mfp": face.mfp.t[i, j],
                                    "fromcell": adj_cells[0],
                                    "tocell": adj_cells[k-j],
                                    "throughcell": i}
                        nx = 10
                        ny = 10

                        T_coef_t[i, j, k] = midpoint2D(f, phi_limits, xi_limits, nx, ny, **kwargs_t)
                        T_coef_s[i, j, k] = midpoint2D(f, phi_limits, xi_limits, nx, ny, **kwargs_s)

                    elif int_method == 'quad':
                        #T_coef_s[i, j, k] = 0
                        #T_coef_t[i, j, k] = 0

                        T_coef_s[i, j, k] = integrate.nquad(f, [phi_limits, xi_limits],
                                                            args=(x_comp,
                                                                  y_comp,
                                                                  x_coords,
                                                                  y_coords,
                                                                  reg,
                                                                  face.mfp.s[i, j],
                                                                  adj_cells[0],
                                                                  adj_cells[k-j],
                                                                  i),
                                                            opts=dict([('epsabs', 1.49e-2),
                                                                       ('epsrel', 10.00e-4),
                                                                       ('limit', 2)]))[0]

                        T_coef_t[i, j, k] = integrate.nquad(f, [phi_limits, xi_limits],
                                                            args=(x_comp,
                                                                  y_comp,
                                                                  x_coords,
                                                                  y_coords,
                                                                  reg,
                                                                  face.mfp.t[i, j],
                                                                  adj_cells[0],
                                                                  adj_cells[k-j],
                                                                  i),
                                                            opts=dict([('epsabs', 1.49e-2),
                                                                       ('epsrel', 10.00e-4),
                                                                       ('limit', 2)]))[0]
                    #stop if nan is detected
                    if np.isnan(T_coef_t[i, j, k]) or np.isnan(T_coef_s[i, j, k]):
                        print 'T_coef = nan detected'
                        print 'i, j, k = ',i,j,k
                        print ('T_coef_t[i, j, k] = ', (T_coef_t[i, j, k]))
                        print ('T_coef_s[i, j, k] = ', (T_coef_s[i, j, k]))
                        print
                        print 'x_comp = ',x_comp
                        print 'y_comp = ',y_comp
                        print 'x_coords = ',x_coords
                        print 'y_coords = ',y_coords
                        print 'reg = ',reg
                        print 'face.mfp.t[i, j] = ',face.mfp.t[i, j]
                        print 'adj_cells[0] = ',adj_cells[0]
                        print 'adj_cells[k-j] = ',adj_cells[k-j]
                        sys.exit()
                    else:
                        pass

                    trans_coef_file.write(('{:>6d}'*3+'{:>12.3E}'*4+'\n').format(int(T_from[i, j, k]),
                                                                                 int(T_to[i, j, k]),
                                                                                 int(T_via[i, j, k]),
                                                                                 T_coef_s[i, j, k],
                                                                                 T_coef_t[i, j, k],
                                                                                 face.mfp.s[i, j],
                                                                                 face.mfp.t[i, j]))

            print_progress(progress, outof)
        print '\n'
        trans_coef_file.close()

        # create t_coef_sum arrays for use later
        tcoef_sum_s = np.zeros((self.nCells, 4))
        tcoef_sum_t = np.zeros((self.nCells, 4))
        for i in range(0, self.nCells):
            for k in range(0, int(self.nSides[i])):
                tcoef_sum_s[i, k] = np.sum(T_coef_s[np.where((T_via == i) & (T_from == self.adjCell[i, k]))])
                tcoef_sum_t[i, k] = np.sum(T_coef_t[np.where((T_via == i) & (T_from == self.adjCell[i, k]))])

        T_coef_dict = {}
        T_coef_dict['s'] = T_coef_s
        T_coef_dict['t'] = T_coef_t
        T_coef_dict['from_cell'] = T_from
        T_coef_dict['to_cell'] = T_to
        T_coef_dict['via_cell'] = T_via
        T_coef_dict['sum_s'] = tcoef_sum_s
        T_coef_dict['sum_t'] = tcoef_sum_t
        T_coef = namedtuple('T_coef', T_coef_dict.keys())(*T_coef_dict.values())

        return T_coef

    def solve_matrix(self, face, cell, T_coef):
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

        # M_row = 0
        for i in range(0, self.nCells):
            for j in range(0, int(self.nSides[i])):
                for k in range(0, int(self.nSides[i])):
                    
                    curcell = int(i)
                    tocell = int(self.adjCell[i, j])
                    fromcell = int(self.adjCell[i, k])
        
                    # curcell_type = int(inp.iType[curcell])
                    # tocell_type = int(inp.iType[tocell])
                    fromcell_type = int(self.iType[fromcell])
        
                    T_coef_loc = np.where((T_coef.via_cell == curcell) &
                                          (T_coef.from_cell == fromcell) &
                                          (T_coef.to_cell == tocell))

                    face_loc = T_coef_loc[:2]

                    # FROM NORMAL CELL
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
                            uncoll_ss = T_coef.s[T_coef_loc]
                            uncoll_tt = T_coef.t[T_coef_loc]
        
                        coll_ss = 0 
                        coll_tt = (1-T_coef.sum_t[i, k]) * face.ci.t[i, k] * \
                                  (cell.P0i.t[i]*face.lfrac[i, j] + (1-cell.P0i.t[i])*cell.ci.t[i]*cell.Pi.t[i]*face.lfrac[i, j])
                        coll_ts = 0 
                        coll_st = (1-T_coef.sum_s[i, k]) * face.ci.s[i, k] * \
                                  (cell.P0i.t[i]*face.lfrac[i, j] + (1-cell.P0i.t[i])*cell.ci.t[i]*cell.Pi.t[i]*face.lfrac[i, j])
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

                    # FROM PLASMA CORE
                    elif fromcell_type == 1:  # if fromcell is plasma core cell
                        # column from, to swapped because incoming is defined as a function of outgoing via albedo condition
                        M_row_s = np.where((flux_pos[:, 1] == curcell) & (flux_pos[:, 2] == tocell))[0][0]
                        M_col_s = np.where((flux_pos[:, 1] == curcell) & (flux_pos[:, 2] == fromcell))[0][0]
                        M_row_t = M_row_s + num_fluxes
                        M_col_t = M_col_s + num_fluxes
        
                        # uncollided flux from slow group to slow group
                        if fromcell == tocell:
                            uncoll_ss = 0.0
                            uncoll_tt = 0.0
                        else:
                            uncoll_ss = face.alb.s[face_loc] * T_coef.s[T_coef_loc]
                            uncoll_tt = face.alb.t[face_loc] * T_coef.t[T_coef_loc]
                        
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
        
                    # FROM WALL
                    elif fromcell_type == 2:  # if fromcell is wall cell
                        # column from, to swapped because incoming is defined as a function of outgoing via reflection coefficient
                        M_row_s = np.where((flux_pos[:, 1] == curcell) & (flux_pos[:, 2] == tocell))[0][0]
                        M_col_s = np.where((flux_pos[:, 1] == curcell) & (flux_pos[:, 2] == fromcell))[0][0]
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
                            uncoll_refl_ss = face.refl.n.s[face_loc] * T_coef.s[T_coef_loc]
                            # thermal neutrals can hit the wall, reflect as thermals, and stream uncollided into adjacent cells
                            uncoll_refl_tt = face.refl.n.t[face_loc] * T_coef.t[T_coef_loc]
                            # thermal neutrals can reflect back as slow and then stay slow , but we'll treat this later
                            uncoll_refl_ts = 0
                            # slow neutrals can reflect back as slow neutrals, and end up as thermal, but that implies a collision
                            uncoll_refl_st = 0 
        
                            # slow neutrals can hit the wall, be reemitted as slow, and stream uncollided into adjacent cells
                            uncoll_reem_ss = (1.0 - face.refl.n.s[face_loc])*(1-face.f_abs[face_loc]) * T_coef.s[T_coef_loc]
                            # thermal neutrals can hit the wall, be reemitted as slow, and then end up thermal again, but that implies a collision.
                            uncoll_reem_tt = 0
                            # thermal neutrals can hit the wall, be reemitted as slow, and then stream uncollided into adjacent cells
                            uncoll_reem_ts = (1.0 - face.refl.n.t[face_loc])*(1-face.f_abs[face_loc]) * T_coef.s[T_coef_loc]
                            # slow neutrals can hit the wall, be reemitted as slow, and then end up thermal again, but that implies a collision.
                            uncoll_reem_st = 0 
        
                        # COLLIDED FLUX
                        # slow neutrals can hit the wall, reflect as slow neutrals, but a collision removes them from the slow group
                        coll_refl_ss = 0
                        # thermal neutrals can hit the wall, reflect as thermal neutrals, and then have a collision and stay thermal afterward
                        coll_refl_tt = face.refl.n.t[face_loc] * (1-T_coef.sum_t[i, k]) * \
                                       face.ci.t[i, k]*(cell.P0i.t[i]*face.lfrac[i, j] + (1-cell.P0i.t[i])*cell.ci.t[i]*cell.Pi.t[i]*face.lfrac[i, j])
                        # thermal neutrals can hit the wall, reflect as thermal neutrals, but they won't reenter the slow group
                        coll_refl_ts = 0
                        # slow neutrals can hit the wall, reflect as slow neutrals, and have a collision to enter and stay in the thermal group
        
                        coll_refl_st = face.refl.n.s[face_loc] * (1-T_coef.sum_s[i, k]) * \
                                       face.ci.s[i, k]*(cell.P0i.t[i]*face.lfrac[i, j] + (1-cell.P0i.t[i])*cell.ci.t[i]*cell.Pi.t[i]*face.lfrac[i, j])
        
                        
                        # slow neutrals can hit the wall, be reemitted as slow, but a collision removes them from the slow group
                        coll_reem_ss = 0

                        # print 'face.ci.s[i, k] = ',face.ci.s[i, k]
                        # print 'cell.P0i.t[i] = ',cell.P0i.t[i]
                        # print 'face.lfrac[i, j] = ',face.lfrac[i, j]
                        # print 'cell.ci.t[i] = ',cell.ci.t[i]
                        # print 'cell.Pi.t[i] = ',cell.Pi.t[i]
                        # print
                        # print
                        # print 'face_loc = ',face_loc
                        # print 'face.refl.n.s[face_loc] = ',face.refl.n.s[face_loc]
                        # print 'face.f_abs[face_loc] = ',face.f_abs[face_loc]
                        #
                        # print 'T_coef.sum_s[i, k] = ',T_coef.sum_s
                        #
                        # #sys.exit()
                        # print 'face.ci.s[i, k] = ',face.ci.s[i, k]
                        # print 'cell.P0i.t[i] = ',cell.P0i.t[i]
                        # print 'face.lfrac[i, j] = ',face.lfrac[i, j]
                        # print 'cell.P0i.t[i] = ',cell.P0i.t[i]
                        # print 'cell.ci.t[i] = ',cell.ci.t[i]
                        # print 'face.lfrac[i, j] = ',face.lfrac[i, j]

                        # thermal neutrals can hit the wall, be reemitted as slow, and then collide to enter and stay in the thermal group
                        coll_reem_tt = (1-face.refl.n.t[face_loc])*(1-face.f_abs[face_loc])*(1-T_coef.sum_s[i, k]) * \
                                       face.ci.s[i, k]*(cell.P0i.t[i]*face.lfrac[i, j] + (1-cell.P0i.t[i])*cell.ci.t[i]*cell.Pi.t[i]*face.lfrac[i, j])
                        # thermal neutrals can hit the wall, be reemitted as slow, but a collision removes them from the slow group
                        coll_reem_ts = 0

                        # slow neutrals can hit the wall, be reemitted as slow, and then collide to enter and stay in the thermal group
                        coll_reem_st = (1-face.refl.n.s[face_loc])*(1-face.f_abs[face_loc])*(1-T_coef.sum_s[i, k]) * \
                                       face.ci.s[i, k]*(cell.P0i.t[i]*face.lfrac[i, j] + (1-cell.P0i.t[i])*cell.ci.t[i]*cell.Pi.t[i]*face.lfrac[i, j])
                        
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
        flux_cells = self.adjCell[np.where(face.s_ext > 0)]
        flux_vals = face.s_ext[np.where(face.s_ext > 0)]
        
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
                    
                    T_coef_loc = np.where((T_coef.via_cell == cell_io) &
                                          (T_coef.from_cell == cell_from) &
                                          (T_coef.to_cell == cell_to))
                    face_from_loc = T_coef_loc[:2]
                    face_to_loc = [T_coef_loc[0], T_coef_loc[-1]]
        
                    # add uncollided slow flux from slow external source
                    source[i] = source[i] + incoming_flux * T_coef.s[T_coef_loc][0]
                    # add collided slow flux from slow external source
                    source[i] = source[i] + incoming_flux * 0
                    # add collided thermal flux from slow external source
                    source[i+num_fluxes] = source[i+num_fluxes] + incoming_flux * (1-T_coef.sum_s[face_from_loc]) * \
                                           face.ci.s[face_from_loc]*(cell.P0i.t[cell_io]*face.lfrac[face_to_loc] + (1-cell.P0i.t[cell_io])*cell.ci.t[cell_io]*cell.Pi.t[cell_io]*face.lfrac[face_to_loc])
        
            if group == 1:
                # ADD CONTRIBUTION FROM VOLUMETRIC SOURCE (I.E. RECOMBINATION)
                source[i] = source[i] + 0  # cell_area[cell_io]*cell_ni[cell_io]*cell_sv_rec[cell_io]*P_i_t[cell_io]*0.25 # assumed that all neutrals from recombination are thermal

                # ADD CONTRIBUTION FROM IMPINGING IONS REFLECTING AS THERMAL NEUTRALS
                # loop over "from cells and determine if any are wall cells
                for count, cell_from in enumerate(self.adjCell[cell_io, :self.nSides[cell_io]]):
                    # if it's a wall cell
                    if self.iType[cell_from] == 2:
                        
                        # calculate flux to wall segment
                        ni = cell.n.i[cell_io]
                        Ti = cell.T.i[cell_io] * 1.0E3 * 1.6021E-19
                        vel = sqrt(2.0*Ti/3.343583719E-27)
                        wall_length = face.lside[cell_io, count]
                        R0_wall = 1.4 # TODO: Read in R values for each wall segment
                        flux2wall = 2.0*pi*ni*vel*wall_length*R0_wall
                        
                        # calculate returning neutral flux
                        refl_coef = face.refl.n.t[cell_io, count]
                        incoming_flux = flux2wall * refl_coef
                        
                        # calculate uncollided source to cell_to
                        T_coef_loc = np.where((T_coef.via_cell == cell_io) & (T_coef.from_cell == cell_from) & (T_coef.to_cell == cell_to))
                        face_from_loc = T_coef_loc[:2]
                        face_to_loc = [T_coef_loc[0], T_coef_loc[-1]]

                        source[i] = source[i] + incoming_flux * T_coef.t[T_coef_loc]
                        
                        # calculate collided source to cell_to
                        source[i] = source[i] + incoming_flux * (1.0-T_coef.sum_t[face_from_loc]) * \
                                    face.ci.t[face_from_loc]*(cell.P0i.t[cell_io]*face.lfrac[face_to_loc] + (1.0-cell.P0i.t[cell_io])*cell.ci.t[cell_io]*cell.Pi.t[cell_io]*face.lfrac[face_to_loc])
                        if incoming_flux < 0:
                            print 'incoming flux less than zero'
                            print 'stopping'
                            sys.exit()

        # CREATE FINAL MATRIX AND SOLVE
        if m_sparse == 0 or m_sparse == 2:
            M_matrix = np.identity(M_size) - M_matrix
            flux_out = spsolve(M_matrix, source)
        if m_sparse == 1 or m_sparse == 2:
            # multiply m_data by -1 and append "identify matrix"
            # note: we're taking advantage of the way coo_matrix handles duplicate
            # entries to achieve the same thing as I - M
            m_row = np.concatenate((np.asarray(m_row), np.arange(0, M_size)))
            m_col = np.concatenate((np.asarray(m_col), np.arange(0, M_size)))
            m_data = np.concatenate((-np.asarray(m_data), np.ones(M_size)))
            m_sp_final = coo_matrix((m_data, (m_row, m_col))).tocsc()
            flux_out = spsolve(m_sp_final, source)

        # create outgoing flux arrays, dictionary, and namedtuple
        flux_out_s = np.zeros((self.nCells, 4))
        flux_out_t = np.zeros((self.nCells, 4))
        flux_counter = 0
        for g, g1 in enumerate(range(num_en_groups)):
            for i, v1 in enumerate(range(self.nCells)):
                for j, v2 in enumerate(range(self.nSides[i])):
                    if g == 0:
                        flux_out_s[i, j] = flux_out[flux_counter]
                    if g == 1:
                        flux_out_t[i, j] = flux_out[flux_counter]
                    flux_counter += 1

        flux_out_dict = {}
        flux_out_dict['s'] = flux_out_s
        flux_out_dict['t'] = flux_out_t
        flux_out_dict['tot'] = flux_out_s + flux_out_t
        flux_out = namedtuple('flux_out', flux_out_dict.keys())(*flux_out_dict.values())

        # create incoming flux arrays, dictionary, and namedtuple
        flux_inc_s = np.zeros(flux_out_s.shape)
        flux_inc_t = np.zeros(flux_out_t.shape)

        flux_out_dest = self.adjCell[:self.nCells, :]
        flux_out_src = flux_out_dest * 0
        for i in range(self.nCells):
            flux_out_src[i] = i

        for i in range(0, self.nCells):
            for k in range(0, self.nSides[i]):
                if face.adj.int_type[i, k] == 0:
                    flux_inc_s[i, k] = flux_out_s[np.where((flux_out_dest == i) & (flux_out_src == self.adjCell[i, k]))]
                    flux_inc_t[i, k] = flux_out_t[np.where((flux_out_dest == i) & (flux_out_src == self.adjCell[i, k]))]
                if face.adj.int_type[i, k] == 1:
                    flux_inc_s[i, k] = flux_out_s[i, k] * face.alb.s[i, k]
                    flux_inc_t[i, k] = flux_out_t[i, k] * face.alb.t[i, k]
                    flux_inc_s[i, k] = flux_inc_s[i, k] + face.s_ext[i, k]
                if face.adj.int_type[i, k] == 2:
                    flux_inc_s[i, k] = flux_out_s[i, k] * (
                            face.refl.n.s[i, k] + (1.0 - face.refl.n.s[i, k]) * (1 - face.f_abs[i, k])) + \
                                       flux_out_t[i, k] * (1.0 - face.refl.n.t[i, k]) * (1 - face.f_abs[i, k])
                    flux_inc_t[i, k] = flux_out_t[i, k] * face.refl.n.t[i, k]
                    flux_inc_s[i, k] = flux_inc_s[i, k] + face.s_ext[i, k]

        flux_inc_dict = {}
        flux_inc_dict['s'] = flux_inc_s
        flux_inc_dict['t'] = flux_inc_t
        flux_inc_dict['tot'] = flux_inc_s + flux_inc_t
        flux_inc = namedtuple('flux_inc', flux_inc_dict.keys())(*flux_inc_dict.values())

        flux_dict = {}
        flux_dict['inc'] = flux_inc
        flux_dict['out'] = flux_out
        flux = namedtuple('flux', flux_dict.keys())(*flux_dict.values())

        return flux

    def calc_neutral_dens(self, cell, face, T_coef, flux):
        # first calculate the fluxes into cells from the fluxes leaving the cells

        
        # calculate ionization rate
        cell_izn_rate_s = np.zeros(self.nCells)
        cell_izn_rate_t = np.zeros(self.nCells)
        for i in range(0, self.nCells):
            # add contribution to ionization rate from fluxes streaming into cell
            for k in range(0, self.nSides[i]):
                    cell_izn_rate_s[i] = cell_izn_rate_s[i] + flux.inc.s[i, k] * (1.0-T_coef.sum_s[i, k]) * \
                                         ((1.0-face.ci.s[i, k]) + face.ci.s[i, k]*(1.0-cell.ci.t[i])*(1.0-cell.P0i.t[i])/(1-cell.ci.t[i]*(1.0-cell.P0i.t[i])))
                                            
                    cell_izn_rate_t[i] = cell_izn_rate_t[i] + flux.inc.t[i, k] * (1.0-T_coef.sum_t[i, k]) * \
                                         ((1.0-face.ci.t[i, k]) + face.ci.t[i, k] * (1.0-cell.ci.t[i])*(1.0-cell.P0i.t[i])/(1-cell.ci.t[i]*(1.0-cell.P0i.t[i])))
                                          
            # add contribution to ionization rate from volumetric recombination within the cell
            cell_izn_rate_s[i] = cell_izn_rate_s[i] + 0 # all recombination neutrals are assumed to be thermal
            cell_izn_rate_t[i] = cell_izn_rate_t[i] + 0*(1-cell.P0i.t[i])*cell.area[i]*cell.n.i[i]*cell.sv.rec[i] * \
                                 (1.0 - cell.ci.t[i] + cell.ci.t[i]*(1.0-cell.ci.t[i])*(1-cell.P0i.t[i])/(1.0-cell.ci.t[i]*(cell.P0i.t[i])))
        
        cell_izn_rate = cell_izn_rate_s + cell_izn_rate_t
        
        # calculate neutral densities from ionization rates
        cell_nn_s = cell_izn_rate_s / (cell.n.i*cell.sv.ion*cell.area)
        cell_nn_t = cell_izn_rate_t / (cell.n.i*cell.sv.ion*cell.area)
        cell_nn = cell_izn_rate / (cell.n.i*cell.sv.ion*cell.area)
        
        # # fix negative values (usually won't be necessary)
        for i, v1 in enumerate(cell_nn):
            if v1<0:
                # get neutral densities of adjacent cells
                # average the positive values
                print i, 'Found a negative density. Fixing by using average of surrounding cells.'
                # print 'You probably need to adjust the way you are calculating the transmission coefficients.'
                nn_sum = 0
                nn_count = 0
                for j, v2 in enumerate(self.adjCell[i]):
                    if self.iType[v2] == 0:
                        if cell_nn[j] > 0:
                            nn_sum = nn_sum + cell_nn[j]
                            nn_count += 1
                cell_nn[i] = nn_sum / nn_count

        izn_rate_dict = {}
        izn_rate_dict['s'] = cell_izn_rate_s
        izn_rate_dict['t'] = cell_izn_rate_t
        izn_rate_dict['tot'] = cell_izn_rate
        izn_rate = namedtuple('izn_rate', izn_rate_dict.keys())(*izn_rate_dict.values())

        nn_dict = {}
        nn_dict['s'] = cell_nn_s
        nn_dict['t'] = cell_nn_t
        nn_dict['tot'] = cell_nn
        nn = namedtuple('nn', izn_rate_dict.keys())(*nn_dict.values())

        return izn_rate, nn


        # print 'Checking particle balance...'
        # relerr = abs(np.sum(flux_out_tot)+np.sum(cell_izn_rate)-np.sum(flux_in_tot))/ \
        # ((np.sum(flux_out_tot)+np.sum(cell_izn_rate)+np.sum(flux_in_tot))/2)
        # if relerr<0.001: # this is arbitrary. It's just here to alert if there's a problem in the particle balance.
        # print 'Particle balance passes. {5.3f}% relative error.'.format(relerr*100)
        # else:
        # print 'Particle balance failed. {}% relative error.'.format(relerr*100)
        # for i, (v1, v2) in enumerate(zip(flux_in_tot, flux_out_tot)):
        # print i, np.sum(v1), np.sum(v2)+cell_izn_rate[i]
        
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
