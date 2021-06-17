#!/usr/bin/python

"""
Various physics-ish functions for neutpy
"""


import numpy as np
from scipy import integrate
from math import pi
from math import sin, exp, sqrt
from collections import namedtuple
from scipy.constants import physical_constants

m_p = physical_constants['proton mass'][0]

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
        print('an NAN was found in the '+array_type+' '+en_grp+' mfp array')
        print('indices:')
        print(nan_locs)
        print()
        print('vn at those indices')
        print(vn[nan_locs])
        print()
        print('ne at those indices')
        print(ne[nan_locs])
        print()
        print('ni at those indices')
        print(svion[nan_locs])
        print()
        print('svion at those indices')
        print(vn[nan_locs])
        print()
        print('svcx at those indices')
        print(svcx[nan_locs])
        print()
        print('svel at those indices')
        print(svel[nan_locs])
        print()
        print('mfp array')
        print(mfp)
        print('stopping.')
        raise


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
    refle = namedtuple('refle', list(refle_dict.keys()))(*list(refle_dict.values()))

    refln_dict = {}
    refln_dict['s'] = refle_s
    refln_dict['t'] = refle_t
    refln = namedtuple('refln', list(refln_dict.keys()))(*list(refln_dict.values()))

    refl_dict = {}
    refl_dict['e'] = refle
    refl_dict['n'] = refln
    refl = namedtuple('refl', list(refl_dict.keys()))(*list(refl_dict.values()))

    alb_dict = {}
    alb_dict['s'] = alb_s
    alb_dict['t'] = alb_t
    alb = namedtuple('alb', list(alb_dict.keys()))(*list(alb_dict.values()))

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