#!/usr/bin/python
# coding=utf-8

from __future__ import division
import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy import integrate
import sys
from math import pi
import os
import re
import pandas as pd
from math import sin, exp, sqrt, acos, degrees
from collections import namedtuple
from scipy.constants import physical_constants
from shapely.geometry import LineString, Point

m_p = physical_constants['proton mass'][0]

def isnamedtupleinstance(x):
    """

    :param x:
    :return:
    """
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

class NeutpyTools:

    def __init__(self, neut=None):

        # get vertices in R, Z geometry
        self.xs, self.ys = self.calc_cell_pts(neut)

        # localize densities, ionization rates, and a few other parameters that might be needed.
        self.n_n_slow = neut.nn.s
        self.n_n_thermal = neut.nn.t
        self.n_n_total = neut.nn.tot
        self.izn_rate_slow = neut.izn_rate.s
        self.izn_rate_thermal = neut.izn_rate.t
        self.izn_rate_total = neut.izn_rate.tot
        self.flux_in_s = neut.flux.inc.s
        self.flux_in_t = neut.flux.inc.t
        self.flux_in_tot = self.flux_in_s + self.flux_in_t
        self.flux_out_s = neut.flux.out.s
        self.flux_out_t = neut.flux.out.t
        self.flux_out_tot = self.flux_out_s + self.flux_out_t

        self.create_flux_outfile()
        self.create_cell_outfile()

        flux_s_xcomp, flux_s_ycomp, flux_s_mag = self.calc_flow('slow', norm=True)
        flux_t_xcomp, flux_t_ycomp, flux_t_mag = self.calc_flow('thermal', norm=True)
        flux_tot_xcomp, flux_tot_ycomp, flux_tot_mag = self.calc_flow('total', norm=True)

        self.vars = {}
        self.vars['n_n_slow'] = neut.nn.s
        self.vars['n_n_thermal'] = neut.nn.t
        self.vars['n_n_total'] = neut.nn.tot

        self.vars['flux_s_xcomp'] = flux_s_xcomp
        self.vars['flux_s_ycomp'] = flux_s_ycomp
        self.vars['flux_s_mag'] = flux_s_mag

        self.vars['flux_t_xcomp'] = flux_t_xcomp
        self.vars['flux_t_ycomp'] = flux_t_ycomp
        self.vars['flux_t_mag'] = flux_t_mag

        self.vars['flux_tot_xcomp'] = flux_tot_xcomp
        self.vars['flux_tot_ycomp'] = flux_tot_ycomp
        self.vars['flux_tot_mag'] = flux_tot_mag

        print 'attempting to start plot_cell_vals'
        self.plot_cell_vals()

    def create_flux_outfile(self):
        # create face output data file
        f = open('neutpy_face_data.dat', 'w')
        f.write(('{:^18s}' * 18).format('x1', 'x2', 'x3', 'y1', 'y2', 'y3',
                                        'flxout_s1', 'flxout_s2', 'flxout_s3',
                                        'flxin_s1', 'flxin_s2', 'flxin_s3',
                                        'flxout_t1', 'flxout_t2', 'flxout_t3',
                                        'flxin_t1', 'flxin_t2', 'flxin_t3'))
        for i, pt in enumerate(self.xs):
            f.write(('\n' + '{:>18.5f}' * 6 + '{:>18.5E}' * 12).format(
                self.xs[i, 0], self.xs[i, 1], self.xs[i, 2],
                self.ys[i, 0], self.ys[i, 1], self.ys[i, 2],
                self.flux_out_s[i, 0],
                self.flux_out_s[i, 1],
                self.flux_out_s[i, 2],
                self.flux_in_s[i, 0],
                self.flux_in_s[i, 1],
                self.flux_in_s[i, 2],
                self.flux_out_t[i, 0],
                self.flux_out_t[i, 1],
                self.flux_out_t[i, 2],
                self.flux_in_t[i, 0],
                self.flux_in_t[i, 1],
                self.flux_in_t[i, 2]))
        f.close()

    def create_cell_outfile(self):
        df = pd.DataFrame()
        df['R'] = pd.Series(np.mean(self.xs, axis=1), name='R')
        df['Z'] = pd.Series(np.mean(self.ys, axis=1), name='Z')
        df['n_n_slow'] = pd.Series(self.n_n_slow, name='n_n_slow')
        df['n_n_thermal'] = pd.Series(self.n_n_thermal, name='n_n_thermal')
        df['n_n_total'] = pd.Series(self.n_n_total, name='n_n_total')
        df['izn_rate_slow'] = pd.Series(self.izn_rate_slow, name='izn_rate_slow')
        df['izn_rate_thermal'] = pd.Series(self.izn_rate_thermal, name='izn_rate_thermal')
        df['izn_rate_total'] = pd.Series(self.izn_rate_thermal, name='izn_rate_total')
        #cell_df = iterate_namedtuple(neut.cell, df)
        df.to_csv(os.getcwd() + '/outputs/neutpy_cell_values.txt')

    @staticmethod
    def calc_cell_pts(neut):
        sys.setrecursionlimit(100000)

        def loop(neut, oldcell, curcell, cellscomplete, xcoords, ycoords):
            beta[curcell, :neut.nSides[curcell]] = np.cumsum(
                np.roll(neut.angles[curcell, :neut.nSides[curcell]], 1) - 180) + 180

            # if first cell:
            if oldcell == 0 and curcell == 0:
                # rotate cell by theta0 value (specified)
                beta[curcell, :neut.nSides[curcell]] = beta[curcell, :neut.nSides[curcell]] + neut.cell1_theta0
                x_comp = np.cos(np.radians(beta[curcell, :neut.nSides[curcell]])) * neut.lsides[curcell,
                                                                                    :neut.nSides[curcell]]
                y_comp = np.sin(np.radians(beta[curcell, :neut.nSides[curcell]])) * neut.lsides[curcell,
                                                                                    :neut.nSides[curcell]]
                xcoords[curcell, :neut.nSides[curcell]] = np.roll(np.cumsum(x_comp), 1) + neut.cell1_ctr_x
                ycoords[curcell, :neut.nSides[curcell]] = np.roll(np.cumsum(y_comp), 1) + neut.cell1_ctr_y

            # for all other cells:
            else:

                # adjust all values in beta for current cell such that the side shared
                # with oldcell has the same beta as the oldcell side
                oldcell_beta = beta[oldcell, :][np.where(neut.adjCell[oldcell, :] == curcell)][0]
                delta_beta = beta[curcell, np.where(neut.adjCell[curcell, :] == oldcell)] + 180 - oldcell_beta
                beta[curcell, :neut.nSides[curcell]] = beta[curcell, :neut.nSides[curcell]] - delta_beta

                # calculate non-shifted x- and y- coordinates
                x_comp = np.cos(np.radians(beta[curcell, :neut.nSides[curcell]])) * neut.lsides[curcell,
                                                                                    :neut.nSides[curcell]]
                y_comp = np.sin(np.radians(beta[curcell, :neut.nSides[curcell]])) * neut.lsides[curcell,
                                                                                    :neut.nSides[curcell]]
                xcoords[curcell, :neut.nSides[curcell]] = np.roll(np.cumsum(x_comp),
                                                                  1)  # xcoords[oldcell,np.where(neut.adjCell[oldcell,:]==curcell)[0][0]]
                ycoords[curcell, :neut.nSides[curcell]] = np.roll(np.cumsum(y_comp),
                                                                  1)  # ycoords[oldcell,np.where(neut.adjCell[oldcell,:]==curcell)[0][0]]

                cur_in_old = np.where(neut.adjCell[oldcell, :] == curcell)[0][0]
                old_in_cur = np.where(neut.adjCell[curcell, :] == oldcell)[0][0]
                mdpt_old_x = (xcoords[oldcell, cur_in_old] + np.roll(xcoords[oldcell, :], -1)[cur_in_old]) / 2
                mdpt_old_y = (ycoords[oldcell, cur_in_old] + np.roll(ycoords[oldcell, :], -1)[cur_in_old]) / 2
                mdpt_cur_x = (xcoords[curcell, old_in_cur] + np.roll(xcoords[curcell, :], -1)[old_in_cur]) / 2
                mdpt_cur_y = (ycoords[curcell, old_in_cur] + np.roll(ycoords[curcell, :], -1)[old_in_cur]) / 2

                xshift = mdpt_old_x - mdpt_cur_x
                yshift = mdpt_old_y - mdpt_cur_y

                xcoords[curcell, :] = xcoords[curcell,
                                      :] + xshift  # xcoords[oldcell,np.where(neut.adjCell[oldcell,:]==curcell)[0][0]]
                ycoords[curcell, :] = ycoords[curcell,
                                      :] + yshift  # ycoords[oldcell,np.where(neut.adjCell[oldcell,:]==curcell)[0][0]]

            # continue looping through adjacent cells
            for j, newcell in enumerate(neut.adjCell[curcell, :neut.nSides[curcell]]):
                # if the cell under consideration is a normal cell (>3 sides) and not complete, then move into that cell and continue
                if neut.nSides[newcell] >= 3 and cellscomplete[newcell] == 0:
                    cellscomplete[newcell] = 1
                    loop(neut, curcell, newcell, cellscomplete, xcoords, ycoords)

            return xcoords, ycoords

        xcoords = np.zeros(neut.adjCell.shape)
        ycoords = np.zeros(neut.adjCell.shape)
        beta = np.zeros(neut.adjCell.shape)  # beta is the angle of each side with respect to the +x axis.

        ## Add initial cell to the list of cells that are complete
        cellscomplete = np.zeros(neut.nCells)
        cellscomplete[0] = 1
        xs, ys = loop(neut, 0, 0, cellscomplete, xcoords, ycoords)
        return xs, ys

    def plot_cell_lines(self, dir=''):

        # plot cell diagram
        # grid = plt.figure(figsize=(160,240))
        grid = plt.figure(figsize=(8, 12))
        ax1 = grid.add_subplot(111)
        ax1.set_title('Neutrals Mesh', fontsize=30)
        ax1.set_ylabel(r'Z ($m$)', fontsize=30)
        ax1.set_xlabel(r'R ($m$)', fontsize=30)
        ax1.tick_params(labelsize=15)
        ax1.axis('equal')
        for i, (v1, v2) in enumerate(zip(self.xs, self.ys)):
            # if neut.nSides[i] == len(v1):
            v1 = np.append(v1, v1[0])
            v2 = np.append(v2, v2[0])
            # elif neut.nSides[i] == 3 and len(v1) == 4:
            #    v1[3] = v1[0]
            # elif neut.nSides[i]==3 and len(v1)==3:
            #    v1 = np.append(v1,v1[0])
            #    v2 = np.append(v2,v2[0])
            ax1.plot(v1, v2, color='black', lw=1)
        grid.savefig(dir+'neutpy_mesh.png', dpi=300, transparent=True, bbox_inches="tight")

    def plot_cell_vals(self, title='n_n_total', dir=os.getcwd(), nSides=None, logscale=False, cmap='viridis'):
        print 'beginning plot_cell_cals'
        var = self.n_n_total

        if logscale:
            colors = np.log10(var)
        else:
            colors = var

        patches = []
        for i,v in enumerate(var):
            if nSides is not None:
                verts = np.column_stack((self.xs[i, :nSides[i]], self.ys[i, :nSides[i]]))
            else:
                verts = np.column_stack((self.xs[i, :3], self.ys[i, :3]))

            polygon = Polygon(verts, closed=True)
            patches.append(polygon)

        collection1 = PatchCollection(patches, cmap=cmap)
        collection1.set_array(np.array(colors))

        fig, ax1 = plt.subplots(figsize=(8, 12))
        cax = ax1.add_collection(collection1)
        ax1.axis('equal')
        ax1.set_title(title, fontsize=30)
        ax1.set_ylabel(r'Z ($m$)', fontsize=30)
        ax1.set_xlabel(r'R ($m$)', fontsize=30)
        ax1.tick_params(labelsize=30)
        cb = fig.colorbar(cax)
        cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=30)
        print 'attempting to show plot'
        plt.show()
        # print 'attempting to save plot in '+dir+'/'+title+'.png'
        # fig.savefig('figure.png', dpi=300, transparent=True, bbox_inches="tight")

    def interp_RZ(self, var):
        x = np.average(self.xs, axis=1)
        y = np.average(self.ys, axis=1)
        d = self.vars[var]
        return Rbf(x, y, d)

    def calc_flow(self, ntrl_pop='tot', norm=True):
        """Creates interpolation functions for net flux directions and magnitudes

           flux_in: fluxes coming into cells. Can be slow, thermal, total or any other fluxes. Array of size (nCells, 3)
           flux_in: fluxes leaving cells. Can be slow, thermal, total or any other fluxes. Array of size (nCells, 3)
           norm: returns normalized x- and y-component interpolation functions. Useful for plotting quiver plots
                    with equally sized arrows or if you only care about the direction of the flux (You can still get
                    the magnitude from "flux_net_av_mag" interpolation object.)
            """

        if ntrl_pop is 'slow':
            flux_in = self.flux_in_s[:, :-1]
            flux_out = self.flux_out_s[:, :-1]
        elif ntrl_pop is 'thermal':
            flux_in = self.flux_in_t[:, :-1]
            flux_out = self.flux_out_t[:, :-1]
        elif ntrl_pop is 'total':
            flux_in = self.flux_in_tot[:, :-1]
            flux_out = self.flux_out_tot[:, :-1]

        flux_net = flux_out - flux_in

        cent_pts_x = np.average(self.xs, axis=1)
        cent_pts_y = np.average(self.ys, axis=1)

        x_comp = np.roll(self.xs, -1, axis=1) - self.xs
        y_comp = np.roll(self.ys, -1, axis=1) - self.ys
        lside = np.sqrt(x_comp**2 + y_comp**2)
        perim = np.sum(lside, axis=1).reshape((-1, 1))
        l_frac = lside / perim

        side_angles = np.arctan2(y_comp, x_comp)
        side_angles = np.where(side_angles < 0, side_angles+2*pi, side_angles)

        outwd_nrmls = side_angles + pi/2
        outwd_nrmls = np.where(outwd_nrmls < 0, outwd_nrmls+2*pi, outwd_nrmls)
        outwd_nrmls = np.where(outwd_nrmls >= 2*pi, outwd_nrmls-2*pi, outwd_nrmls)

        flux_net_dir = np.where(flux_net < 0, outwd_nrmls+pi, outwd_nrmls)
        flux_net_dir = np.where(flux_net_dir < 0, flux_net_dir+2*pi, flux_net_dir)
        flux_net_dir = np.where(flux_net_dir >= 2*pi, flux_net_dir-2*pi, flux_net_dir)

        # x- and y-component of fluxes
        flux_net_xcomp = np.abs(flux_net)*np.cos(flux_net_dir)
        flux_net_ycomp = np.abs(flux_net)*np.sin(flux_net_dir)

        # side-length weighted average of x- and y-components of flux
        flux_net_xcomp_av = np.sum(flux_net_xcomp * l_frac, axis=1)
        flux_net_ycomp_av = np.sum(flux_net_ycomp * l_frac, axis=1)

        # normalized x- and y-components
        flux_net_xcomp_av_norm = flux_net_xcomp_av / np.sqrt(flux_net_xcomp_av**2 + flux_net_ycomp_av**2)
        flux_net_ycomp_av_norm = flux_net_ycomp_av / np.sqrt(flux_net_xcomp_av**2 + flux_net_ycomp_av**2)

        # side-length weighted average flux magnitude
        flux_net_av_mag = np.sqrt(flux_net_xcomp_av**2 + flux_net_ycomp_av**2)

        # create averaged net x- and y-component interpolation functions
        if norm:
            flux_xcomp = flux_net_xcomp_av_norm
            flux_ycomp = flux_net_ycomp_av_norm
        else:
            flux_xcomp = flux_net_xcomp_av
            flux_ycomp = flux_net_ycomp_av

        return flux_xcomp, flux_ycomp, flux_net_av_mag

    def common_plots(self):
        pass


class RgxToVal(object):

    def __getattr__(self, item):
        return self.get(item)

    def __setattr__(self, key, value):
        self[key] = value

    def __init__(self, obj, regex, str):
        self.regex = regex
        self.type = obj
        self.string = str

    def __call__(self, line, *args, **kwargs):
        result = re.match(self.regex, line)
        if not result:
            pass
        else:
            self.value = result.group(1)




